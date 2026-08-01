#!/usr/bin/env python3
"""
reg_spill_probe.py — find the register/spill "knee" of the single-row softmax kernel.

Background
----------
softmax.py loads a whole row as ONE tile of next_power_of_2(N) elements and keeps
it in registers, doing max -> sub -> exp -> sum -> div entirely on-chip. That is
what makes it a clean single-pass, 2MN-DRAM-traffic kernel (~82% of H20 HBM BW).

But that only holds while the tile fits in the register file. Each thread must
hold `BLOCK / (num_warps*32)` elements. Once that exceeds what ptxas can keep in
the 255-register/thread budget, ptxas spills to DRAM-backed "local memory":
  * the tidy 2MN traffic silently degrades into scattered extra DRAM I/O, and
  * register pressure crushes occupancy, so latency can no longer be hidden.

This script locates that knee WITHOUT a GPU: it compiles the kernel with Triton,
then assembles the generated PTX with `ptxas -v` and reads registers-per-thread
and spill bytes straight from the assembler. It sweeps the tile width BLOCK for
num_warps in {4,8,16,32} to show the knee move RIGHT as warps (threads) increase
— because the knee sits at a roughly constant elements-per-thread, and
elements-per-thread = BLOCK / (num_warps*32).

Hopper (H20 = sm_90a) facts used below:
  * hard cap 255 registers per thread (architectural)
  * 65536 32-bit registers per SM (shared by all resident threads)
  * Triton hard limit TRITON_MAX_TENSOR_NUMEL = 1048576 elements per tile

Usage
-----
    python reg_spill_probe.py                       # H20 defaults
    python reg_spill_probe.py --arch sm_80          # other archs (cap auto-parsed)
    python reg_spill_probe.py --warps 4 8 16 32
    python reg_spill_probe.py --elems 32 64 128 256 512
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile

REGS_PER_SM = 65536       # Hopper 32-bit registers per SM
MAX_REGS_PER_THREAD = 255  # architectural hard cap


@triton.jit
def _row_softmax(i_ptr, o_ptr, COL_NUMS: tl.constexpr, BLOCK: tl.constexpr):
    # Mirrors the per-row body of softmax_kernel in softmax.py: one register-resident tile.
    col = tl.arange(0, BLOCK)
    mask = col < COL_NUMS
    x = tl.load(i_ptr + col, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(o_ptr + col, e / tl.sum(e, axis=0), mask=mask)


def _find_ptxas():
    for cand in (shutil.which("ptxas"), "/usr/local/cuda/bin/ptxas"):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("ptxas not found (need CUDA toolkit on PATH or /usr/local/cuda)")


def compile_and_measure(block, num_warps, cap, ptxas, cache_base):
    """Compile the tile, assemble its PTX, return (regs, spill_store, spill_load)."""
    cache = os.path.join(cache_base, f"tc_{block}_{num_warps}")
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = cache

    src = ASTSource(
        fn=_row_softmax,
        signature={"i_ptr": "*fp32", "o_ptr": "*fp32", "COL_NUMS": "constexpr", "BLOCK": "constexpr"},
        constexprs={"COL_NUMS": block, "BLOCK": block},
    )
    kernel = compile(src, target=GPUTarget("cuda", cap, 32), options={"num_warps": num_warps})

    # Locate the generated PTX and read the exact .target so ptxas arch always matches
    # (Triton emits sm_90a for Hopper, which ptxas -arch=sm_90 would reject).
    ptx_path = None
    for root, _dirs, files in os.walk(cache):
        for f in files:
            if f.endswith(".ptx"):
                ptx_path = os.path.join(root, f)
    if ptx_path is None:
        raise RuntimeError(f"no .ptx produced for BLOCK={block}, warps={num_warps}")

    with open(ptx_path) as fh:
        ptx_text = fh.read()
    m = re.search(r"\.target\s+(sm_\w+)", ptx_text)
    ptxas_arch = m.group(1) if m else f"sm_{cap}"

    out = subprocess.run(
        [ptxas, f"-arch={ptxas_arch}", "-v", ptx_path, "-o", os.devnull],
        capture_output=True, text=True,
    ).stderr

    regs = re.search(r"Used (\d+) registers", out)
    ss = re.search(r"(\d+) bytes spill stores", out)
    sl = re.search(r"(\d+) bytes spill loads", out)
    return (
        int(regs.group(1)) if regs else -1,
        int(ss.group(1)) if ss else 0,
        int(sl.group(1)) if sl else 0,
    )


def approx_blocks_per_sm(regs, threads):
    """Rough register-limited resident blocks per SM (illustrates occupancy collapse)."""
    if regs <= 0:
        return "?"
    return REGS_PER_SM // max(1, regs * threads)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sm_90a", help="target arch (default sm_90a = H20); cap auto-parsed")
    ap.add_argument("--warps", type=int, nargs="+", default=[4, 8, 16, 32], help="num_warps values to sweep")
    ap.add_argument("--elems", type=int, nargs="+", default=[32, 64, 128, 256, 512],
                    help="elements-per-thread points to probe (BLOCK = elems * warps * 32)")
    args = ap.parse_args()

    cap = int(re.sub(r"\D", "", args.arch))  # sm_90a -> 90, sm_80 -> 80
    ptxas = _find_ptxas()
    cache_base = tempfile.mkdtemp(prefix="regprobe_")

    print(f"arch={args.arch} (cap {cap})   register file: {REGS_PER_SM}/SM, "
          f"hard cap {MAX_REGS_PER_THREAD}/thread   Triton tile limit: 1048576 elems\n")
    print("Probing BLOCK = elems_per_thread * (num_warps*32). Knee = first nonzero spill.\n")

    knees = {}
    try:
        for w in args.warps:
            threads = w * 32
            print(f"=== num_warps={w}  ({threads} threads) " + "=" * 40)
            print(f"{'BLOCK':>8} {'elems/thr':>10} {'regs/thr':>9} {'spill_st':>9} "
                  f"{'spill_ld':>9} {'~blk/SM':>8}  note")
            knee = None
            for ept in args.elems:
                block = ept * threads
                if block > 1048576:
                    print(f"{block:>8} {ept:>10} {'—':>9} {'—':>9} {'—':>9} {'—':>8}  "
                          f"exceeds TRITON_MAX_TENSOR_NUMEL -> CompilationError")
                    continue
                regs, ss, sl = compile_and_measure(block, w, cap, ptxas, cache_base)
                spilled = ss > 0 or sl > 0
                note = ""
                if regs >= MAX_REGS_PER_THREAD:
                    note = "reg-capped"
                if spilled and knee is None:
                    knee = (block, ept)
                    note = (note + "  <-- KNEE (spill starts)").strip()
                elif spilled:
                    note = (note + "  spilling").strip()
                print(f"{block:>8} {ept:>10} {regs:>9} {ss:>9} {sl:>9} "
                      f"{str(approx_blocks_per_sm(regs, threads)):>8}  {note}")
            knees[w] = knee
            print()

        print("=== summary: knee position vs num_warps " + "=" * 30)
        print(f"{'num_warps':>10} {'knee BLOCK':>12} {'elems/thr@knee':>16}")
        prev = None
        for w in args.warps:
            k = knees.get(w)
            if k is None:
                print(f"{w:>10} {'(no spill in range)':>12}")
                continue
            block, ept = k
            trend = ""
            if prev is not None:
                trend = f"  (x{block // prev}) BLOCK vs prev warp count" if block >= prev else ""
            print(f"{w:>10} {block:>12} {ept:>16}{trend}")
            prev = block
        print("\nTakeaway: the knee sits near a constant elems/thread; doubling warps"
              "\nroughly doubles the BLOCK you can keep register-resident — but the 255"
              "\nreg/thread cap and the 1048576-elem tile limit are the absolute ceilings."
              "\nPast the knee, switch softmax.py to the online 2-pass blocked kernel.")
    finally:
        shutil.rmtree(cache_base, ignore_errors=True)


if __name__ == "__main__":
    main()
