"""
Faithful CPU simulation of softmax_kernel_2pass index/mask arithmetic on a FLAT
buffer, so we can prove the mask bug (and its fix) without a GPU.

We model exactly what the Triton kernel does per program:
  - a flat input buffer  i_flat  of length ROW_NUMS*COL_NUMS
  - a flat output buffer o_flat  (empty_like -> filled with garbage sentinel)
  - masked gather/scatter with out-of-bounds guarded (Triton masks OOB lanes;
    an UNMASKED OOB lane is undefined -> here it reads adjacent rows / would be
    a fault on the real device, which is exactly the danger).
"""
import numpy as np

def ref_softmax(x):
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)

def run_kernel(x, COL_BLOCKS, mask_mode):
    ROW_NUMS, COL_NUMS = x.shape
    i_flat = x.reshape(-1).astype(np.float32)
    N = i_flat.size
    o_flat = np.full(N, np.nan, np.float32)          # empty_like -> undefined
    col_off = np.arange(0, COL_BLOCKS)

    def gather(base, mask):
        idx = base + col_off
        out = np.full(COL_BLOCKS, -np.inf, np.float32)
        ok = mask & (idx >= 0) & (idx < N)           # unmasked+OOB would fault
        out[ok] = i_flat[idx[ok]]
        return out

    def scatter(base, vals, mask):
        idx = base + col_off
        ok = mask & (idx >= 0) & (idx < N)
        o_flat[idx[ok]] = vals[ok]

    for row in range(ROW_NUMS):
        m = -np.inf
        l = 0.0
        cols = list(range(0, COL_NUMS, COL_BLOCKS))
        for col in cols:
            if mask_mode == "buggy":
                mask = col_off < ROW_NUMS            # the bug on line 27
            else:
                mask = (col + col_off) < COL_NUMS     # the fix
            i = gather(row * COL_NUMS + col, mask)
            m_old = m
            m = max(m, i.max())
            l = l * np.exp(m_old - m) + np.exp(i - m).sum()
        for col in cols:
            if mask_mode == "buggy":
                mask = col_off < ROW_NUMS
            else:
                mask = (col + col_off) < COL_NUMS
            i = gather(row * COL_NUMS + col, mask)
            o = np.exp(i - m) / l
            scatter(row * COL_NUMS + col, o, mask)
    return o_flat.reshape(ROW_NUMS, COL_NUMS)

def check(name, ROW_NUMS, COL_NUMS, COL_BLOCKS):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((ROW_NUMS, COL_NUMS)).astype(np.float32)
    ref = ref_softmax(x)
    print(f"\n=== {name}: shape=({ROW_NUMS},{COL_NUMS}) COL_BLOCKS={COL_BLOCKS} "
          f"blocks/row={len(range(0,COL_NUMS,COL_BLOCKS))} ===")
    for mode in ("buggy", "fixed"):
        o = run_kernel(x, COL_BLOCKS, mode)
        row_sums = o.sum(axis=1)
        max_err = np.nanmax(np.abs(o - ref))
        ok = np.allclose(np.nan_to_num(o), ref, atol=1e-5)
        print(f"  {mode:5s}: max|err|={max_err:.3e}  rowsum[min,max]="
              f"[{np.nanmin(row_sums):.4f},{np.nanmax(row_sums):.4f}]  "
              f"nan={np.isnan(o).any()}  allclose={ok}")

if __name__ == "__main__":
    # Case A mirrors softmax_test(): COL_BLOCKS > COL_NUMS -> 1 block, all-True mask
    check("test-like", 64, 40, 8192)
    # Case B: multiple blocks per row -> exercises the online/blocked path
    check("multi-block", 32, 300, 64)
