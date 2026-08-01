import triton
import triton.language as tl
import torch
import re
import os
import shutil

if os.path.isdir("~/.triton/cache"):
    shutil.rmtree("~/.triton/cache")
if os.path.isdir("~/.cache/triton"):
    shutil.rmtree("~/.cache/triton")
os.environ["TRITON_CACHE_DIR"] = "__triton_cache__"
if os.path.isdir(os.environ["TRITON_CACHE_DIR"]):
    shutil.rmtree(os.environ["TRITON_CACHE_DIR"])


@triton.autotune(
    configs=[
        triton.Config({'M_BLOCK': 128, 'N_BLOCK': 256, 'K_BLOCK': 64}, num_stages=1, num_warps=4),
        triton.Config({'M_BLOCK': 128, 'N_BLOCK': 256, 'K_BLOCK': 64}, num_stages=4, num_warps=8), # import performance gain
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_naive(
    m_ptr, n_ptr, o_ptr, 
    M, N, K, 
    M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr, K_BLOCK: tl.constexpr
):
    # one program computes one [M_BLOCK, N_BLOCK] tile of o = m @ n
    row = tl.program_id(axis=0) * M_BLOCK
    col = tl.program_id(axis=1) * N_BLOCK

    m_row_off = row + tl.arange(0, M_BLOCK)[:, None]   # [M_BLOCK, 1] rows of m / o
    n_col_off = col + tl.arange(0, N_BLOCK)[None, :]   # [1, N_BLOCK] cols of n / o

    acc = tl.zeros((M_BLOCK, N_BLOCK), dtype=tl.float32)
    for k in tl.range(0, K, K_BLOCK):
        m_col_off = k + tl.arange(0, K_BLOCK)[None, :]  # [1, K_BLOCK] cols of m
        n_row_off = k + tl.arange(0, K_BLOCK)[:, None]  # [K_BLOCK, 1] rows of n

        m_off = m_row_off * K + m_col_off               # [M_BLOCK, K_BLOCK]
        m_mas = (m_row_off < M) & (m_col_off < K)
        m = tl.load(m_ptr + m_off, mask=m_mas, other=0.0)

        n_off = n_row_off * N + n_col_off               # [K_BLOCK, N_BLOCK]
        n_mas = (n_row_off < K) & (n_col_off < N)
        n = tl.load(n_ptr + n_off, mask=n_mas, other=0.0)

        acc = tl.dot(m, n, acc)
        # acc += tl.dot(m, n, input_precision="ieee")

    c = acc.to(tl.float16)
    o_off = m_row_off * N + n_col_off                   # [M_BLOCK, N_BLOCK]
    o_mas = (m_row_off < M) & (n_col_off < N)
    tl.store(o_ptr + o_off, c, mask=o_mas)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)
    ],
    key=['M', 'N', 'K', 'ENABLE_GROUP_ORDERING'],
)
@triton.jit
def matmul_kernel_swizzle(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ENABLE_GROUP_ORDERING: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if ENABLE_GROUP_ORDERING:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_op(m, n, kernel_type):
    M, K1 = m.shape
    K2, N = n.shape
    assert K1 == K2, f"incompatible dims: {tuple(m.shape)} @ {tuple(n.shape)}"
    K = K1
    o = torch.empty((M, N), dtype=m.dtype, device=m.device)

    if kernel_type == "triton1":
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel_swizzle[grid](
            m, n, o,  #
            M, N, K,  #
            m.stride(0), m.stride(1),  #
            n.stride(0), n.stride(1),  #
            o.stride(0), o.stride(1),  #
            ENABLE_GROUP_ORDERING = True
        )
    elif kernel_type == "triton2":
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel_swizzle[grid](
            m, n, o,  #
            M, N, K,  #
            m.stride(0), m.stride(1),  #
            n.stride(0), n.stride(1),  #
            o.stride(0), o.stride(1),  #
            ENABLE_GROUP_ORDERING = False
        )
    elif kernel_type == "triton3":
        GRID = lambda META: ((M + META['M_BLOCK'] - 1) // META['M_BLOCK'], (N + META['N_BLOCK'] - 1) // META['N_BLOCK'])
        matmul_kernel_naive[GRID](
            m, n, o, 
            M, N, K,
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return o


def matmul_test():
    torch.manual_seed(0)
    M = 1024
    K = 2048
    N = 1024
    m = torch.randn(M, K, dtype=torch.float16, device='cuda:0')
    n = torch.randn(K, N, dtype=torch.float16, device='cuda:0')
    o_ref = torch.matmul(m, n)
    print(o_ref)
    o = matmul_op(m, n, kernel_type="triton1")
    print(o)
    is_close = torch.allclose(o, o_ref, atol=1e-2, rtol=0)
    print(f"is_close: {is_close}")
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(o - o_ref))}')


def matmul_bench():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["torch", "triton1", "triton2", "triton3"],  # Label name for the lines
            line_names=["Torch", "Triton1", "Triton2", "Triton3"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("black", "-"), ("red", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-fp16",
            args={},
        )
    )
    def benchmark(M, N, K, provider):
        DEVICE = 'cuda:0'
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        if provider == 'triton1':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_op(a, b, kernel_type="triton1"), quantiles=quantiles)
        if provider == 'triton2':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_op(a, b, kernel_type="triton2"), quantiles=quantiles)
        if provider == 'triton3':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_op(a, b, kernel_type="triton3"), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    device_id = torch.cuda.current_device()
    gpu_type = torch.cuda.get_device_name(device_id)
    safe_gpu_type = re.sub(r'[\\/:*?"<>| ]+', "_", gpu_type)
    save_path = os.path.abspath(f"__bench__gpu{device_id}_{safe_gpu_type}")
    print(f"save_path: {save_path}")
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=save_path)


if __name__ == '__main__':
    matmul_test()
    matmul_bench()

