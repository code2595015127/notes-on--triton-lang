import triton
import triton.language as tl
@triton.jit
def vec_add_kernel(a_ptr, b_ptr, c_ptr, num_elements:tl.constexpr, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    off = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mas = off < num_elements
    a = tl.load(a_ptr+off, mask=mas)
    b = tl.load(b_ptr+off, mask=mas)
    c = a + b
    tl.store(c_ptr+off, c, mask=mas)


import torch
def vec_add_op(a, b):
    c = torch.empty_like(a)
    num_elements = a.numel()
    BLOCK_SIZE = 1024
    BLOCK_NUM = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    vec_add_kernel[(BLOCK_NUM,)](a, b, c, num_elements, BLOCK_SIZE)
    return c


def vec_add_test(device_triton):
    torch.manual_seed(0)
    size = (987654,)
    a = torch.randn(size, device=device_triton)
    b = torch.randn(size, device=device_triton)
    c_eager  = a + b
    c_triton = vec_add_op(a, b)

    is_close = torch.allclose(c_eager, c_triton)
    print(f"is_close: {is_close}")
    if not is_close:
        for i in range(c_eager.numel()):
            print(f"c_eager[{i}]: {c_eager[i]}")
            print(f"c_triton[{i}]: {c_triton[i]}")
            if c_eager[i] != c_triton[i]:
                break
    print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(c_eager - c_triton))}')


def vec_add_bench(device_triton):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],  # Argument names to use as an x-axis for the plot.
            x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
            x_log=True,  # x axis is logarithmic.
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
            line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
            line_names=['Triton', 'Torch'],  # Label name for the lines.
            styles=[('blue', '-'), ('green', '-')],  # Line styles.
            ylabel='GB/s',  # Label name for the y-axis.
            plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
            args={},  # Values for function arguments not in `x_names` and `y_name`.
        ))
    def vec_add_bench(size, provider):
        x = torch.rand(size, device=device_triton, dtype=torch.float32)
        y = torch.rand(size, device=device_triton, dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: vec_add_op(x, y), quantiles=quantiles)
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    import os
    import re
    device_id = torch.cuda.current_device()
    gpu_type = torch.cuda.get_device_name(device_id)
    safe_gpu_type = re.sub(r'[\\/:*?"<>| ]+', "_", gpu_type)
    save_path = os.path.abspath(f"__bench__gpu{device_id}_{safe_gpu_type}")
    print(f"save_path: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    vec_add_bench.run(print_data=True, show_plots=True, save_path=save_path)

if __name__ == "__main__":
    """
    torch.rand(...) 未指定 device 时
    default_device = cpu → 创建在 cpu
    default_device = cuda → 创建在 cuda.current_device()
    default_device = cuda:1 → 创建在 cuda:1
    """
    torch.set_default_device("cuda") # "cuda:0"
    torch.cuda.set_device(0)
    print(f"default: {torch.get_default_device()}")
    print(f"current: {torch.cuda.current_device()}")
    device_triton = triton.runtime.driver.active.get_active_torch_device()
    print(f"device_triton: {device_triton}")

    vec_add_test(device_triton)

    vec_add_bench(device_triton)
