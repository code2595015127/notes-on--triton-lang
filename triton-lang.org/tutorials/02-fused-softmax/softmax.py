import triton
import triton.language as tl

@triton.jit
def softmax_kernel_1pass(i_ptr, o_ptr, COL_NUMS: tl.constexpr, COL_BLOCKS: tl.constexpr, ROW_NUMS: tl.constexpr, ROW_BLOCKS: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid*ROW_BLOCKS
    row_end = (pid+1)*ROW_BLOCKS
    col_off = tl.arange(0, COL_BLOCKS)
    col_mas = col_off < COL_NUMS
    for row in tl.range(row_start, row_end):
        if row < ROW_NUMS:
            i = tl.load(i_ptr + row*COL_NUMS + col_off, mask=col_mas, other=-float('inf'))
            i_max = tl.max(i)
            i_sta = tl.sub(i, i_max)
            i_exp = tl.exp(i_sta)
            i_sum = tl.sum(i_exp, axis=0)
            o = i_exp / i_sum
            tl.store(o_ptr + row*COL_NUMS + col_off, o, mask=col_mas)

@triton.jit
def softmax_kernel_2pass(i_ptr, o_ptr, COL_NUMS: tl.constexpr, COL_BLOCKS: tl.constexpr, ROW_NUMS: tl.constexpr, ROW_BLOCKS: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid*ROW_BLOCKS
    row_end = (pid+1)*ROW_BLOCKS
    col_off = tl.arange(0, COL_BLOCKS)
    for row in tl.range(row_start, row_end):
        if row < ROW_NUMS:
            l = 0.0
            m = -float('inf')
            for col in tl.range(0, COL_NUMS, COL_BLOCKS):
                col_mas = (col + col_off) < COL_NUMS
                i = tl.load(i_ptr + row*COL_NUMS + col + col_off, mask=col_mas, other=-float('inf'))
                m_old = m
                i_max = tl.max(i, axis=0)
                m     = tl.maximum(m, i_max)
                i_exp = tl.exp(i - m)
                i_sum = tl.sum(i_exp, axis=0)
                l     = l * tl.exp(m_old - m) + i_sum
                
            for col in tl.range(0, COL_NUMS, COL_BLOCKS):
                col_mas = (col + col_off) < COL_NUMS
                i = tl.load(i_ptr + row*COL_NUMS + col + col_off, mask=col_mas, other=-float('inf'))
                o = tl.exp(i - m) / l
                tl.store(o_ptr + row*COL_NUMS + col + col_off, o, mask=col_mas)

import torch
import math
def softmax_op(i, kernel_type='1pass'):
    o = torch.empty_like(i)
    ROW_NUMS, COL_NUMS = i.shape
    ROW_BLOCKS = 2
    GRID_SHAPE = (ROW_NUMS + ROW_BLOCKS - 1) // ROW_BLOCKS
    if kernel_type=='1pass':
        COL_BLOCKS = triton.next_power_of_2(COL_NUMS)
        softmax_kernel_1pass[(GRID_SHAPE,)](i, o, COL_NUMS, COL_BLOCKS, ROW_NUMS, ROW_BLOCKS)
    elif kernel_type=='2pass':
        COL_BLOCKS = int(math.pow(2,13))
        softmax_kernel_2pass[(GRID_SHAPE,)](i, o, COL_NUMS, COL_BLOCKS, ROW_NUMS, ROW_BLOCKS)
    else:
        raise ValueError(f"kernel_type {kernel_type} not supported. Please use kernel_type={kernel_type}")
    return o

def softmax_naive(x):
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

def softmax_test():
    torch.manual_seed(0)
    i = torch.randn(12345, 2345, device='cuda:0')
    o_eager = softmax_naive(i)
    print(o_eager)
    o_triton1 = softmax_op(i, kernel_type='1pass')
    print(o_triton1)
    print(torch.allclose(o_triton1, o_eager))
    o_triton2 = softmax_op(i, kernel_type='2pass')
    print(o_triton2)
    print(torch.allclose(o_triton2, o_eager))

def softmax_bench():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],  # argument names to use as an x-axis for the plot
            x_vals=[
                128 * i for i in 
                list(range(2, 100))
                +list(range(100,1000,100))
                # +list(range(1000,10000,1000))
            ],  # different possible values for `x_name`
            line_arg='provider',  # argument name whose value corresponds to a different line in the plot
            line_vals=['triton1', 'triton2', 'torch', 'naive'],  # possible values for `line_arg``
            line_names=["Triton1", "Triton2", "Torch", "Naive"],  # label name for the lines
            styles=[('orange', '-'), ('green', '-'), ('blue', '-'), ('red', '-')],  # line styles
            ylabel="GB/s",  # label name for the y-axis
            plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
            args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
        ))
    def benchmark(M, N, provider):
        DEVICE = torch.device('cuda:0')
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        stream = getattr(torch, DEVICE.type).Stream()
        getattr(torch, DEVICE.type).set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == 'triton1':
            ms = triton.testing.do_bench(lambda: softmax_op(x, kernel_type='1pass'))
        if provider == 'triton2':
            ms = triton.testing.do_bench(lambda: softmax_op(x, kernel_type='2pass'))
        if provider == 'naive':
            ms = triton.testing.do_bench(lambda: softmax_naive(x))
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    import os
    import re
    import shutil
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
    softmax_test()
    softmax_bench()
