import os
import triton
import torch
from op import vector_add_op

DEVICE = "cuda:0"

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals =['triton_persis_true', 'triton_persis_false', 'torch'],  # Possible values for `line_arg`.
        line_names=['triton_persis_true', 'triton_persis_false', 'torch'],  # Label name for the lines.
        styles=[('red', '-'), ('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton_persis_true':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add_op(x, y, use_persistent=True),  quantiles=quantiles)
    if provider == 'triton_persis_false':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add_op(x, y, use_persistent=False), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

os.makedirs("./__perf_B200__", exist_ok=True)
benchmark.run(print_data=True, show_plots=True, save_path="./__perf_B200__")
