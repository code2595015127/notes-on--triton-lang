import torch
import math
from kernel import vector_add_kernel, vector_add_kernel_software_pipeline

def vector_add_op(a: torch.Tensor, b: torch.Tensor, use_persistent: bool = False) -> torch.Tensor:
    assert a.numel() == b.numel()
    assert a.device == b.device
    c = torch.empty_like(a)
    n_elements = a.numel()

    TILE_SIZE = 1024
    tile_num = math.ceil(n_elements / float(TILE_SIZE))

    if use_persistent: 
        props = torch.cuda.get_device_properties("cuda:0")
        # print("props: ", props)
        sm_cnt = props.multi_processor_count
        # print("sm_cnt: ", sm_cnt)
        block_num = sm_cnt
    else:
        block_num = tile_num
    # print("block_num: ", block_num)
    grid = lambda meta: (block_num,)

    if use_persistent:
        vector_add_kernel_software_pipeline[grid](a, b, c, n_elements, block_num, tile_num, TILE_SIZE)
    else:
        vector_add_kernel[grid](a, b, c, n_elements, block_num, tile_num, TILE_SIZE)

    return c

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device="cuda:0")
    b = torch.randn(1024, 1024, device="cuda:0")

    c_triton = vector_add_op(a, b, use_persistent=True)
    print(c_triton)

    c_torch = a + b
    print(c_torch)

    print(
        f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(c_triton-c_torch))}'
    )

