import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    n_elements,
    block_num,
    tile_num,
    TILE_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    for tile_start in range(block_id*TILE_SIZE, n_elements, block_num*TILE_SIZE):
        offset = tile_start + tl.arange(0, TILE_SIZE)
        mask = offset < n_elements
        a_tile = tl.load(a_ptr+offset, mask=mask)
        b_tile = tl.load(b_ptr+offset, mask=mask)
        c_tile = tl.add(a_tile, b_tile)
        tl.store(c_ptr+offset, c_tile, mask=mask)


@triton.jit
def vector_add_kernel_software_pipeline(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    n_elements,
    block_num,
    tile_num,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # 起始 offset
    offs = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)

    # -------------------------
    # Prefetch tile 0
    # -------------------------
    mask = offs < n_elements
    a_buf = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b_buf = tl.load(b_ptr + offs, mask=mask, other=0.0)

    # 循环处理后续 tiles
    for _ in range(1, tl.cdiv(n_elements - pid * TILE_SIZE, block_num*TILE_SIZE)):
        # 计算当前 tile
        c_buf = a_buf + b_buf

        # 预取下一 tile
        offs_next = offs + block_num*TILE_SIZE
        mask_next = offs_next < n_elements
        a_next = tl.load(a_ptr + offs_next, mask=mask_next, other=0.0)
        b_next = tl.load(b_ptr + offs_next, mask=mask_next, other=0.0)

        # 写回当前 tile
        tl.store(c_ptr + offs, c_buf, mask=mask)

        # 切换 buffer
        offs = offs_next
        mask = mask_next
        a_buf = a_next
        b_buf = b_next

    # -------------------------
    # 处理最后一个 tile
    # -------------------------
    c_buf = a_buf + b_buf
    tl.store(c_ptr + offs, c_buf, mask=mask)

