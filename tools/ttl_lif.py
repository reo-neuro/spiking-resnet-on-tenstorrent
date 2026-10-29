# Multi-step LIF (SimpleLIFNode, tau=2, decay_input, v_th=1, hard reset to 0) as one tt-lang kernel, in three
# layouts: make_lif takes [1, 1, T*R, C] (a conv output), make_lif4 takes the T = 4 steps as four [1, 1, R, C]
# tensors (one conv per step), make_lif_nchw takes [T*N, C, H, W] (a BN output).
# Each node walks its share of the tensor and keeps the membrane potential in L1 across the T steps:
#   h_t = v_{t-1} + (x_t - v_{t-1}) / 2,  s_t = (h_t >= 1),  v_t = h_t * (1 - s_t),  v_0 = 0
# tt-lang has no >= or where that work in both 1.1.5 and 1.1.10. make_lif and make_lif4 (S25, S27) carry h and use
#   s = lt(1-, h)  (1- = 0.99999994, the fp32 just below 1, so h > 1- iff h >= 1),  v = h * lt(h, 1),
# with the constants from fill; make_lif_nchw uses s = sign(relu(floor(h))) and v = h - h * s (about three times
# the SFPU work of lt). Every form is bit-exact fp32 against the CPU when run through ttnn (D22 for tt-xla).
# make_lif_add (S30) is make_lif with the residual add of a SEW block fused in: out = s + r.
# make_lif2 (S32) takes the stem split in two: two [1, 1, 2*R, C] inputs with two time steps each.
# Run standalone (ttnn device): python ttl_lif.py [T] [R] [C] [fp32] [split4 [bf16out] | split2 | res]
# (split4: make_lif4, T = 4; split2: make_lif2, T = 4; res: make_lif_add with r random integers 0 .. 3)
# make_lif, make_lif_add and make_lif4 are fp32 only (the 1s are an fp32 fill).
import sys

import torch
import ttl

TILE = 32
# a block of one value: ttl.block.fill in tt-lang 1.1.10 (the container), ttl.math.fill in 1.1.5 (tt-xla 1.4.0)
fill = ttl.block.fill if hasattr(ttl, "block") else ttl.math.fill


def _blocks(RT, CT, nodes, max_tiles):
    """The block (rb, bw), in tiles, for RT x CT tiles per time step over `nodes` nodes.

    Within 10% of the fewest tiles per node, the widest block, then the tallest, up to max_tiles. Bigger blocks
    move more tiles per copy and spread the SFPU set-up over more tiles; a block one tile wide out of two reads only
    half the DRAM banks and is slower (S25). When rb does not divide RT, the last block of a column is moved up to
    end at RT, so a few rows are computed and written twice with the same values.
    """
    blocks = [(bw, rb) for bw in range(1, CT + 1) if CT % bw == 0 for rb in range(1, min(RT, max_tiles // bw) + 1)]
    load = {b: -(-(-(-RT // b[1]) * (CT // b[0])) // nodes) * b[1] * b[0] for b in blocks}
    least = min(load.values())
    bw, rb = max(b for b in blocks if load[b] * 10 <= least * 11)
    return rb, bw


def make_lif(T, max_tiles=32, grid="auto"):
    """LIF on x = [1, 1, T*R, C] (R, C multiples of 32; time step t in rows t*R .. t*R+R-1) or on [1, T, R, C].

    [1, 1, T*R, C] is the layout tt-mlir gives a conv output ([1, 1, N*H*W, C] with T*N images), so in SEW-ResNet18
    with the BN folded into the convs the kernel sits between two convs with no permute (S18) and, taking the 4D
    shape as is, with no reshape either (S21; a 2D [T*R, C] kernel left one reshape on each side). When R is not a
    multiple of 32 (layer4: N*H*W = 392), a time step would start in the middle of a tile, so the caller hands
    [1, T, R, C] instead: tiles are cut from the last two dims, so each step's R rows are padded to whole tiles
    (one reshape on each side, instead of going through a 7x7 NCHW padded to 32x32, S23). A unit of work is one
    rb x bw block of a time step (_blocks), and the units are split evenly over all nodes.

    grid: "auto" is the device's compute grid under ttnn (11 x 10 on the p150a), but tt-xla 1.4.0 resolves the kernel
    against a stub 8 x 8 Wormhole device, so through tt-xla the kernel ran on 64 cores until S26; pass (11, 10) there.

    Per step the compute writes s and the next h, each in one fused SFPU pass (tt-lang gives every store its own
    pass, so h rather than v is carried: v is recomputed from h where it is needed). S25: per step two passes instead
    of three, half the copies into DEST (3 instead of 6) and about half the SFPU instructions of the S17 form, and
    bit-exact like it.
    """

    @ttl.operation(grid=grid)
    def lif(x, out):
        # [1, T, R, C]: step t is x[0, t], rows 0 .. RT-1; [1, 1, T*R, C]: step t is rows t*RT .. t*RT+RT-1
        split = 1 if T > 1 and x.shape[1] == T else 0
        RT = -(-x.shape[2] // TILE) if split else x.shape[2] // TILE // T
        rstep = RT * (1 - split)
        CT = x.shape[3] // TILE
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        nodes = grid_cols * grid_rows
        rb, bw = _blocks(RT, CT, nodes, max_tiles)
        nb = CT // bw
        units = -(-RT // rb) * nb
        per_node = -(-units // nodes)
        rlast = RT - rb

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(rb, bw), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(out, shape=(rb, bw), block_count=2)
        h_dfb = ttl.make_dataflow_buffer_like(x, shape=(rb, bw), block_count=2)

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                if first + i < units:
                    with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                        hn.store(xv * 0.5)
                    for t in range(T - 1):
                        with h_dfb.wait() as hv:
                            with s_dfb.reserve() as so:
                                so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))
                            with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                                v = hv * ttl.math.lt(hv, fill(1.0, shape=(rb, bw), dtype=torch.float32))
                                hn.store(v + (xv - v) * 0.5)
                    with h_dfb.wait() as hv, s_dfb.reserve() as so:
                        so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    for t in range(T):
                        r = t * rstep + row
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[0, t * split, r : r + rb, col : col + bw], blk)
                            tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    for t in range(T):
                        r = t * rstep + row
                        with s_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[0, t * split, r : r + rb, col : col + bw])
                            tx.wait()

    return lif


def make_lif_add(T, max_tiles=8, grid="auto"):
    """make_lif that also adds r to its spikes: out = s + r, with r laid out like out (S30).

    The second LIF of a SEW block (cnf="ADD") feeds `identity + sn2(...)` and nothing else, so taking the residual
    here drops the separate ttnn.add (8 of them, 0.71 ms in S27): the add read s and r and wrote the sum, while the
    kernel now reads r once more instead of writing s. The values are small integers (s is 0 or 1, r at most 3), so
    the sum is exact whichever unit adds it and even when tt-xla rounds every CB read (D22). Blocks and arithmetic as
    in make_lif; r is read in the same blocks as x, step by step. With its four DFBs the kernel is fastest with
    blocks of at most 8 tiles: layer1 (4 x 2 instead of 8 x 2) 220 -> 208 us, the other layers the same (S32).
    """

    @ttl.operation(grid=grid)
    def lif_add(x, r, out):
        split = 1 if T > 1 and x.shape[1] == T else 0
        RT = -(-x.shape[2] // TILE) if split else x.shape[2] // TILE // T
        rstep = RT * (1 - split)
        CT = x.shape[3] // TILE
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        nodes = grid_cols * grid_rows
        rb, bw = _blocks(RT, CT, nodes, max_tiles)
        nb = CT // bw
        units = -(-RT // rb) * nb
        per_node = -(-units // nodes)
        rlast = RT - rb

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(rb, bw), block_count=2)
        r_dfb = ttl.make_dataflow_buffer_like(r, shape=(rb, bw), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(out, shape=(rb, bw), block_count=2)
        h_dfb = ttl.make_dataflow_buffer_like(x, shape=(rb, bw), block_count=2)

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                if first + i < units:
                    with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                        hn.store(xv * 0.5)
                    for t in range(T - 1):
                        with h_dfb.wait() as hv:
                            with r_dfb.wait() as rv, s_dfb.reserve() as so:
                                so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv) + rv)
                            with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                                v = hv * ttl.math.lt(hv, fill(1.0, shape=(rb, bw), dtype=torch.float32))
                                hn.store(v + (xv - v) * 0.5)
                    with h_dfb.wait() as hv, r_dfb.wait() as rv, s_dfb.reserve() as so:
                        so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv) + rv)

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    for t in range(T):
                        rr = t * rstep + row
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[0, t * split, rr : rr + rb, col : col + bw], blk)
                            tx.wait()
                        with r_dfb.reserve() as blk:
                            tx = ttl.copy(r[0, t * split, rr : rr + rb, col : col + bw], blk)
                            tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    for t in range(T):
                        rr = t * rstep + row
                        with s_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[0, t * split, rr : rr + rb, col : col + bw])
                            tx.wait()

    return lif_add


def make_lif4(max_tiles=32, bf16_out=False, grid="auto"):
    """make_lif for T = 4 with each time step its own input, x0 .. x3 = [1, 1, R, C], into out = [1, 1, 4*R, C].

    SEW-ResNet18's stem conv runs as one conv per time step (STEM_SPLIT=4 in xla-sew18.py, S24); reading the four
    outputs directly saves concatenating them first (0.5 ms). tt-lang compiles the kernel from its source and cannot
    pick a tensor by a loop variable, so the four reads are written out. Blocks and arithmetic as in make_lif.
    bf16_out=True writes the spikes (exactly 0 or 1) to a bf16 out (S26): half the bytes written, and the stem's
    maxpool, which runs in bf16, needs no typecast in front. A store needs the block's dtype and the kernel cannot
    pick a dtype by a captured value, so that variant has a compute thread of its own.
    """

    @ttl.operation(grid=grid)
    def lif(x0, x1, x2, x3, out):
        RT = x0.shape[2] // TILE
        CT = x0.shape[3] // TILE
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        nodes = grid_cols * grid_rows
        rb, bw = _blocks(RT, CT, nodes, max_tiles)
        nb = CT // bw
        units = -(-RT // rb) * nb
        per_node = -(-units // nodes)
        rlast = RT - rb

        x_dfb = ttl.make_dataflow_buffer_like(x0, shape=(rb, bw), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(out, shape=(rb, bw), block_count=2)
        h_dfb = ttl.make_dataflow_buffer_like(x0, shape=(rb, bw), block_count=2)

        if bf16_out:
            s32_dfb = ttl.make_dataflow_buffer_like(x0, shape=(rb, bw), block_count=2)

            # the spikes go through an fp32 DFB and a pass of their own that casts them to bf16: tt-lang only lets
            # a pass mix fp32 and bf16 tiles when every fp32 input goes straight into a typecast, and tt-lang 1.1.5
            # folds an fp32 -> fp32 typecast away before that check (1.1.10 lets one through)
            @ttl.compute()
            def compute():
                node_col, node_row = ttl.node(dims=2)
                first = (node_row * grid_cols + node_col) * per_node
                for i in range(per_node):
                    if first + i < units:
                        with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                            hn.store(xv * 0.5)
                        for t in range(3):
                            with h_dfb.wait() as hv:
                                with s32_dfb.reserve() as so:
                                    so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))
                                with s32_dfb.wait() as sv, s_dfb.reserve() as so:
                                    so.store(ttl.math.typecast(sv, torch.bfloat16))
                                with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                                    v = hv * ttl.math.lt(hv, fill(1.0, shape=(rb, bw), dtype=torch.float32))
                                    hn.store(v + (xv - v) * 0.5)
                        with h_dfb.wait() as hv, s32_dfb.reserve() as so:
                            so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))
                        with s32_dfb.wait() as sv, s_dfb.reserve() as so:
                            so.store(ttl.math.typecast(sv, torch.bfloat16))
        else:
            @ttl.compute()
            def compute():
                node_col, node_row = ttl.node(dims=2)
                first = (node_row * grid_cols + node_col) * per_node
                for i in range(per_node):
                    if first + i < units:
                        with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                            hn.store(xv * 0.5)
                        for t in range(3):
                            with h_dfb.wait() as hv:
                                with s_dfb.reserve() as so:
                                    so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))
                                with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                                    v = hv * ttl.math.lt(hv, fill(1.0, shape=(rb, bw), dtype=torch.float32))
                                    hn.store(v + (xv - v) * 0.5)
                        with h_dfb.wait() as hv, s_dfb.reserve() as so:
                            so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x0[0, 0, row : row + rb, col : col + bw], blk)
                        tx.wait()
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x1[0, 0, row : row + rb, col : col + bw], blk)
                        tx.wait()
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x2[0, 0, row : row + rb, col : col + bw], blk)
                        tx.wait()
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x3[0, 0, row : row + rb, col : col + bw], blk)
                        tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    for t in range(4):
                        r = t * RT + row
                        with s_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[0, 0, r : r + rb, col : col + bw])
                            tx.wait()

    return lif


def make_lif2(max_tiles=32, grid="auto"):
    """make_lif4 for the stem conv split in two (S32): x0, x1 = [1, 1, 2*R, C] hold time steps 0, 1 and 2, 3.

    Two convs over 16 images each run in 0.97 ms where four over 8 take 1.07 ms (tt-mlir picks l1_full and
    act_block_h_override 64 for both); reading the two outputs here saves their concat. T = 4 only; blocks and
    arithmetic as in make_lif.
    """

    @ttl.operation(grid=grid)
    def lif(x0, x1, out):
        RT = x0.shape[2] // TILE // 2
        CT = x0.shape[3] // TILE
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        nodes = grid_cols * grid_rows
        rb, bw = _blocks(RT, CT, nodes, max_tiles)
        nb = CT // bw
        units = -(-RT // rb) * nb
        per_node = -(-units // nodes)
        rlast = RT - rb

        x_dfb = ttl.make_dataflow_buffer_like(x0, shape=(rb, bw), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(out, shape=(rb, bw), block_count=2)
        h_dfb = ttl.make_dataflow_buffer_like(x0, shape=(rb, bw), block_count=2)

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                if first + i < units:
                    with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                        hn.store(xv * 0.5)
                    for t in range(3):
                        with h_dfb.wait() as hv:
                            with s_dfb.reserve() as so:
                                so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))
                            with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                                v = hv * ttl.math.lt(hv, fill(1.0, shape=(rb, bw), dtype=torch.float32))
                                hn.store(v + (xv - v) * 0.5)
                    with h_dfb.wait() as hv, s_dfb.reserve() as so:
                        so.store(ttl.math.lt(fill(0.9999999403953552, shape=(rb, bw), dtype=torch.float32), hv))

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    for t in range(2):
                        r = t * RT + row
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x0[0, 0, r : r + rb, col : col + bw], blk)
                            tx.wait()
                    for t in range(2):
                        r = t * RT + row
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x1[0, 0, r : r + rb, col : col + bw], blk)
                            tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                u = first + i
                if u < units:
                    rbi = u // nb
                    row = rbi * rb
                    if row > rlast:
                        row = rlast
                    col = (u - rbi * nb) * bw
                    for t in range(4):
                        r = t * RT + row
                        with s_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[0, 0, r : r + rb, col : col + bw])
                            tx.wait()

    return lif


def make_lif_nchw(T):
    """The same LIF on x = [T*N, C, H, W] as laid out by the BN before it (H, W padded up to whole tiles).

    Time step t of image n is x[t*N + n], so every (n, c) plane is its own tiles and each time step starts on a
    tile; no reshape is needed around the kernel. The N*C planes of one time step are split evenly over all nodes,
    and a block is one plane (HT x WT tiles). Padding elements are computed too and never read back.

    Unlike make_lif, all arithmetic runs on the SFPU and follows SimpleLIFNode's order, so h is bit-exact fp32:
    h_t = v + (x * 0.5 - v * 0.5) (= v + (x - v) / 2), h_0 = x * 0.5, v = h - h * s. Adding two CBs directly
    (as in (v + x) * 0.5) goes through the FPU, which rounds fp32 operands to about 10 mantissa bits (S16).
    Exactness also needs UnpackToDestFp32 on the CBs the SFPU reads; tt-lang sets it when run through ttnn, but
    tt-xla 1.4.0 does not pass it on, so under tt-xla every fp32 CB read is rounded the same way (D22).
    """

    @ttl.operation(grid="auto")
    def lif(x, out):
        B, C, H, W = x.shape
        N = B // T
        HT = -(-H // TILE)
        WT = -(-W // TILE)
        planes = N * C
        grid_cols, grid_rows = ttl.grid_size(dims=2)
        per_node = -(-planes // (grid_cols * grid_rows))

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(HT, WT), block_count=2)
        s_dfb = ttl.make_dataflow_buffer_like(out, shape=(HT, WT), block_count=2)
        h_dfb = ttl.make_dataflow_buffer_like(x, shape=(HT, WT), block_count=2)
        v_dfb = ttl.make_dataflow_buffer_like(x, shape=(HT, WT), block_count=2)

        @ttl.compute()
        def compute():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                if first + i < planes:
                    with x_dfb.wait() as xv, h_dfb.reserve() as hn:
                        hn.store(xv * 0.5)
                    for t in range(T - 1):
                        # spike of step t, and the reset that carries v into step t + 1
                        with h_dfb.wait() as hv:
                            with s_dfb.reserve() as so:
                                so.store(ttl.math.sign(ttl.math.relu(ttl.math.floor(hv))))
                            with v_dfb.reserve() as vn:
                                vn.store(hv - hv * ttl.math.sign(ttl.math.relu(ttl.math.floor(hv))))
                        with x_dfb.wait() as xv, v_dfb.wait() as vp, h_dfb.reserve() as hn:
                            hn.store(vp + (xv * 0.5 - vp * 0.5))
                    with h_dfb.wait() as hv, s_dfb.reserve() as so:
                        so.store(ttl.math.sign(ttl.math.relu(ttl.math.floor(hv))))

        @ttl.datamovement()
        def read():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                p = first + i
                if p < planes:
                    n = p // C
                    c = p - n * C
                    for t in range(T):
                        with x_dfb.reserve() as blk:
                            tx = ttl.copy(x[t * N + n, c, 0:HT, 0:WT], blk)
                            tx.wait()

        @ttl.datamovement()
        def write():
            node_col, node_row = ttl.node(dims=2)
            first = (node_row * grid_cols + node_col) * per_node
            for i in range(per_node):
                p = first + i
                if p < planes:
                    n = p // C
                    c = p - n * C
                    for t in range(T):
                        with s_dfb.wait() as blk:
                            tx = ttl.copy(blk, out[t * N + n, c, 0:HT, 0:WT])
                            tx.wait()

    return lif


def lif_reference(x, T):
    """SimpleLIFNode (tau=2, decay_input, v_th=1, v_reset=0) on [T*R, C] in fp32; also returns h."""
    xs = x.float().reshape(T, -1, x.shape[1])
    v = torch.zeros_like(xs[0])
    spikes, hs = [], []
    for t in range(T):
        h = v + (xs[t] - v) / 2.0
        s = (h >= 1.0).float()
        v = h * (1.0 - s)
        spikes.append(s)
        hs.append(h)
    return torch.cat(spikes), torch.cat(hs)


if __name__ == "__main__":
    import ttnn

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    C = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    dt = sys.argv[4] if len(sys.argv) > 4 else "fp32"
    assert dt == "fp32", "make_lif and make_lif4 are fp32 only"
    split4 = len(sys.argv) > 5 and sys.argv[5] == "split4"
    bf16_out = split4 and len(sys.argv) > 6 and sys.argv[6] == "bf16out"
    res = len(sys.argv) > 5 and sys.argv[5] == "res"
    split2 = len(sys.argv) > 5 and sys.argv[5] == "split2"
    tdt, ndt = torch.float32, ttnn.float32

    torch.manual_seed(0)
    x = (torch.randn(T * R, C) * 1.5 + 0.5).to(tdt)
    # h exactly at, just below and just above 1 at t = 0 and t = 1 (x0 = 1 leaves v0 = 0.5), and a negative x
    below, above = torch.nextafter(torch.tensor(2.0), torch.tensor(0.0)), torch.nextafter(torch.tensor(2.0), torch.tensor(3.0))
    edge = [[2.0, 1.5], [below, 1.5], [above, 1.5], [1.0, 1.5], [1.0, torch.nextafter(torch.tensor(1.5), torch.tensor(0.0))],
            [-3.0, 2.0]] if T > 1 else []
    for c, (e0, e1) in enumerate(edge):
        x[0, c], x[R, c] = float(e0), float(e1)
    r = torch.randint(0, 4, (T * R, C)).float() if res else torch.zeros(T * R, C)
    device = ttnn.open_device(device_id=0)
    try:
        conv = dict(dtype=ndt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_tt = ttnn.from_torch(torch.zeros(1, 1, T * R, C), **dict(conv, dtype=ttnn.bfloat16 if bf16_out else ndt))
        if split4:
            assert T == 4
            make_lif4(bf16_out=bf16_out)(*[ttnn.from_torch(xs.reshape(1, 1, R, C), **conv) for xs in x.chunk(4)], out_tt)
        elif split2:
            assert T == 4
            make_lif2()(*[ttnn.from_torch(xs.reshape(1, 1, 2 * R, C), **conv) for xs in x.chunk(2)], out_tt)
        elif res:
            make_lif_add(T)(ttnn.from_torch(x.reshape(1, 1, T * R, C), **conv),
                            ttnn.from_torch(r.reshape(1, 1, T * R, C), **conv), out_tt)
        else:
            make_lif(T)(ttnn.from_torch(x.reshape(1, 1, T * R, C), **conv), out_tt)
        s = ttnn.to_torch(out_tt).float().reshape(T * R, C)
    finally:
        ttnn.close_device(device)

    ref, h = lif_reference(x, T)
    ref = ref + r
    bad = s != ref
    if edge:
        print(f"threshold edge cases: kernel {s[[0, R], :len(edge)].tolist()}, reference {ref[[0, R], :len(edge)].tolist()}, "
              f"h {h[[0, R], :len(edge)].tolist()}")
    near = (h - 1.0).abs() < 1e-2
    print(f"T={T} R={R} C={C} {dt}{' split4' if split4 else ''}{' split2' if split2 else ''}{' bf16out' if bf16_out else ''}{' res' if res else ''}: spikes {int(ref.sum())}/{ref.numel()}, mismatches {int(bad.sum())} "
          f"(of which |h-1|<1e-2: {int((bad & near).sum())}), values seen {sorted(set(s.unique().tolist()))[:6]}")
    edge_bad = int(bad[[0, R], :len(edge)].sum()) if edge else 0  # exact at the threshold: no slack here
    print("PASS" if int((bad & ~near).sum()) == 0 and edge_bad == 0 else "FAIL")
