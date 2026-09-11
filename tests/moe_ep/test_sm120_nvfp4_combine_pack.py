"""Bit-level coverage of SM120's quantized combine epilogue transpose."""

import os

import pytest
import torch


@pytest.mark.arch_sm120
def test_sm120_nvfp4_combine_pack() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("needs SM120")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cuda.bindings import driver as cuda

    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    from moe_sm120_nvfp4_split.sm120_mma import pack_combine_e2m1

    @cute.kernel
    def pack_kernel(values: cute.Tensor, output: cute.Tensor):
        sample = cute.arch.block_idx()[0]
        lane = cute.arch.lane_idx()
        words = pack_combine_e2m1(
            values[sample, 0, lane],
            values[sample, 1, lane],
            values[sample, 2, lane],
            values[sample, 3, lane],
        )
        if lane < 4:
            for i in cutlass.range_constexpr(4):
                output[sample, lane, i] = words[i]

    @cute.jit
    def launch(values: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        pack_kernel(values, output).launch(
            grid=[values.shape[0], 1, 1], block=[32, 1, 1], stream=stream
        )

    # Exact codes, midpoint ties (round to even), saturation, and signed zero
    # occupy every lane/accumulator position across the independent warps.
    positive = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 100],
        device="cuda",
    )
    cases = torch.cat((positive, -positive))
    indices = torch.arange(32 * 4 * 32, device="cuda").reshape(32, 4, 32)
    indices = (indices + indices // 32 + indices // 128) % cases.numel()
    values = cases[indices].contiguous()
    output = torch.empty((32, 4, 4), device="cuda", dtype=torch.int32)
    values_cute = cutlass_torch.from_dlpack(values)
    output_cute = cutlass_torch.from_dlpack(output)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, values_cute, output_cute, stream)
    compiled(values_cute, output_cute, stream)

    # Put even codes first so argmin resolves exact midpoint ties correctly.
    code_order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device="cuda")
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    nearest = (values.abs()[..., None] - magnitudes[code_order]).abs().argmin(-1)
    codes = (code_order[nearest] | (torch.signbit(values).long() << 3)).to(torch.uint8)
    codes = codes.reshape(32, 4, 8, 4)
    row0 = torch.cat((codes[:, 0], codes[:, 2]), dim=1).transpose(1, 2)
    row1 = torch.cat((codes[:, 1], codes[:, 3]), dim=1).transpose(1, 2)
    rows = torch.cat((row0, row1), dim=-1)
    expected = rows[..., 0::2] | (rows[..., 1::2] << 4)
    torch.testing.assert_close(output.view(torch.uint8), expected, atol=0, rtol=0)


@pytest.mark.arch_sm120
@pytest.mark.parametrize(
    "hidden,store_tiles,tile_tokens", [(96, 1, 128), (256, 4, 128), (768, 4, 256)]
)
def test_sm120_nvfp4_combine_store_partial_tiles(
    hidden, store_tiles, tile_tokens
) -> None:
    """Check partial rows, half hidden tiles, and data/scale scratch reuse exactly."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("needs SM120")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    from types import SimpleNamespace

    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cuda.bindings import driver as cuda

    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    # Match the frontend's import order: its shared input stager loads the
    # common MegaMoE helpers before selecting the SM120 kernel source tree.
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import quant_stage  # noqa: F401

    bootstrap_paths()
    from moe_sm120_nvfp4_split.kernel_fc2_combine import Sm120Fc2CombineKernel

    # Exercise the production store helper without GEMM or NVSHMEM setup.
    store = object.__new__(Sm120Fc2CombineKernel)
    store.hidden = hidden
    store.mma_tiler = (64, tile_tokens, 128)
    store.compute_warp_id = (0, 1, 2, 3)
    store.epilog_sync_bar_id = 1
    store.token_back_by_dispatch = True
    store.combine_store_tiles = store_tiles
    tokens = tile_tokens + 1

    @cute.kernel
    def store_kernel(
        data: cute.Tensor, scales: cute.Tensor, out: cute.Tensor, sf: cute.Tensor
    ):
        tid = cute.arch.thread_idx()[0]
        allocator = utils.SmemAllocator()
        s_data = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((tile_tokens, 8 * store_tiles))
        )
        s_sf = allocator.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((tile_tokens, 4 * store_tiles))
        )
        for token_tile in cutlass.range_constexpr(2):
            valid = tile_tokens if token_tile == 0 else 1
            for tile in cutlass.range_constexpr((hidden + 63) // 64):
                for local_row in cutlass.range(tid, tile_tokens, 128):
                    if local_row < valid:
                        for i in cutlass.range_constexpr(8):
                            word = tile * 8 + i
                            if word < hidden // 8:
                                s_data[local_row, (tile % store_tiles) * 8 + i] = data[
                                    token_tile * tile_tokens + local_row, word
                                ]
                        for i in cutlass.range_constexpr(4):
                            block = tile * 4 + i
                            if block < hidden // 16:
                                s_sf[local_row, (tile % store_tiles) * 4 + i] = scales[
                                    token_tile * tile_tokens + local_row, block
                                ]
                args = SimpleNamespace(combine_output=out, combine_sf=sf)
                work = SimpleNamespace(
                    valid_tokens_in_tile=cutlass.Int32(valid),
                    cumulative_data_physical_row=cutlass.Int32(0),
                    tile_n_idx=cutlass.Int32(token_tile),
                    tile_m_idx=tile,
                )
                if tile % store_tiles == store_tiles - 1:
                    store._store_quantized_combine_tile(
                        args, work, s_data, s_sf, tid // 32, tid % 32
                    )

    @cute.jit
    def launch(
        data: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
        sf: cute.Tensor,
        stream: cuda.CUstream,
    ):
        store_kernel(data, scales, out, sf).launch(
            grid=[1, 1, 1], block=[128, 1, 1], stream=stream
        )

    data = torch.randint(
        -(1 << 31), 1 << 31, (tokens, hidden // 8), device="cuda", dtype=torch.int32
    )
    scales = torch.randn((tokens, hidden // 16), device="cuda", dtype=torch.bfloat16)
    out = torch.full(
        (tokens + 1, 1, hidden // 2), 0xA5, device="cuda", dtype=torch.uint8
    )
    sf = torch.full(
        (tokens + 1, 1, hidden // 16), -123.0, device="cuda", dtype=torch.bfloat16
    )
    tensors = [cutlass_torch.from_dlpack(t) for t in (data, scales, out, sf)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch, *tensors, stream)
    compiled(*tensors, stream)
    torch.testing.assert_close(out[:tokens, 0], data.view(torch.uint8), atol=0, rtol=0)
    torch.testing.assert_close(sf[:tokens, 0], scales, atol=0, rtol=0)
    assert (out[tokens] == 0xA5).all()
    assert (sf[tokens] == -123.0).all()


@pytest.mark.arch_sm120
def test_sm120_nvfp4_fused_combine_quantization() -> None:
    """Packed PTX must preserve scalar rounding, scales and FP4 bits at edges."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("needs SM120")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        bootstrap_paths,
    )

    bootstrap_paths()
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    from cuda.bindings import driver as cuda
    from moe_sm120_nvfp4_split.sm120_mma import (
        fused_combine_quant_pack,
        pack_combine_e2m1,
    )

    @cute.kernel
    def check(
        values: cute.Tensor,
        output: cute.Tensor,
        scales: cute.Tensor,
        fused: cutlass.Constexpr[bool],
    ):
        s = cute.arch.block_idx()[0]
        lane = cute.arch.lane_idx()
        v0 = values[s, 0, lane]
        v1 = values[s, 1, lane]
        v2 = values[s, 2, lane]
        v3 = values[s, 3, lane]
        if cutlass.const_expr(fused):
            w0, w1, w2, w3, s0, s1 = fused_combine_quant_pack(v0, v1, v2, v3)
        else:
            v0 = cutlass.Float32(v0.to(cutlass.BFloat16))
            v1 = cutlass.Float32(v1.to(cutlass.BFloat16))
            v2 = cutlass.Float32(v2.to(cutlass.BFloat16))
            v3 = cutlass.Float32(v3.to(cutlass.BFloat16))
            s0 = cute.arch.fmax(cute.math.absf(v0), cute.math.absf(v2))
            s1 = cute.arch.fmax(cute.math.absf(v1), cute.math.absf(v3))
            for mask in (4, 8, 16):
                s0 = cute.arch.fmax(s0, cute.arch.shuffle_sync(s0, lane ^ mask))
                s1 = cute.arch.fmax(s1, cute.arch.shuffle_sync(s1, lane ^ mask))
            d0 = cutlass.Float32(s0.to(cutlass.BFloat16)) * cutlass.Float32(1 / 6)
            d1 = cutlass.Float32(s1.to(cutlass.BFloat16)) * cutlass.Float32(1 / 6)
            r0 = cute.arch.fmin(
                cute.arch.rcp_approx(d0), cutlass.Float32(3.4028234663852886e38)
            )
            r1 = cute.arch.fmin(
                cute.arch.rcp_approx(d1), cutlass.Float32(3.4028234663852886e38)
            )
            r0 = r0 * cute.arch.fmin(d0 * cutlass.Float32(1e30), cutlass.Float32(1))
            r1 = r1 * cute.arch.fmin(d1 * cutlass.Float32(1e30), cutlass.Float32(1))
            w0, w1, w2, w3 = pack_combine_e2m1(v0 * r0, v1 * r1, v2 * r0, v3 * r1)
        if lane < 4:
            output[s, lane, 0] = w0
            output[s, lane, 1] = w1
            output[s, lane, 2] = w2
            output[s, lane, 3] = w3
            scales[s, lane, 0] = s0.to(cutlass.BFloat16)
            scales[s, lane, 1] = s1.to(cutlass.BFloat16)

    @cute.jit
    def launch(
        v: cute.Tensor,
        o: cute.Tensor,
        s: cute.Tensor,
        mode: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        check(v, o, s, mode).launch(
            grid=[v.shape[0], 1, 1], block=[32, 1, 1], stream=stream
        )

    generator = torch.Generator(device="cuda").manual_seed(531)
    values = torch.randn((8192, 4, 32), device="cuda", generator=generator)
    values *= torch.logspace(-39, 30, 8192, device="cuda")[:, None, None]
    values[:32] = 0
    values[16:32] = -0.0
    edge = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            0.5,
            0.75,
            1,
            1.25,
            1.5,
            1.75,
            2,
            2.5,
            3,
            3.5,
            4,
            5,
            6,
            -0.25,
            -0.5,
            -0.75,
            -1,
            -1.25,
            -1.5,
            -1.75,
            -2,
            -2.5,
            -3,
            -3.5,
            -4,
            -5,
            -6,
            1.00390625,
            -1.00390625,
        ],
        device="cuda",
    )
    values[32:64] = edge[None, None, :]
    values[64:96] = edge.roll(7)[None, None, :]
    outputs = []
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    for mode in (False, True):
        out = torch.empty((8192, 4, 4), device="cuda", dtype=torch.int32)
        sf = torch.empty((8192, 4, 2), device="cuda", dtype=torch.bfloat16)
        tensors = [ct.from_dlpack(t) for t in (values, out, sf)]
        compiled = cute.compile(launch, *tensors, mode, stream)
        compiled(*tensors, stream)
        outputs.append((out, sf))
    assert torch.equal(outputs[0][0], outputs[1][0]), int(
        (outputs[0][0] != outputs[1][0]).sum()
    )
    assert torch.equal(outputs[0][1].view(torch.int16), outputs[1][1].view(torch.int16))
