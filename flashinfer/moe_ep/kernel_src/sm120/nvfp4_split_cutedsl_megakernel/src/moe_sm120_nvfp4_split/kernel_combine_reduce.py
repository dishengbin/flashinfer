# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""BF16 and NVFP4 top-k reduction for the SM120 Split-MegaMoE path."""

from __future__ import annotations

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Float32, Int32, T

from common.megamoe_constants import Nvfp4E2M1RcpLimit

BF16_VECTOR_THREADS = 512
BF16_HIDDEN_PER_THREAD = 8
NVFP4_VECTOR_THREADS = 128
NVFP4_HIDDEN_PER_THREAD = 16


def _normalize_combine_format(combine_format: str) -> str:
    token = combine_format.strip().lower()
    if token == "nvfp4":
        token = "16e2m1xbf16"
    if token not in ("bf16", "16e2m1xbf16"):
        raise ValueError(
            f"combine_format must be 'bf16' or '16e2m1xbf16', got {combine_format!r}."
        )
    return token


def _topk_reduce_launch_geometry(
    tokens: int,
    hidden: int,
    threads: int,
    hidden_per_thread: int = BF16_HIDDEN_PER_THREAD,
) -> tuple[int, list[int]]:
    """Return a 1-D launch that is valid beyond CUDA's grid-y limit."""
    if tokens <= 0 or hidden <= 0 or threads <= 0 or hidden_per_thread <= 0:
        raise ValueError(
            "tokens, hidden, threads, and hidden_per_thread must be positive, got "
            f"tokens={tokens}, hidden={hidden}, threads={threads}, "
            f"hidden_per_thread={hidden_per_thread}."
        )
    hidden_blocks = (hidden + threads * hidden_per_thread - 1) // (
        threads * hidden_per_thread
    )
    total_blocks = hidden_blocks * tokens
    if total_blocks > (1 << 31) - 1:
        raise ValueError(
            "top-k reduction grid exceeds CUDA gridDim.x: "
            f"hidden_blocks * tokens = {hidden_blocks} * {tokens} = {total_blocks}."
        )
    return hidden_blocks, [total_blocks, 1, 1]


@cute.jit
def _cvt_e2m1_to_fp32(e2m1_reg: cute.Tensor, fp32_reg: cute.Tensor) -> None:
    """Decode packed E2M1 through the Blackwell e2m1-to-f16 instruction."""

    src_words = cute.recast_tensor(e2m1_reg, Int32)
    for word_idx in cutlass.range_constexpr(cute.size(src_words)):
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32()] * 8),
            [src_words[word_idx].ir_value()],
            "{\n"
            "  .reg .b8  b0, b1, b2, b3;\n"
            "  .reg .b32 p0, p1, p2, p3;\n"
            "  .reg .b16 c0, d0, c1, d1, c2, d2, c3, d3;\n"
            "  mov.b32 {b0, b1, b2, b3}, $8;\n"
            "  cvt.rn.f16x2.e2m1x2 p0, b0;\n"
            "  cvt.rn.f16x2.e2m1x2 p1, b1;\n"
            "  cvt.rn.f16x2.e2m1x2 p2, b2;\n"
            "  cvt.rn.f16x2.e2m1x2 p3, b3;\n"
            "  mov.b32 {c0, d0}, p0;\n"
            "  mov.b32 {c1, d1}, p1;\n"
            "  mov.b32 {c2, d2}, p2;\n"
            "  mov.b32 {c3, d3}, p3;\n"
            "  cvt.f32.f16 $0, c0;\n"
            "  cvt.f32.f16 $1, d0;\n"
            "  cvt.f32.f16 $2, c1;\n"
            "  cvt.f32.f16 $3, d1;\n"
            "  cvt.f32.f16 $4, c2;\n"
            "  cvt.f32.f16 $5, d2;\n"
            "  cvt.f32.f16 $6, c3;\n"
            "  cvt.f32.f16 $7, d3;\n"
            "}",
            "=f,=f,=f,=f,=f,=f,=f,=f,r",
            has_side_effects=False,
        )
        for elem_idx in cutlass.range_constexpr(8):
            fp32_reg[word_idx * 8 + elem_idx] = Float32(
                llvm.extractvalue(T.f32(), result, [elem_idx])
            )


@cute.kernel
def topk_reduce_bf16_vec_kernel(
    combine_output: cute.Tensor,
    topk_score: Optional[cute.Tensor],
    reduced_output: cute.Tensor,
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    hidden_blocks: cutlass.Constexpr[int],
    store_dtype: cutlass.Constexpr[str],
):
    """BF16 reduce with one thread handling one 8-hidden vector."""

    linear_block_idx, _, _ = cute.arch.block_idx()
    hidden_vec_block_idx = linear_block_idx % Int32(hidden_blocks)
    token_idx = linear_block_idx // Int32(hidden_blocks)
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]
    vec_idx = hidden_vec_block_idx * block_dim + tid
    base_h = vec_idx * Int32(BF16_HIDDEN_PER_THREAD)

    if base_h < Int32(hidden):
        acc = cute.make_rmem_tensor((BF16_HIDDEN_PER_THREAD,), cutlass.Float32)
        for i in cutlass.range_constexpr(0, BF16_HIDDEN_PER_THREAD, 1):
            acc[i] = Float32(0.0)

        copy_atom_bf16_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        for k in cutlass.range_constexpr(0, num_topk, 1):
            score_value = Float32(1.0)
            if cutlass.const_expr(topk_score is not None):
                score_value = Float32(topk_score[token_idx, Int32(k)])
            score_pair = (score_value, score_value)

            in_regs = cute.make_rmem_tensor(
                (BF16_HIDDEN_PER_THREAD,),
                cutlass.BFloat16,
            )
            for i in cutlass.range_constexpr(0, BF16_HIDDEN_PER_THREAD, 1):
                in_regs[i] = cutlass.BFloat16(0.0)
            in_row = combine_output[token_idx, Int32(k), None]
            in_tile = cute.local_tile(
                in_row,
                (BF16_HIDDEN_PER_THREAD,),
                (base_h // Int32(BF16_HIDDEN_PER_THREAD),),
            )
            in_aligned_iter = cute.make_ptr(
                in_tile.element_type,
                in_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=16,
            )
            in_tile = cute.make_tensor(in_aligned_iter, in_tile.layout)
            cute.copy(
                copy_atom_bf16_vec,
                cute.coalesce(in_tile),
                cute.coalesce(in_regs),
            )

            for pair_i in cutlass.range_constexpr(
                0,
                BF16_HIDDEN_PER_THREAD // 2,
                1,
            ):
                val_pair = (
                    Float32(in_regs[2 * pair_i]),
                    Float32(in_regs[2 * pair_i + 1]),
                )
                old_acc_pair = (acc[2 * pair_i], acc[2 * pair_i + 1])
                if cutlass.const_expr(topk_score is not None):
                    acc_pair = cute.arch.fma_packed_f32x2(
                        val_pair,
                        score_pair,
                        old_acc_pair,
                    )
                else:
                    acc_pair = cute.arch.add_packed_f32x2(
                        old_acc_pair,
                        val_pair,
                    )
                acc[2 * pair_i] = acc_pair[0]
                acc[2 * pair_i + 1] = acc_pair[1]

        out_row = reduced_output[token_idx, None]
        out_tile = cute.local_tile(
            out_row,
            (BF16_HIDDEN_PER_THREAD,),
            (base_h // Int32(BF16_HIDDEN_PER_THREAD),),
        )
        if cutlass.const_expr(store_dtype == "bf16"):
            out_regs = cute.make_rmem_tensor(
                (BF16_HIDDEN_PER_THREAD,),
                cutlass.BFloat16,
            )
            out_regs.store(acc.load().to(cutlass.BFloat16))
            out_aligned_iter = cute.make_ptr(
                out_tile.element_type,
                out_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=16,
            )
            out_tile = cute.make_tensor(out_aligned_iter, out_tile.layout)
            cute.copy(
                copy_atom_bf16_vec,
                cute.coalesce(out_regs),
                cute.coalesce(out_tile),
            )
        else:
            for i in cutlass.range_constexpr(0, BF16_HIDDEN_PER_THREAD, 1):
                out_tile[i] = acc[i]


@cute.kernel
def topk_reduce_nvfp4_vec_kernel(
    combine_output: cute.Tensor,
    combine_sf: cute.Tensor,
    topk_score: Optional[cute.Tensor],
    reduced_output: cute.Tensor,
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    hidden_blocks: cutlass.Constexpr[int],
):
    """Decode per-16 E2M1 blocks and reduce one 16-hidden vector per thread."""

    linear_block_idx, _, _ = cute.arch.block_idx()
    hidden_vec_block_idx = linear_block_idx % Int32(hidden_blocks)
    token_idx = linear_block_idx // Int32(hidden_blocks)
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]
    vec_idx = hidden_vec_block_idx * block_dim + tid
    base_h = vec_idx * Int32(NVFP4_HIDDEN_PER_THREAD)

    if base_h < Int32(hidden):
        acc = cute.make_rmem_tensor((NVFP4_HIDDEN_PER_THREAD,), cutlass.Float32)
        for i in cutlass.range_constexpr(0, NVFP4_HIDDEN_PER_THREAD, 1):
            acc[i] = Float32(0.0)

        copy_atom_fp4 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float4E2M1FN,
            num_bits_per_copy=64,
        )

        for k in cutlass.range_constexpr(0, num_topk, 1):
            score_value = Float32(1.0)
            if cutlass.const_expr(topk_score is not None):
                score_value = Float32(topk_score[token_idx, Int32(k)])
            score_pair = (score_value, score_value)

            in_regs = cute.make_rmem_tensor(
                (NVFP4_HIDDEN_PER_THREAD,), cutlass.Float4E2M1FN
            )
            in_row = combine_output[token_idx, Int32(k), None]
            in_tile = cute.local_tile(
                in_row,
                (NVFP4_HIDDEN_PER_THREAD,),
                (base_h // Int32(NVFP4_HIDDEN_PER_THREAD),),
            )
            in_aligned_iter = cute.make_ptr(
                in_tile.element_type,
                in_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=8,
            )
            in_tile = cute.make_tensor(in_aligned_iter, in_tile.layout)
            cute.copy(
                copy_atom_fp4,
                cute.coalesce(in_tile),
                cute.coalesce(in_regs),
            )
            values = cute.make_rmem_tensor((NVFP4_HIDDEN_PER_THREAD,), cutlass.Float32)
            _cvt_e2m1_to_fp32(in_regs, values)

            scale = Float32(
                combine_sf[
                    token_idx,
                    Int32(k),
                    base_h // Int32(NVFP4_HIDDEN_PER_THREAD),
                ]
            ) * Float32(Nvfp4E2M1RcpLimit)
            scale_pair = (scale, scale)
            for pair_i in cutlass.range_constexpr(0, NVFP4_HIDDEN_PER_THREAD // 2, 1):
                dequant_pair = cute.arch.mul_packed_f32x2(
                    (values[2 * pair_i], values[2 * pair_i + 1]),
                    scale_pair,
                )
                old_acc_pair = (acc[2 * pair_i], acc[2 * pair_i + 1])
                if cutlass.const_expr(topk_score is not None):
                    acc_pair = cute.arch.fma_packed_f32x2(
                        dequant_pair,
                        score_pair,
                        old_acc_pair,
                    )
                else:
                    acc_pair = cute.arch.add_packed_f32x2(
                        old_acc_pair,
                        dequant_pair,
                    )
                acc[2 * pair_i] = acc_pair[0]
                acc[2 * pair_i + 1] = acc_pair[1]

        out_row = reduced_output[token_idx, None]
        out_tile = cute.local_tile(
            out_row,
            (NVFP4_HIDDEN_PER_THREAD,),
            (base_h // Int32(NVFP4_HIDDEN_PER_THREAD),),
        )
        out_regs = cute.make_rmem_tensor((NVFP4_HIDDEN_PER_THREAD,), cutlass.BFloat16)
        out_regs.store(acc.load().to(cutlass.BFloat16))
        out_aligned_iter = cute.make_ptr(
            out_tile.element_type,
            out_tile.iterator.toint(),
            AddressSpace.gmem,
            assumed_align=32,
        )
        out_tile = cute.make_tensor(out_aligned_iter, out_tile.layout)
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=256,
            ),
            cute.coalesce(out_regs),
            cute.coalesce(out_tile),
        )


def _validate_tensors(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    *,
    combine_sf: Optional[torch.Tensor] = None,
    combine_format: str = "bf16",
) -> Tuple[int, int, int, str]:
    combine_format = _normalize_combine_format(combine_format)
    if combine_output.dim() != 3:
        raise ValueError(
            "combine_output must be three-dimensional, got "
            f"{tuple(combine_output.shape)}."
        )
    if reduced_output.dim() != 2:
        raise ValueError(
            f"reduced_output must have shape (T, H), got {tuple(reduced_output.shape)}."
        )
    if reduced_output.dtype != torch.bfloat16:
        raise TypeError(f"reduced_output must be BF16, got {reduced_output.dtype}.")
    if not combine_output.is_cuda or not reduced_output.is_cuda:
        raise ValueError("combine_output and reduced_output must be CUDA tensors.")
    if combine_output.device != reduced_output.device:
        raise ValueError("combine_output and reduced_output must share a device.")

    tokens, hidden = map(int, reduced_output.shape)
    num_topk = int(combine_output.shape[1])
    if tokens <= 0 or num_topk <= 0 or hidden <= 0:
        raise ValueError(
            "top-k reduction dimensions must be positive, got "
            f"tokens={tokens}, num_topk={num_topk}, hidden={hidden}."
        )
    if int(combine_output.shape[0]) != tokens:
        raise ValueError(
            "combine_output and reduced_output must have the same token count."
        )
    if combine_output.stride(-1) != 1 or reduced_output.stride(-1) != 1:
        raise ValueError("top-k reduction requires contiguous hidden dimensions.")
    if reduced_output.stride(0) % BF16_HIDDEN_PER_THREAD != 0:
        raise ValueError("reduced_output rows must preserve 16-byte alignment.")

    if combine_format == "bf16":
        if combine_output.dtype != torch.bfloat16:
            raise TypeError(
                f"BF16 combine_output must be BF16, got {combine_output.dtype}."
            )
        if tuple(combine_output.shape) != (tokens, num_topk, hidden):
            raise ValueError(
                "BF16 combine_output shape must be "
                f"{(tokens, num_topk, hidden)}, got {tuple(combine_output.shape)}."
            )
        if combine_sf is not None:
            raise ValueError("BF16 combine does not use combine_sf.")
        if hidden % BF16_HIDDEN_PER_THREAD != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by {BF16_HIDDEN_PER_THREAD}."
            )
        if combine_output.stride(-2) % BF16_HIDDEN_PER_THREAD != 0:
            raise ValueError(
                "BF16 combine_output rows must preserve 16-byte alignment."
            )
        if (
            combine_output.data_ptr() % 16
            or combine_output.stride(0) * combine_output.element_size() % 16
            or reduced_output.data_ptr() % 16
        ):
            raise ValueError("BF16 combine tensors must preserve 16-byte alignment.")
    else:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4_dtype is None or combine_output.dtype != fp4_dtype:
            raise TypeError(
                "NVFP4 combine_output must use torch.float4_e2m1fn_x2, got "
                f"{combine_output.dtype}."
            )
        if hidden % NVFP4_HIDDEN_PER_THREAD != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by {NVFP4_HIDDEN_PER_THREAD}."
            )
        expected_data_shape = (tokens, num_topk, hidden // 2)
        if tuple(combine_output.shape) != expected_data_shape:
            raise ValueError(
                "packed NVFP4 combine_output shape must be "
                f"{expected_data_shape}, got {tuple(combine_output.shape)}."
            )
        if combine_output.stride(-2) % (NVFP4_HIDDEN_PER_THREAD // 2) != 0:
            raise ValueError(
                "NVFP4 combine_output rows must preserve 8-byte alignment."
            )
        if (
            combine_output.data_ptr() % 8
            or combine_output.stride(0) * combine_output.element_size() % 8
            or reduced_output.data_ptr() % 32
            or reduced_output.stride(0) * reduced_output.element_size() % 32
        ):
            raise ValueError(
                "NVFP4 combine data/output must preserve 8/32-byte alignment."
            )
        if combine_sf is None:
            raise ValueError("NVFP4 combine requires a BF16 combine_sf tensor.")
        expected_sf_shape = (tokens, num_topk, hidden // 16)
        if tuple(combine_sf.shape) != expected_sf_shape:
            raise ValueError(
                f"combine_sf shape must be {expected_sf_shape}, got "
                f"{tuple(combine_sf.shape)}."
            )
        if combine_sf.dtype != torch.bfloat16 or not combine_sf.is_cuda:
            raise TypeError("combine_sf must be a CUDA BF16 tensor.")
        if combine_sf.device != combine_output.device:
            raise ValueError("combine_sf must share combine_output's device.")
        if combine_sf.stride(-1) != 1:
            raise ValueError("combine_sf blocks must be contiguous.")

    if topk_score is not None:
        if topk_score.shape != (tokens, num_topk):
            raise ValueError(
                f"topk_score shape must be {(tokens, num_topk)}, got "
                f"{tuple(topk_score.shape)}."
            )
        if topk_score.dtype != torch.float32 or not topk_score.is_cuda:
            raise TypeError("topk_score must be a CUDA FP32 tensor.")
        if topk_score.device != combine_output.device:
            raise ValueError("topk_score must share combine_output's device.")
    return tokens, num_topk, hidden, combine_format


def _infer_assumed_align(tensor: torch.Tensor, max_align: int = 16) -> int:
    ptr = int(tensor.data_ptr())
    for align in (16, 8, 4, 2, 1):
        if align <= max_align and ptr % align == 0:
            return align
    return 1


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    cute_tensor = cutlass_torch.from_dlpack(
        tensor, assumed_align=_infer_assumed_align(tensor)
    )
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def compile_topk_reduce(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    *,
    combine_sf: Optional[torch.Tensor] = None,
    combine_format: str = "bf16",
    threads: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
    enable_iket: bool = False,
):
    """Compile a shape- and format-specialized top-k reduction launcher."""
    tokens, num_topk, hidden, combine_format = _validate_tensors(
        combine_output,
        reduced_output,
        topk_score,
        combine_sf=combine_sf,
        combine_format=combine_format,
    )
    if threads is None:
        threads = (
            BF16_VECTOR_THREADS if combine_format == "bf16" else NVFP4_VECTOR_THREADS
        )
    if threads <= 0:
        raise ValueError(f"threads must be positive, got {threads}.")
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    combine_cute = _to_cute_tensor(combine_output)
    combine_sf_cute = _to_cute_tensor(combine_sf) if combine_sf is not None else None
    reduced_cute = _to_cute_tensor(reduced_output)
    topk_score_cute = _to_cute_tensor(topk_score) if topk_score is not None else None
    hidden_blocks, launch_grid = _topk_reduce_launch_geometry(
        tokens,
        hidden,
        threads,
        (
            BF16_HIDDEN_PER_THREAD
            if combine_format == "bf16"
            else NVFP4_HIDDEN_PER_THREAD
        ),
    )

    @cute.jit
    def _launcher(
        combine_cute: cute.Tensor,
        combine_sf_cute: Optional[cute.Tensor],
        reduced_cute: cute.Tensor,
        topk_score_cute: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(combine_format == "bf16"):
            topk_reduce_bf16_vec_kernel(
                combine_cute,
                topk_score_cute,
                reduced_cute,
                num_topk=num_topk,
                hidden=hidden,
                hidden_blocks=hidden_blocks,
                store_dtype="bf16",
            ).launch(
                grid=launch_grid,
                block=[threads, 1, 1],
                stream=stream,
            )
        else:
            topk_reduce_nvfp4_vec_kernel(
                combine_cute,
                combine_sf_cute,
                topk_score_cute,
                reduced_cute,
                num_topk=num_topk,
                hidden=hidden,
                hidden_blocks=hidden_blocks,
            ).launch(
                grid=launch_grid,
                block=[threads, 1, 1],
                stream=stream,
            )

    compile_kwargs = {}
    if enable_iket:
        compile_kwargs["options"] = "iket"
    compiled = cute.compile(
        _launcher,
        combine_cute,
        combine_sf_cute,
        reduced_cute,
        topk_score_cute,
        stream,
        **compile_kwargs,
    )
    return (
        compiled,
        combine_cute,
        combine_sf_cute,
        reduced_cute,
        topk_score_cute,
        stream,
    )


__all__ = [
    "compile_topk_reduce",
    "topk_reduce_bf16_vec_kernel",
    "topk_reduce_nvfp4_vec_kernel",
]
