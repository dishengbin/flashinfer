# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in two-kernel NVFP4 FC2: contiguous 2-D TMA stores, then finalize.

This internal experiment retains the explored 128x64 C subtiles and four C
buffers. Arbitrary routing cannot be represented by one 2-D TMA destination,
so the intermediate allocation and the finalizer are part of this operation.
The optional dedicated store warp and grouped raster adapt the
fc2_optimization branch's producer/consumer handoff to 2-D TMA stores.
"""

from typing import Any, Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import get_max_active_clusters, make_ptr
from .blackwell.grouped_gemm_tma import Sm100GroupedGemmTmaKernel
from .blackwell.moe_finalize import moe_unpermute

_cache: Dict[Tuple, Any] = {}


def grouped_gemm_tma(
    *,
    a,
    b,
    a_scale,
    b_scale,
    alpha,
    tile_idx_to_expert_idx,
    num_non_exiting_tiles,
    tile_idx_to_mn_limit,
    permuted_idx_to_expanded_idx,
    token_final_scales,
    out,
    config=(4, 0, False, 1),
    enable_pdl=True,
):
    """Write alpha-scaled BF16 expert outputs at A's padded expert offsets."""
    if a.dtype != torch.uint8 or b.dtype != torch.uint8 or out.dtype != torch.bfloat16:
        raise TypeError("TMA FC2 currently requires packed NVFP4 and BF16 output")
    if torch.cuda.get_device_capability(a.device) not in ((10, 0), (10, 3)):
        raise NotImplementedError("TMA FC2 currently targets SM100/SM103")
    m, k = a.shape[0], a.shape[1] * 2
    experts, n = b.shape[:2]
    if m % 256 or k % 256 or n % 256 or out.shape != (m, n):
        raise ValueError("TMA FC2 requires M/N/K multiples of 256 and padded output")
    c_stages, store_warps, raster_m, swizzle = config
    if c_stages not in (4, 8) or store_warps not in (0, 1) or swizzle not in (1, 2, 4):
        raise ValueError("unsupported TMA FC2 stage/store/raster configuration")
    tokens, top_k = token_final_scales.shape
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    max_clusters = get_max_active_clusters(2)
    tensors = (
        a,
        b,
        a_scale,
        b_scale,
        out,
        alpha,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        token_final_scales,
    )
    types = (
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Float32,
    )
    pointers = tuple(
        make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for dtype, tensor in zip(types, tensors, strict=True)
    ) + (None,)
    key = (a.device.index, tuple(config), top_k, enable_pdl, max_clusters)
    compiled = _cache.get(key)
    if compiled is None:
        kernel = Sm100GroupedGemmTmaKernel(
            sf_vec_size=16,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            raster_along_m=raster_m,
            enable_pdl=enable_pdl,
            use_fused_finalize=False,
            num_c_stage=c_stages,
            num_store_warps=store_warps,
            swizzle_size=swizzle,
        )
        compiled = cute.compile(
            kernel.wrapper,
            *pointers,
            m,
            n,
            k,
            experts,
            tokens,
            top_k,
            tile_size=256,
            scaling_vector_size=16,
            max_active_clusters=max_clusters,
            stream=stream,
        )
        _cache[key] = compiled
    compiled(*pointers, m, n, k, experts, tokens, top_k, stream)


def tma_fc2(
    *,
    a,
    b,
    a_scale,
    b_scale,
    alpha,
    tile_idx_to_expert_idx,
    num_non_exiting_tiles,
    tile_idx_to_mn_limit,
    permuted_idx_to_expanded_idx,
    expanded_idx_to_permuted_idx,
    token_final_scales,
    out,
    config=(4, 0, False, 1),
    workspace=None,
    enable_pdl=True,
):
    """Complete FC2 including routing-weight reduction and its workspace."""
    if token_final_scales.dtype != torch.float32 or alpha.dtype != torch.float32:
        raise TypeError("TMA FC2 currently requires FP32 router scales and alpha")
    if workspace is None:
        workspace = torch.empty(
            (a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=a.device
        )
    grouped_gemm_tma(
        a=a,
        b=b,
        a_scale=a_scale,
        b_scale=b_scale,
        alpha=alpha,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        token_final_scales=token_final_scales,
        out=workspace,
        config=config,
        enable_pdl=enable_pdl,
    )
    moe_unpermute(
        permuted_input=workspace,
        output=out,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        topk_scales=token_final_scales,
        num_tokens=out.shape[0],
        top_k=token_final_scales.shape[1],
        enable_pdl=enable_pdl,
    )
    return out
