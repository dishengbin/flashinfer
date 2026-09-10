# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rank-local BF16 combine publication for the SM120 PCIe EP path.

K2's last contributor reduces device-local BF16 route rows and posts a rank
partial. K3 waits on its system-release ready word and reduces the live rank
partials. This module also provides a standalone route-map aggregator for
diagnostics; the production graph does not launch it. No PCIe atomics are used.
"""

from __future__ import annotations

from typing import Any, Optional

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cutlass_dsl import Float32, Int32

from .moe_utils import spin_wait


AGGREGATE_THREADS = 256
REDUCE_THREADS = 512
REDUCE_HIDDEN_PER_THREAD = 8


def _to_cute_tensor(tensor: torch.Tensor, *, assumed_align: int = 16) -> cute.Tensor:
    result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    return result.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))


@cute.kernel
def rank_local_aggregate_bf16_kernel(
    route_output: cute.Tensor,
    route_map: cute.Tensor,
    combine_output: cute.Tensor,
    ready_flags: cute.Tensor,
    peer_rank_ptr_mapper,
    tokens: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
    local_rank: cutlass.Constexpr[int],
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
):
    """Reduce and publish one source token's local expert contributions."""

    source_token_key, _, _ = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]
    src_rank = source_token_key // Int32(tokens)
    src_token = source_token_key - src_rank * Int32(tokens)
    route_count = Int32(0)
    for topk in cutlass.range_constexpr(num_topk):
        if route_map[source_token_key, Int32(topk)] >= Int32(0):
            route_count += Int32(1)

    if route_count > Int32(0):
        local_row = cute.slice_(
            combine_output,
            (src_token, Int32(local_rank), None),
        )
        peer_row = cute.make_tensor(
            peer_rank_ptr_mapper.ptr_map_to_rank(local_row.iterator, src_rank),
            local_row.layout,
        )
        hidden_idx = tid
        while hidden_idx < Int32(hidden):
            value = Float32(0.0)
            for topk in cutlass.range_constexpr(num_topk):
                route = route_map[source_token_key, Int32(topk)]
                if route >= Int32(0):
                    value += Float32(route_output[route, Int32(0), hidden_idx])
            peer_row[hidden_idx] = value.to(cutlass.BFloat16)
            hidden_idx += block_dim

    # DeepEP-style single-writer publication: every expert rank publishes one
    # word, including the no-route case.  K3 can distinguish an absent partial
    # from a peer that has not completed without dense zero rows.
    cute.arch.sync_threads()
    if tid == Int32(0):
        cute.arch.fence_acq_rel_sys()
        local_flag = cute.slice_(
            ready_flags,
            (src_token, Int32(local_rank)),
        )
        peer_flag = peer_rank_ptr_mapper.ptr_map_to_rank(local_flag.iterator, src_rank)
        ready_value = Int32(1)
        if route_count > Int32(0):
            ready_value = Int32(2)
        cute.arch.store(
            peer_flag,
            ready_value,
            sem="release",
            scope="sys",
        )


@cute.kernel
def rank_partial_reduce_bf16_kernel(
    combine_output: cute.Tensor,
    ready_flags: cute.Tensor,
    reduced_output: cute.Tensor,
    topk_idx: Optional[cute.Tensor],
    world_size: cutlass.Constexpr[int],
    num_topk: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    hidden_blocks: cutlass.Constexpr[int],
):
    """Wait for and reduce the sparse rank partials for one source token."""

    linear_block_idx, _, _ = cute.arch.block_idx()
    hidden_block = linear_block_idx % Int32(hidden_blocks)
    token_idx = linear_block_idx // Int32(hidden_blocks)
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]

    rank_mask = Int32((1 << world_size) - 1)
    if cutlass.const_expr(topk_idx is not None):
        rank_mask = Int32(0)
        for slot in cutlass.range_constexpr(num_topk):
            expert = Int32(topk_idx[token_idx, Int32(slot)])
            if expert >= Int32(0):
                rank_mask |= Int32(1) << (expert // Int32(num_experts_per_rank))
    if tid == Int32(0):
        for rank in cutlass.range_constexpr(world_size):
            if rank_mask & Int32(1 << rank):
                spin_wait(
                    ready_flags.iterator + token_idx * Int32(world_size) + Int32(rank),
                    lambda value: value >= Int32(1),
                    fail_sleep_cycles=200,
                )
    cute.arch.sync_threads()

    # Each scalar load/store addresses consecutive BF16 values across lanes.
    hidden_base = hidden_block * block_dim * Int32(REDUCE_HIDDEN_PER_THREAD) + tid
    if hidden_base < Int32(hidden):
        for elem in cutlass.range_constexpr(REDUCE_HIDDEN_PER_THREAD):
            hidden_idx = hidden_base + Int32(elem) * block_dim
            if hidden_idx < Int32(hidden):
                value = Float32(0.0)
                for rank in cutlass.range_constexpr(world_size):
                    # The standalone aggregator publishes 1 for absent rows;
                    # K2 publishes only live rows (2), selected by topk_idx.
                    live = (rank_mask & Int32(1 << rank)) != Int32(0)
                    if cutlass.const_expr(topk_idx is None):
                        live = ready_flags[token_idx, Int32(rank)] == Int32(2)
                    if live:
                        value += Float32(
                            combine_output[token_idx, Int32(rank), hidden_idx]
                        )
                reduced_output[token_idx, hidden_idx] = value.to(cutlass.BFloat16)


def compile_rank_local_combine(
    route_output: torch.Tensor,
    route_map: torch.Tensor,
    combine_output: torch.Tensor,
    ready_flags: torch.Tensor,
    reduced_output: torch.Tensor,
    peer_rank_ptr_mapper_host: Any,
    *,
    world_size: int,
    local_rank: int,
    stream: cuda.CUstream,
    topk_ids: torch.Tensor | None = None,
    num_experts_per_rank: int = 0,
) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    """Compile rank K3, plus a standalone aggregator unless K2 publishes.

    Supplying topk_ids selects K2 publication: there is no aggregate launch,
    and K3 waits only for ranks to which this token actually routed.
    """

    if route_output.dtype != torch.bfloat16 or route_output.ndim != 3:
        raise TypeError("rank-local route_output must be a 3-D BF16 tensor")
    if route_map.dtype != torch.int32 or route_map.ndim != 2:
        raise TypeError("rank-local route_map must be a 2-D Int32 tensor")
    if combine_output.dtype != torch.bfloat16 or combine_output.ndim != 3:
        raise TypeError("rank-local combine_output must be a 3-D BF16 tensor")
    if ready_flags.dtype != torch.int32 or ready_flags.ndim != 2:
        raise TypeError("rank-local ready_flags must be a 2-D Int32 tensor")
    if reduced_output.dtype != torch.bfloat16 or reduced_output.ndim != 2:
        raise TypeError("rank-local reduced_output must be a 2-D BF16 tensor")

    tokens, hidden = map(int, reduced_output.shape)
    num_topk = int(combine_output.shape[1])
    if tuple(route_output.shape[1:]) != (1, hidden):
        raise ValueError("rank-local route_output shape does not match output")
    if tuple(route_map.shape) != (world_size * tokens, num_topk):
        raise ValueError("rank-local route_map shape does not match EP geometry")
    if tuple(combine_output.shape) != (tokens, num_topk, hidden):
        raise ValueError("rank-local combine_output shape does not match output")
    if tuple(ready_flags.shape) != (tokens, world_size):
        raise ValueError("rank-local ready_flags shape does not match EP geometry")
    if not 0 <= local_rank < world_size or world_size > num_topk:
        raise ValueError("invalid rank-local combine rank/top-k geometry")
    if topk_ids is not None:
        if tuple(topk_ids.shape) != (tokens, num_topk) or num_experts_per_rank <= 0:
            raise ValueError("rank-local topk_ids shape or expert geometry is invalid")

    route_cute = _to_cute_tensor(route_output, assumed_align=16)
    map_cute = _to_cute_tensor(route_map, assumed_align=16)
    combine_cute = _to_cute_tensor(combine_output, assumed_align=16)
    ready_cute = _to_cute_tensor(ready_flags, assumed_align=16)
    reduced_cute = _to_cute_tensor(reduced_output, assumed_align=16)
    topk_cute = _to_cute_tensor(topk_ids) if topk_ids is not None else None
    hidden_blocks = (hidden + REDUCE_THREADS * REDUCE_HIDDEN_PER_THREAD - 1) // (
        REDUCE_THREADS * REDUCE_HIDDEN_PER_THREAD
    )

    @cute.jit
    def _aggregate_launcher(
        route_cute: cute.Tensor,
        map_cute: cute.Tensor,
        combine_cute: cute.Tensor,
        ready_cute: cute.Tensor,
        peer_rank_ptr_mapper_host,
        stream: cuda.CUstream,
    ):
        peer_mapper = peer_rank_ptr_mapper_host.make_device_obj()
        rank_local_aggregate_bf16_kernel(
            route_cute,
            map_cute,
            combine_cute,
            ready_cute,
            peer_mapper,
            tokens=tokens,
            world_size=world_size,
            local_rank=local_rank,
            num_topk=num_topk,
            hidden=hidden,
        ).launch(
            grid=[world_size * tokens, 1, 1],
            block=[AGGREGATE_THREADS, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _reduce_launcher(
        combine_cute: cute.Tensor,
        ready_cute: cute.Tensor,
        reduced_cute: cute.Tensor,
        topk_cute,
        stream: cuda.CUstream,
    ):
        rank_partial_reduce_bf16_kernel(
            combine_cute,
            ready_cute,
            reduced_cute,
            topk_cute,
            world_size=world_size,
            num_topk=num_topk,
            num_experts_per_rank=num_experts_per_rank,
            hidden=hidden,
            hidden_blocks=hidden_blocks,
        ).launch(
            grid=[tokens * hidden_blocks, 1, 1],
            block=[REDUCE_THREADS, 1, 1],
            stream=stream,
        )

    aggregate_compiled = None
    if topk_ids is None:
        aggregate_compiled = cute.compile(
            _aggregate_launcher,
            route_cute,
            map_cute,
            combine_cute,
            ready_cute,
            peer_rank_ptr_mapper_host,
            stream,
        )
    reduce_compiled = cute.compile(
        _reduce_launcher,
        combine_cute,
        ready_cute,
        reduced_cute,
        topk_cute,
        stream,
    )
    aggregate_runtime = dict(
        route_cute=route_cute,
        map_cute=map_cute,
        combine_cute=combine_cute,
        ready_cute=ready_cute,
        peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
        stream=stream,
    )
    reduce_runtime = dict(
        combine_cute=combine_cute,
        ready_cute=ready_cute,
        reduced_cute=reduced_cute,
        topk_cute=topk_cute,
        stream=stream,
    )
    return aggregate_compiled, aggregate_runtime, reduce_compiled, reduce_runtime


__all__ = [
    "compile_rank_local_combine",
    "rank_local_aggregate_bf16_kernel",
    "rank_partial_reduce_bf16_kernel",
]
