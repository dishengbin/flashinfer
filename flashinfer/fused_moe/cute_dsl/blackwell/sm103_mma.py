# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# K96 circular-descriptor construction follows dense_blockscaled_gemm_sm103.py.
# Copyright (c) 2026 by FlashInfer team.

"""SM103 K96 mainloop over the existing packed NVFP4 and block-16 SF layouts.

A K768 group streams three K256 TMA tiles through the input ring and issues
eight K96 MMAs. MMA2 and MMA5 cross buffer boundaries; their ordered
consumer releases protect each source through its final asynchronous MMA.
Twelve SF values are recycled after each pair of MMAs, keeping accumulator
turnover independent of a full K768 scale-factor allocation.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


def packed_smem_view(tensor, num_stages):
    # A byte view must restore its swizzle explicitly: recast_tensor drops the
    # swizzle carried by the sub-byte pointer.
    packed = cute.recast_tensor(tensor, cutlass.Uint8)
    rows = cute.size(packed.shape[0][0])
    layout = cute.make_layout(
        ((rows, 16), 1, 8, num_stages),
        stride=((128, 1), 0, 16, packed.stride[3]),
    )
    ptr = cute.recast_ptr(
        packed.iterator, dtype=cutlass.Uint8, swizzle_=cute.make_swizzle(3, 4, 3)
    )
    return cute.make_tensor(ptr, layout)


def make_tma_atom(op, tensor, tiled_mma, num_multicast, operand):
    # Byte TMA preserves packed FP4. The FP4 TMA format uses a different
    # swizzle atom and cannot be consumed by a K96 byte descriptor.
    packed = cute.recast_tensor(tensor, cutlass.Uint8)
    if operand == "a":
        rows = cute.size(tiled_mma.tv_layout_A[1][0])
    else:
        rows = cute.size(tiled_mma.tv_layout_B[1][0])
    smem = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((rows, 128), stride=(128, 1))
    )
    return cpasync.make_tiled_tma_atom(
        op, packed, smem, (rows, 128), num_multicast, internal_type=cutlass.Uint8
    )


def sf_mma_view(mma, ptr, mma_tiler, rows, is_b):
    # Each K96 MMA consumes six block-16 scales, starting at byte 0 or 2
    # of a TMEM word. Two MMAs share three four-byte SF chunks.
    layout = cute.make_layout(((((32, 4), rows // 128), (16, 1)), 1, 12))
    if is_b:
        tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
            mma, (*mma_tiler[:2], 192), 16, layout
        )
    else:
        tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
            mma, (*mma_tiler[:2], 192), 16, layout
        )
    return cute.make_tensor(ptr, tmem_layout)


def mma_issue(mma, acc, a_cur, a_next, b_cur, b_next):
    # next_src encodes the circular-buffer boundary used by K96.
    a_desc = tcgen05.make_umma_smem_desc(
        a_cur.iterator, a_cur.layout, "k", next_src=a_next.iterator
    )
    b_desc = tcgen05.make_umma_smem_desc(
        b_cur.iterator, b_cur.layout, "k", next_src=b_next.iterator
    )
    layout = cute.make_layout(1, stride=0)
    cute.mma_atom_call(
        mma,
        acc,
        cute.make_tensor(a_desc, layout),
        cute.make_tensor(b_desc, layout),
        acc,
    )


@cute.jit
def mainloop(
    native_mma,
    tCtAcc,
    native_sA,
    native_sB,
    tiled_copy_s2t_sfa,
    tCsSFA_compact_s2t,
    tCtSFA_compact_s2t,
    tiled_copy_s2t_sfb,
    tCsSFB_compact_s2t,
    tCtSFB_compact_s2t,
    tCtSFA_native,
    tCtSFB_native,
    ab_pipeline,
    ab_consumer_state,
    peek_ab_full_status,
    k_tile_cnt,
):
    native_mma.set(tcgen05.Field.ACCUMULATE, False)
    for k_group in cutlass.range(k_tile_cnt // 3):
        ab_state_0 = ab_consumer_state.clone()
        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
        ab_consumer_state.advance()
        peek_ab_full_status = cutlass.Boolean(1)
        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 0, ab_state_0.index],
            tCtSFA_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 0, ab_state_0.index],
            tCtSFB_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 1, ab_state_0.index],
            tCtSFA_compact_s2t[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 1, ab_state_0.index],
            tCtSFB_compact_s2t[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 2, ab_state_0.index],
            tCtSFA_compact_s2t[None, None, None, 2],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 2, ab_state_0.index],
            tCtSFB_compact_s2t[None, None, None, 2],
        )
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (0, 0)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (0, 0)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 0, ab_state_0.index],
            native_sA[None, 0, 0, ab_state_0.index],
            native_sB[None, 0, 0, ab_state_0.index],
            native_sB[None, 0, 0, ab_state_0.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (2, 1)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (2, 1)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 3, ab_state_0.index],
            native_sA[None, 0, 0, ab_state_0.index],
            native_sB[None, 0, 3, ab_state_0.index],
            native_sB[None, 0, 0, ab_state_0.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        ab_state_1 = ab_consumer_state.clone()
        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
        ab_consumer_state.advance()
        peek_ab_full_status = cutlass.Boolean(1)
        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 3, ab_state_0.index],
            tCtSFA_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 3, ab_state_0.index],
            tCtSFB_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 0, ab_state_1.index],
            tCtSFA_compact_s2t[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 0, ab_state_1.index],
            tCtSFB_compact_s2t[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 1, ab_state_1.index],
            tCtSFA_compact_s2t[None, None, None, 2],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 1, ab_state_1.index],
            tCtSFB_compact_s2t[None, None, None, 2],
        )
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (0, 0)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (0, 0)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 6, ab_state_0.index],
            native_sA[None, 0, 0, ab_state_1.index],
            native_sB[None, 0, 6, ab_state_0.index],
            native_sB[None, 0, 0, ab_state_1.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        ab_pipeline.consumer_release(ab_state_0)
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (2, 1)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (2, 1)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 1, ab_state_1.index],
            native_sA[None, 0, 0, ab_state_1.index],
            native_sB[None, 0, 1, ab_state_1.index],
            native_sB[None, 0, 0, ab_state_1.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 2, ab_state_1.index],
            tCtSFA_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 2, ab_state_1.index],
            tCtSFB_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 3, ab_state_1.index],
            tCtSFA_compact_s2t[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 3, ab_state_1.index],
            tCtSFB_compact_s2t[None, None, None, 1],
        )
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (0, 0)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (0, 0)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 4, ab_state_1.index],
            native_sA[None, 0, 0, ab_state_1.index],
            native_sB[None, 0, 4, ab_state_1.index],
            native_sB[None, 0, 0, ab_state_1.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        ab_state_2 = ab_consumer_state.clone()
        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
        ab_consumer_state.advance()
        peek_ab_full_status = cutlass.Boolean(1)
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 0, ab_state_2.index],
            tCtSFA_compact_s2t[None, None, None, 2],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 0, ab_state_2.index],
            tCtSFB_compact_s2t[None, None, None, 2],
        )
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (2, 1)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (2, 1)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 7, ab_state_1.index],
            native_sA[None, 0, 0, ab_state_2.index],
            native_sB[None, 0, 7, ab_state_1.index],
            native_sB[None, 0, 0, ab_state_2.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        ab_pipeline.consumer_release(ab_state_1)
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 1, ab_state_2.index],
            tCtSFA_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 1, ab_state_2.index],
            tCtSFB_compact_s2t[None, None, None, 0],
        )
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 2, ab_state_2.index],
            tCtSFA_compact_s2t[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 2, ab_state_2.index],
            tCtSFB_compact_s2t[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t[None, None, None, 3, ab_state_2.index],
            tCtSFA_compact_s2t[None, None, None, 2],
        )
        cute.copy(
            tiled_copy_s2t_sfb,
            tCsSFB_compact_s2t[None, None, None, 3, ab_state_2.index],
            tCtSFB_compact_s2t[None, None, None, 2],
        )
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (0, 0)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (0, 0)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 2, ab_state_2.index],
            native_sA[None, 0, 0, ab_state_2.index],
            native_sB[None, 0, 2, ab_state_2.index],
            native_sB[None, 0, 0, ab_state_2.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        native_mma.set(tcgen05.Field.SFA, tCtSFA_native[None, None, (2, 1)].iterator)
        native_mma.set(tcgen05.Field.SFB, tCtSFB_native[None, None, (2, 1)].iterator)
        mma_issue(
            native_mma,
            tCtAcc[None, 0, 0],
            native_sA[None, 0, 5, ab_state_2.index],
            native_sA[None, 0, 0, ab_state_2.index],
            native_sB[None, 0, 5, ab_state_2.index],
            native_sB[None, 0, 0, ab_state_2.index],
        )
        native_mma.set(tcgen05.Field.ACCUMULATE, True)
        ab_pipeline.consumer_release(ab_state_2)
        if k_group + 1 < k_tile_cnt // 3:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
    return ab_consumer_state
