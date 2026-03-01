# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Optional, List, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide

# -----------------------------------------------------------------------------
# FX Traceability Helpers
# -----------------------------------------------------------------------------

class EmptyForFx(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._is_leaf_module = True

    def forward(self, x):
        return torch.empty(x)

def get_tensor_for_fx(tensor_shape, dtype, name):
    return mpu.get_global_memory_buffer().get_tensor(tensor_shape, dtype, name)

torch.fx.wrap("get_tensor_for_fx")

@torch.fx.wrap
def all_gather_context_parallel(tensor: Tensor, cp_group: Union[torch.distributed.ProcessGroup, None], cp_size: int) -> Tensor:
    """
    A torch.fx traceable wrapper for all-gather.
    """
    if cp_size == 1:
        return tensor

    # 1. Check if we can run the real collective
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        input_shape = list(tensor.shape)
        output_shape = list(input_shape)
        output_shape[0] = input_shape[0] * cp_size

        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        
        try:
            torch.distributed.all_gather_into_tensor(output, tensor, group=cp_group)
            return output
        except Exception:
            pass

    # 2. Simulation Mode (Shape Propagation / Offline Tracing)
    repeat_dims = [1] * tensor.dim()
    repeat_dims[0] = cp_size
    return tensor.repeat(*repeat_dims)

# -----------------------------------------------------------------------------
# DotProductAttention
# -----------------------------------------------------------------------------

class DotProductAttention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = "all_gather",
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.cp_comm_type = cp_comm_type
        self.cp_size = self.config.context_parallel_size

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )

        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Forward pass compatible with torch.fx and Context Parallelism."""
        
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
        )
        assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

        ##NEST 
        # 1. Context Parallelism: All-Gather KV
        if self.cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
            key = all_gather_context_parallel(key, cp_group, self.cp_size)
            value = all_gather_context_parallel(value, cp_group, self.cp_size)

        sq, b, np_q, hn = query.shape
        sk = key.shape[0]

        # 2. Group Query Attention Handling
        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        
        if n_rep > 1:
            key = key.view(sk, b, self.num_query_groups_per_partition, 1, hn)
            key = key.expand(sk, b, self.num_query_groups_per_partition, n_rep, hn)
            key = key.reshape(sk, b, self.num_attention_heads_per_partition, hn)

            value = value.view(sk, b, self.num_query_groups_per_partition, 1, hn)
            value = value.expand(sk, b, self.num_query_groups_per_partition, n_rep, hn)
            value = value.reshape(sk, b, self.num_attention_heads_per_partition, hn)

        # 3. Compute Attention Scores
        query = query.reshape(sq, b * np_q, hn).transpose(0, 1)
        key = key.reshape(sk, b * np_q, hn).transpose(0, 1).transpose(1, 2)

        matmul_input_buffer = get_tensor_for_fx((b * np_q, sq, sk), query.dtype, "mpu")

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query,
            key,
            beta=0.0,
            alpha=self.softmax_scale,
        )

        attention_scores = matmul_result.view(b, np_q, sq, sk)

        # 4. Softmax & Dropout
        
        # [TRACE-SAFE FIX 1] Slice Mask Rows (Query Dim) to match 'sq'

        # [Trace-Safe Logic]
        if attention_mask is not None:
            # 1. Vertical Slice (Query Dim)
            # Slicing with symbolic 'sq' is allowed in FX.
            attention_mask = attention_mask[..., :sq, :]

            # 2. Horizontal Expansion (Key Dim) - The "Branchless" Way
            # instead of "if score_sk > mask_sk", we calculate the ratio.
            
            if self.cp_size > 1 and self.attn_mask_type == AttnMaskType.causal:
                score_sk = attention_scores.shape[-1] # Symbolic
                mask_sk = attention_mask.shape[-1]    # Symbolic
                
                # Calculate integer repeat factor (e.g., 8192 // 4096 = 2)
                # If they are equal, rep = 1.
                rep = score_sk // mask_sk 
                
                # "repeat" works with symbolic integers in FX
                attention_mask = attention_mask.repeat(1, 1, 1, rep)
                
                # Optional: Enforce exact size match (in case division wasn't clean)
                attention_mask = attention_mask[..., :score_sk]

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # 5. Compute Context
        value = value.reshape(sk, b * np_q, hn).transpose(0, 1)
        attention_probs = attention_probs.view(b * np_q, sq, sk)
        context = torch.bmm(attention_probs, value)

        # 6. Final Reshape
        context = context.transpose(0, 1).contiguous()
        context = context.view(sq, b, self.hidden_size_per_partition)

        return context