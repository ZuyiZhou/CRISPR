"""
Training-free visual token compression baselines for Qwen2.5-VL.

Ported from EffiVLM-Bench (Qwen2-VL) and adapted for Qwen2.5-VL-7B-Instruct.

NOTE ON PROVENANCE: this file was ported and adapted from the EffiVLM-Bench
project (see reference below). We have not been able to confirm EffiVLM-Bench's
license terms; the MIT license covering the rest of this repository (see
LICENSE / NOTICE.md) does NOT necessarily apply to this file's ported logic.
If you are the EffiVLM-Bench maintainers or know its license and this
attribution/redistribution needs correction, please open an issue.

Three methods:
  - VisionZipCompressor: attention-based dominant + cosine-similarity merged contextual tokens
  - PruMergePlusCompressor: IQR outlier detection + attention-weighted merge
  - AttentionCaptureHook: forward hook to capture Q/K at vision encoder block 30

Reference: EffiVLM-Bench/kv_cache_compression/qwen2vl_model.py
"""

import math
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# AttentionCaptureHook — captures attention weights at a specific vision block
# =============================================================================

class AttentionCaptureHook:
    """
    Forward hook for Qwen2_5_VLVisionSdpaAttention.

    Registers on model.visual.blocks[target_block_idx].attn to capture
    attention weights and K values during vision encoder forward pass.

    Qwen2.5-VL uses SDPA (no attn weights returned), so we manually compute
    Q @ K^T with the cu_seqlens mask.
    """

    def __init__(self, target_block_idx: int = 30):
        self.target_block_idx = target_block_idx
        self.attn_weights: Optional[torch.Tensor] = None  # [num_heads, seq_len, seq_len]
        self.metric_k: Optional[torch.Tensor] = None       # [seq_len, num_heads * head_dim]
        self._handle = None

    def register(self, model):
        """Register as forward hook on the target block's attention module."""
        attn_module = model.visual.blocks[self.target_block_idx].attn
        self._handle = attn_module.register_forward_hook(self._hook_fn, with_kwargs=True)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.attn_weights = None
        self.metric_k = None

    def _hook_fn(self, module, args, kwargs, output):
        """
        Hook called after Qwen2_5_VLVisionAttention.forward().

        PyTorch 2.10+ with_kwargs=True signature: (module, args, kwargs, output)

        Signature of forward:
            forward(hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None)

        We re-compute Q, K from hidden_states and the module's qkv projection,
        apply RoPE, then compute full attention weights.
        """
        # Parse arguments — hidden_states is args[0], cu_seqlens is args[1]
        hidden_states = args[0]
        cu_seqlens = args[1] if len(args) > 1 else kwargs.get("cu_seqlens")
        rotary_pos_emb = args[2] if len(args) > 2 else kwargs.get("rotary_pos_emb")
        position_embeddings = args[3] if len(args) > 3 else kwargs.get("position_embeddings")

        seq_length = hidden_states.shape[0]
        num_heads = module.num_heads

        # Recompute Q, K, V from the qkv linear
        qkv = module.qkv(hidden_states)  # [seq_len, 3 * dim]
        qkv = qkv.reshape(seq_length, 3, num_heads, -1).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)  # each [seq_len, num_heads, head_dim]

        # Apply RoPE — Qwen2.5-VL uses position_embeddings (cos, sin)
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # q, k: [seq_len, num_heads, head_dim] → transpose to [num_heads, seq_len, head_dim]
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)

        # Compute attention: [num_heads, seq_len, seq_len]
        head_dim = q_t.shape[-1]
        attn = torch.matmul(q_t.float(), k_t.float().transpose(1, 2)) / math.sqrt(head_dim)

        # Apply cu_seqlens mask: only allow attention within each sequence
        attention_mask = torch.full(
            [1, seq_length, seq_length], True, dtype=torch.bool, device=q.device
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = False
        attn = attn.masked_fill(attention_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)

        self.attn_weights = attn.detach()  # [num_heads, seq_len, seq_len]
        # Store k in [seq_len, num_heads * head_dim] format for PruMerge+
        self.metric_k = k.reshape(seq_length, -1).detach()  # [seq_len, num_heads * head_dim]


# =============================================================================
# VisionZipCompressor
# =============================================================================

class VisionZipCompressor:
    """
    VisionZip: dominant token selection by attention score + contextual token
    merging by cosine similarity.

    Reference: qwen2vl_model.py L1255-1346

    Args:
        dominant_ratio: fraction of budget allocated to dominant tokens (default 5.4/6.4)
        contextual_ratio: fraction of budget allocated to contextual tokens (default 1.0/6.4)
    """

    def __init__(self, dominant_ratio: float = 5.4 / 6.4, contextual_ratio: float = 1.0 / 6.4):
        self.dominant_ratio = dominant_ratio
        self.contextual_ratio = contextual_ratio

    @torch.no_grad()
    def __call__(
        self,
        image_tokens: torch.Tensor,          # [N_merged, D] — post-merger tokens
        attn_weights: torch.Tensor,           # [num_heads, N_pre, N_pre] — pre-merger attn
        metric_k: torch.Tensor,               # [N_pre, num_heads * head_dim] — pre-merger K
        budgets: float,                        # fraction of tokens to keep (e.g., 1/9)
        window_index: Optional[torch.Tensor] = None,  # window reorder index
        reverse_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            selected_tokens: [M, D] — compressed tokens in original order
            keep_indices: [M] — indices into image_tokens
        """
        N_merged = image_tokens.shape[0]
        D = image_tokens.shape[1]

        # Step 1: Compute per-pre-merger-token importance
        # attn_weights: [num_heads, N_pre, N_pre] → mean over heads and query dim
        attn_avg = attn_weights.mean(dim=0).mean(dim=0)  # [N_pre]

        # Step 2: Pool to post-merger resolution (4 pre-merger tokens → 1 post-merger)
        # But note: in Qwen2.5-VL with window attention, tokens may be reordered.
        # The merger operates on groups of spatial_merge_size^2 = 4 adjacent tokens.
        attention_sum = attn_avg.view(-1, 4).mean(dim=1)  # [N_merged]

        # Step 3: Determine budget split
        total_budget = max(1, int(N_merged * budgets))
        dominant_num = max(1, int(total_budget * self.dominant_ratio))
        contextual_num = max(1, total_budget - dominant_num)  # use remainder to avoid rounding loss

        # Step 4: Select dominant tokens by attention score
        all_indices = attention_sum.topk(dominant_num, dim=0).indices
        all_indices = all_indices.sort().values

        # Create mask: True = not selected (to be filtered/merged)
        mask = torch.ones(N_merged, dtype=torch.bool, device=image_tokens.device)
        mask.scatter_(0, all_indices, False)

        # Gather dominant tokens
        dominant_tokens = image_tokens[~mask]  # [dominant_num, D]

        # Step 5: Pool metric_k to post-merger resolution for similarity computation
        metric_pooled = metric_k.view(-1, 4, metric_k.shape[-1]).mean(dim=1)  # [N_merged, K_dim]

        # Get filtered (non-dominant) tokens and their metrics
        filtered_indices = torch.where(mask)[0]
        metric_filtered = metric_pooled[mask]  # [N_filtered, K_dim]
        hidden_filtered = image_tokens[mask]   # [N_filtered, D]

        # Normalize metric for cosine similarity
        metric_normalized = F.normalize(metric_filtered, p=2, dim=-1)

        # Step 6: Select contextual targets by uniform sampling from filtered tokens
        n_filtered = metric_normalized.shape[0]
        if n_filtered == 0 or contextual_num == 0:
            # All tokens are dominant, no contextual merging needed
            keep_indices = all_indices
            return image_tokens[keep_indices], keep_indices

        step = max(1, n_filtered // contextual_num)
        target_indices = torch.arange(0, n_filtered, step, device=image_tokens.device)[:contextual_num]
        contextual_indices = filtered_indices[target_indices]  # indices into image_tokens

        target_tokens_metric = metric_normalized[target_indices]  # [contextual_num, K_dim]

        # Step 7: Merge remaining tokens to nearest contextual target
        # Create mask for non-target filtered tokens
        non_target_mask = torch.ones(n_filtered, dtype=torch.bool, device=image_tokens.device)
        non_target_mask[target_indices] = False

        tokens_to_merge_metric = metric_normalized[non_target_mask]  # [n_remaining, K_dim]
        hidden_to_merge = hidden_filtered[non_target_mask]           # [n_remaining, D]

        if tokens_to_merge_metric.shape[0] > 0 and contextual_num > 0:
            # Cosine similarity: [n_remaining, contextual_num]
            similarity = torch.matmul(tokens_to_merge_metric.float(), target_tokens_metric.float().T)
            assign_one_hot = torch.zeros(
                tokens_to_merge_metric.shape[0], contextual_num,
                dtype=image_tokens.dtype, device=image_tokens.device
            )
            assign_one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)

            counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
            aggregated = (
                torch.matmul(assign_one_hot.T.float(), hidden_to_merge.float()) / counts.float()
            ).to(image_tokens.dtype)

            target_hidden = hidden_filtered[target_indices]
            contextual_tokens = target_hidden + aggregated
        else:
            contextual_tokens = hidden_filtered[target_indices]

        # Step 8: Combine and sort by original index
        all_keep_indices = torch.cat([all_indices, contextual_indices])
        all_keep_indices, sort_order = all_keep_indices.sort()

        # Build output tensor preserving order
        combined = torch.cat([dominant_tokens, contextual_tokens], dim=0)
        # Map: first dominant_num entries → dominant, rest → contextual
        # After sorting all_keep_indices, we need to place tokens correctly
        output = torch.zeros(len(all_keep_indices), D, dtype=image_tokens.dtype, device=image_tokens.device)

        # Determine which positions in sorted output are dominant vs contextual
        dominant_mask = torch.isin(all_keep_indices, all_indices)
        dominant_positions = torch.where(dominant_mask)[0]
        contextual_positions = torch.where(~dominant_mask)[0]

        output[dominant_positions] = dominant_tokens
        output[contextual_positions] = contextual_tokens

        return output, all_keep_indices


# =============================================================================
# PruMerge+ Compressor
# =============================================================================

def complement_idx(idx: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Get complement indices: all indices in [0, dim) that are NOT in idx.

    Reference: siglip_model.py L574-587

    Args:
        idx: [B, K] selected indices
        dim: total number of elements
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    n_idx = idx.shape[-1]
    dims = idx.shape[:-1] + (-1,)
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


def outlier_detection_iqr(attn: torch.Tensor) -> float:
    """
    IQR-based outlier detection to determine reduction ratio.

    Reference: siglip_model.py L595-615

    Args:
        attn: [B, N] attention scores per token
    Returns:
        ratio of tokens that are outliers (above Q3 + 1.5 * IQR)
    """
    ratios = []
    for i in range(attn.shape[0]):
        cur_attn = attn[i].to(dtype=torch.float32).cpu().numpy().flatten()
        Q1 = np.percentile(cur_attn, 25)
        Q3 = np.percentile(cur_attn, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = np.where(cur_attn > upper_bound)[0]
        ratio = len(outlier_indices) / len(cur_attn)
        ratios.append(ratio)
    return sum(ratios) / len(ratios)


class PruMergePlusCompressor:
    """
    PruMerge+: IQR outlier detection for token selection + attention-weighted
    cosine-similarity merge.

    Reference: qwen2vl_model.py L1562-1698

    Args:
        top_k_neighbors: number of most-similar tokens to aggregate (default 32)
    """

    def __init__(self, top_k_neighbors: int = 32):
        self.top_k_neighbors = top_k_neighbors

    @torch.no_grad()
    def __call__(
        self,
        image_tokens: torch.Tensor,   # [N_merged, D] — post-merger tokens
        attn_weights: torch.Tensor,    # [num_heads, N_pre, N_pre] — pre-merger attn
        metric_k: torch.Tensor,        # [N_pre, num_heads * head_dim] — pre-merger K
        budgets: float,                 # fraction to keep (e.g., 1/9)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            updated_tokens: [M, D] — compressed and merged tokens
            keep_indices: [M] — sorted indices into image_tokens
        """
        device = image_tokens.device
        N_merged = image_tokens.shape[0]
        D = image_tokens.shape[1]

        # Add batch dimension for compatibility with complement_idx
        image_features = image_tokens.unsqueeze(0)  # [1, N, D]
        B = 1

        # Step 1: Pool attention weights to post-merger resolution
        num_heads, N_pre, _ = attn_weights.shape
        N_pre_pooled = N_pre // 4
        attn_pooled = attn_weights.view(num_heads, N_pre_pooled, 4, N_pre_pooled, 4)
        attn_pooled = attn_pooled.mean(dim=(2, 4))  # [num_heads, N_pooled, N_pooled]

        # Step 2: Pool K to post-merger resolution
        desired_layer_k = metric_k.view(N_pre // 4, 4, -1).mean(dim=1).unsqueeze(0)  # [1, N_merged, K_dim]
        k_C = desired_layer_k.shape[-1]

        # Cls attention: average over all heads and query positions
        cls_attn = attn_pooled.mean(dim=[0, 1]).unsqueeze(0)  # [1, N_merged]

        # Step 3: IQR outlier detection
        reduction_ratio = outlier_detection_iqr(cls_attn)
        budgets_token = max(int(budgets * N_merged), 1)
        iqr_token = max(int(N_merged * reduction_ratio), 1)

        # Step 4: Select token indices
        if budgets_token > iqr_token:
            # IQR outliers + arithmetic progression sampling
            _, iqr_idx = torch.topk(cls_attn, iqr_token, dim=1, largest=True)  # [1, iqr_token]

            remaining_tokens = budgets_token - iqr_token
            step_length = max(1, int(N_merged / budgets_token))
            arithmetic_sequence = torch.arange(0, N_merged, step_length, device=device)

            # Filter out already-selected IQR indices
            iqr_set = iqr_idx[0].flatten()
            filtered_mask = ~torch.isin(arithmetic_sequence, iqr_set)
            filtered_sequence = arithmetic_sequence[filtered_mask]

            if len(filtered_sequence) > remaining_tokens:
                filtered_sequence = filtered_sequence[:remaining_tokens]
            elif len(filtered_sequence) < remaining_tokens:
                # Rare case: fill from remaining available indices
                all_used = torch.cat([iqr_set, filtered_sequence])
                available = torch.tensor(
                    [x for x in range(N_merged) if x not in all_used.tolist()],
                    device=device
                )
                if len(available) > 0:
                    extra = available[:remaining_tokens - len(filtered_sequence)]
                    filtered_sequence = torch.cat([filtered_sequence, extra])

            idx = torch.cat([iqr_idx[0], filtered_sequence])[:budgets_token].unsqueeze(0)  # [1, budgets_token]
        else:
            _, idx = torch.topk(cls_attn, budgets_token, dim=1, largest=True)  # [1, budgets_token]

        # Step 5: Gather selected features and their complements
        C = D
        index_features = idx.unsqueeze(-1).expand(-1, -1, C)  # [1, M, D]
        index_k = idx.unsqueeze(-1).expand(-1, -1, k_C)       # [1, M, K_dim]

        x_others = torch.gather(image_features, dim=1, index=index_features)     # [1, M, D]
        Key_others = torch.gather(desired_layer_k, dim=1, index=index_k)         # [1, M, K_dim]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)                 # [1, M]

        compl = complement_idx(idx, N_merged)                                     # [1, N-M]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_Key = torch.gather(desired_layer_k, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, k_C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

        # Step 6: Normalize keys for cosine similarity
        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        _, left_tokens, _ = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        # Step 7: For each selected token, find top-k similar and merge
        for b in range(B):
            for i in range(left_tokens):
                key_norm_i = Key_others_norm[b, i, :].unsqueeze(0).unsqueeze(0)  # [1, 1, K_dim]

                # Gather all other tokens (other selected + non-selected)
                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)
                after_i_Key = Key_others_norm[b, i + 1:, :].unsqueeze(0)
                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b, :, :].unsqueeze(0)], dim=1)

                before_i_x = x_others[b, :i, :].unsqueeze(0)
                after_i_x = x_others[b, i + 1:, :].unsqueeze(0)
                rest_x = torch.cat([before_i_x, after_i_x, non_topk[b, :, :].unsqueeze(0)], dim=1)

                before_i_attn = x_others_attn[b, :i].unsqueeze(0)
                after_i_attn = x_others_attn[b, i + 1:].unsqueeze(0)
                rest_attn = torch.cat([before_i_attn, after_i_attn, non_topk_attn[b, :].unsqueeze(0)], dim=1)

                # Cosine similarity
                cos_sim = torch.bmm(key_norm_i, rest_Keys.transpose(1, 2))  # [1, 1, n_rest]

                cos_sim_num = max(min(self.top_k_neighbors, cos_sim.shape[2]), 1)
                _, cluster_indices = torch.topk(cos_sim, k=cos_sim_num, dim=2, largest=True)

                cluster_tokens = rest_x[:, cluster_indices.squeeze(), :]   # [1, k, D]
                weights = rest_attn[:, cluster_indices.squeeze()].unsqueeze(-1)  # [1, k, 1]

                weighted_avg = torch.sum(cluster_tokens * weights, dim=1)  # [1, D]
                updated_x_others[b, i, :] = x_others[b, i, :] + weighted_avg.squeeze(0)

        # Sort indices and reorder tokens to match
        sorted_vals, sort_order = idx.squeeze(0).sort()
        keep_indices = sorted_vals  # [M]
        output_tokens = updated_x_others.squeeze(0)[sort_order].to(dtype=image_tokens.dtype)

        return output_tokens, keep_indices
