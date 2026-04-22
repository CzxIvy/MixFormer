"""
UI-MixFormer: User-Item Decoupled MixFormer (Section 3.4).

Introduces user-item decoupling strategy for Request Level Batching (RLB)
optimization. User-side computations are shared across multiple candidate
items within a single request, reducing redundant computation.

Key modifications:
  1. Feature Decoupling: non-sequential features split into user/item groups
  2. Masked HeadMixing: unidirectional user-to-item fusion via mask matrix
  3. Separate user/item computation paths for RLB compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLUFFN, PerHeadSwiGLUFFN, head_mixing
from .mixformer_block import CrossAttention, OutputFusion
from .mixformer import SequentialEmbedding, TaskNetwork


def create_ui_mask(num_heads: int, head_dim: int, num_user_heads: int) -> torch.Tensor:
    """Create mask matrix M for user-item decoupled HeadMixing (Eq. 10).

    The mask ensures that user-side heads do not contain item-side information
    after HeadMixing, enabling request-level sharing of user-side computations.

    M[i, j] = 0  if i < N_U and j >= N_U * D/N
    M[i, j] = 1  otherwise

    Args:
        num_heads: Total number of heads (N).
        head_dim: Dimension per head (D).
        num_user_heads: Number of user-side heads (N_U).
    Returns:
        M: (N, D) mask tensor.
    """
    mask = torch.ones(num_heads, head_dim)
    sub_dim = head_dim // num_heads  # D/N
    # Zero out item-side information in user-side heads
    for i in range(num_user_heads):
        mask[i, num_user_heads * sub_dim:] = 0.0
    return mask


class UIQueryMixer(nn.Module):
    """Query Mixer with User-Item Decoupled HeadMixing (Section 3.4.2).

    Uses a mask matrix to enforce unidirectional user-to-item information
    flow in HeadMixing, enabling user-side computation to be reused
    across multiple candidate items.

    HeadMixing_decouple(·) = M ⊙ HeadMixing(·)  (Eq. 11)

    Args:
        num_heads: Total number of heads (N).
        head_dim: Dimension per head (D).
        ffn_hidden_dim: Hidden dim for per-head SwiGLU FFN.
        num_user_heads: Number of user-side heads (N_U).
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        num_user_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_user_heads = num_user_heads

        self.norm1 = RMSNorm(head_dim, eps=norm_eps)
        self.norm2 = RMSNorm(head_dim, eps=norm_eps)
        self.per_head_ffn = PerHeadSwiGLUFFN(num_heads, head_dim, ffn_hidden_dim)

        # Register mask as buffer (non-trainable, moves with model to device)
        mask = create_ui_mask(num_heads, head_dim, num_user_heads)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) feature heads (user + item).
        Returns:
            q: (B, N, D) query representations with decoupled HeadMixing.
        """
        # Eq. 11: Masked HeadMixing with pre-norm and residual
        mixed = head_mixing(self.norm1(x))
        mixed = mixed * self.mask.unsqueeze(0)  # Apply UI decoupling mask
        p = mixed + x

        # Per-head FFN with pre-norm and residual (Eq. 4)
        q = self.per_head_ffn(self.norm2(p)) + p
        return q


class UIMixFormerBlock(nn.Module):
    """User-Item Decoupled MixFormer Block (Section 3.4).

    Same structure as MixFormerBlock but uses UIQueryMixer with masked
    HeadMixing for user-item decoupling.

    Args:
        num_heads: Total number of heads (N).
        head_dim: Dimension per head (D).
        ffn_hidden_dim: Hidden dim for per-head FFNs.
        seq_ffn_hidden_dim: Hidden dim for sequence FFN in Cross Attention.
        num_user_heads: Number of user-side heads (N_U).
        norm_eps: Epsilon for RMSNorm.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        seq_ffn_hidden_dim: int,
        num_user_heads: int,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        total_dim = num_heads * head_dim

        self.query_mixer = UIQueryMixer(
            num_heads, head_dim, ffn_hidden_dim, num_user_heads, norm_eps
        )
        self.cross_attention = CrossAttention(
            num_heads, head_dim, total_dim, seq_ffn_hidden_dim, norm_eps, dropout
        )
        self.output_fusion = OutputFusion(
            num_heads, head_dim, ffn_hidden_dim, norm_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        seq: torch.Tensor,
        seq_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) feature heads (user + item).
            seq: (B, T, N*D) behavior sequence.
            seq_mask: (B, T) boolean mask.
        Returns:
            o: (B, N, D) output features.
        """
        q = self.query_mixer(x)
        z = self.cross_attention(q, seq, seq_mask)
        o = self.output_fusion(z)
        return o


class UIFeatureEmbedding(nn.Module):
    """Feature Embedding with User-Item Decoupling (Section 3.4.1).

    Partitions non-sequential features into disjoint user-side and item-side
    subsets, projecting them into N_U and N_G heads respectively.
    Total head number N = N_U + N_G is preserved.

    Args:
        non_seq_feature_specs: List of (vocab_size, embed_dim) for all non-seq features.
        user_feature_indices: Indices of user-side features in the specs list.
        item_feature_indices: Indices of item-side features in the specs list.
        num_heads: Total number of heads (N).
        head_dim: Dimension per head (D).
    """

    def __init__(
        self,
        non_seq_feature_specs: list,
        user_feature_indices: list,
        item_feature_indices: list,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        # All embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size, embed_dim in non_seq_feature_specs
        ])

        self.user_indices = user_feature_indices
        self.item_indices = item_feature_indices

        # Compute user/item embedding dimensions
        D_user = sum(non_seq_feature_specs[i][1] for i in user_feature_indices)
        D_item = sum(non_seq_feature_specs[i][1] for i in item_feature_indices)
        D_ns = D_user + D_item

        # Compute N_U and N_G (Section 3.4.1)
        # N_G = floor(D_item * N / D_ns), N_U = N - N_G
        self.num_item_heads = max(1, round(D_item * num_heads / D_ns))
        self.num_user_heads = num_heads - self.num_item_heads

        # Pad user/item dims to be divisible by their head counts
        user_pad = (self.num_user_heads - D_user % self.num_user_heads) % self.num_user_heads
        item_pad = (self.num_item_heads - D_item % self.num_item_heads) % self.num_item_heads
        self.user_pad = user_pad
        self.item_pad = item_pad
        self.user_split_dim = (D_user + user_pad) // self.num_user_heads
        self.item_split_dim = (D_item + item_pad) // self.num_item_heads

        # Per-head projections for user and item heads
        self.user_head_projs = nn.ModuleList([
            nn.Linear(self.user_split_dim, head_dim, bias=False)
            for _ in range(self.num_user_heads)
        ])
        self.item_head_projs = nn.ModuleList([
            nn.Linear(self.item_split_dim, head_dim, bias=False)
            for _ in range(self.num_item_heads)
        ])

    def forward(self, non_seq_features: list) -> torch.Tensor:
        """
        Args:
            non_seq_features: List of (B,) tensors, one per feature.
        Returns:
            X: (B, N, D) with user heads first, then item heads.
        """
        # Embed user and item features separately
        user_embeds = [self.embeddings[i](non_seq_features[i]) for i in self.user_indices]
        item_embeds = [self.embeddings[i](non_seq_features[i]) for i in self.item_indices]

        e_user = torch.cat(user_embeds, dim=-1)  # (B, D_user)
        e_item = torch.cat(item_embeds, dim=-1)  # (B, D_item)

        # Pad if needed
        if self.user_pad > 0:
            e_user = F.pad(e_user, (0, self.user_pad))
        if self.item_pad > 0:
            e_item = F.pad(e_item, (0, self.item_pad))

        # Split and project user heads
        heads = []
        for j in range(self.num_user_heads):
            start = j * self.user_split_dim
            end = start + self.user_split_dim
            heads.append(self.user_head_projs[j](e_user[:, start:end]))

        # Split and project item heads
        for j in range(self.num_item_heads):
            start = j * self.item_split_dim
            end = start + self.item_split_dim
            heads.append(self.item_head_projs[j](e_item[:, start:end]))

        return torch.stack(heads, dim=1)  # (B, N, D)


class UIMixFormer(nn.Module):
    """User-Item Decoupled MixFormer (UI-MixFormer, Section 3.4).

    Variant of MixFormer with user-item decoupling strategy that enables
    Request Level Batching (RLB) for efficient inference. User-side
    computations can be shared across multiple candidate items.

    Key differences from MixFormer:
      1. Feature embedding splits into user/item groups
      2. HeadMixing uses mask for unidirectional user→item fusion
      3. User-side heads are reusable across candidates

    Args:
        config: MixFormerConfig with user/item feature indices specified.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.user_feature_indices is not None, (
            "user_feature_indices must be specified for UI-MixFormer"
        )
        assert config.item_feature_indices is not None, (
            "item_feature_indices must be specified for UI-MixFormer"
        )

        # User-Item decoupled feature embedding (Section 3.4.1)
        self.feature_embedding = UIFeatureEmbedding(
            config.non_seq_feature_specs,
            config.user_feature_indices,
            config.item_feature_indices,
            config.num_heads,
            config.head_dim,
        )
        num_user_heads = self.feature_embedding.num_user_heads

        # Sequential embedding (shared, user-side)
        self.seq_embedding = SequentialEmbedding(
            config.seq_feature_specs, config.total_dim
        )

        # UI-MixFormer blocks with masked HeadMixing
        self.blocks = nn.ModuleList([
            UIMixFormerBlock(
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                seq_ffn_hidden_dim=config.seq_ffn_hidden_dim,
                num_user_heads=num_user_heads,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

        # Task networks
        self.task_networks = nn.ModuleList([
            TaskNetwork(config.total_dim, config.task_hidden_dims, config.dropout)
            for _ in range(config.num_tasks)
        ])

    def forward(
        self,
        non_seq_features: list,
        seq_features: list,
        seq_mask: torch.Tensor = None,
    ) -> list:
        """
        Args:
            non_seq_features: List of (B,) tensors for non-sequential features.
            seq_features: List of (B, T) tensors for sequential features.
            seq_mask: (B, T) boolean mask.
        Returns:
            logits: List of (B,) tensors, one per task.
        """
        # Embed features with user-item decoupling
        x = self.feature_embedding(non_seq_features)  # (B, N, D)
        seq = self.seq_embedding(seq_features)  # (B, T, N*D)

        # Pass through UI-MixFormer blocks
        for block in self.blocks:
            x = block(x, seq, seq_mask)

        # Final normalization and flatten
        x = self.final_norm(x)
        x_flat = x.reshape(x.size(0), -1)  # (B, N*D)

        # Task predictions
        logits = [task_net(x_flat) for task_net in self.task_networks]
        return logits

    def forward_user_side(
        self,
        non_seq_features: list,
        seq_features: list,
        seq_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute user-side representations (for RLB optimization).

        In production, this is computed once per request and shared across
        all candidate items, significantly reducing redundant computation.

        Args:
            non_seq_features: List of (B,) tensors for ALL features.
            seq_features: List of (B, T) tensors.
            seq_mask: (B, T) boolean mask.
        Returns:
            user_repr: (B, N_U, D) user-side head representations.
        """
        x = self.feature_embedding(non_seq_features)
        seq = self.seq_embedding(seq_features)

        for block in self.blocks:
            x = block(x, seq, seq_mask)

        x = self.final_norm(x)
        num_user_heads = self.feature_embedding.num_user_heads
        return x[:, :num_user_heads, :]  # (B, N_U, D)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
