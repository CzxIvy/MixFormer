"""
MixFormer: Unified Transformer-style architecture for recommender systems.

Jointly models sequential behaviors and feature interactions within a single
backbone through L stacked MixFormer blocks. (Section 3.1-3.3)

Architecture:
  Input Layer → [MixFormer Block × L] → Task Networks

Input Layer:
  - Sequential features → embed & project to N*D dim
  - Non-sequential features → embed, split into N heads, project each to D dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm
from .mixformer_block import MixFormerBlock


class SequentialEmbedding(nn.Module):
    """Embedding layer for sequential features (Section 3.2).

    Embeds each feature of every action in the behavior sequence, concatenates
    them, and projects to the model dimension N*D.

    Args:
        feature_specs: List of (vocab_size, embed_dim) for each sequential feature.
        total_dim: Target dimension N*D for each action representation.
    """

    def __init__(self, feature_specs: list, total_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size, embed_dim in feature_specs
        ])
        concat_dim = sum(dim for _, dim in feature_specs)
        # Project to total_dim (N*D) to align with model dimension
        if concat_dim != total_dim:
            self.proj = nn.Linear(concat_dim, total_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, seq_features: list) -> torch.Tensor:
        """
        Args:
            seq_features: List of (B, T) tensors, one per sequential feature.
        Returns:
            S: (B, T, total_dim) sequence representation.
        """
        embeds = [emb(feat) for emb, feat in zip(self.embeddings, seq_features)]
        s = torch.cat(embeds, dim=-1)  # (B, T, concat_dim)
        return self.proj(s)


class NonSequentialEmbedding(nn.Module):
    """Embedding and splitting layer for non-sequential features (Section 3.2).

    Embeds all non-sequential features, concatenates into e_ns ∈ R^{D_ns},
    splits into N contiguous subvectors of dimension d = D_ns/N, and projects
    each subvector to D dimensions via a learnable linear matrix (Eq. 2).

    Args:
        feature_specs: List of (vocab_size, embed_dim) for each feature.
        num_heads: Number of heads N.
        head_dim: Dimension per head D.
    """

    def __init__(self, feature_specs: list, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for vocab_size, embed_dim in feature_specs
        ])

        D_ns = sum(dim for _, dim in feature_specs)
        # Pad D_ns to be divisible by N if necessary
        self.pad_size = (num_heads - D_ns % num_heads) % num_heads
        D_ns_padded = D_ns + self.pad_size
        self.split_dim = D_ns_padded // num_heads  # d = D_ns / N

        # Per-head projection W_j ∈ R^{D×d} (Eq. 2)
        self.head_projs = nn.ModuleList([
            nn.Linear(self.split_dim, head_dim, bias=False)
            for _ in range(num_heads)
        ])

    def forward(self, non_seq_features: list) -> torch.Tensor:
        """
        Args:
            non_seq_features: List of (B,) tensors, one per feature.
        Returns:
            X: (B, N, D) feature head representations.
        """
        embeds = [emb(feat) for emb, feat in zip(self.embeddings, non_seq_features)]
        e_ns = torch.cat(embeds, dim=-1)  # (B, D_ns)

        # Pad if needed
        if self.pad_size > 0:
            e_ns = F.pad(e_ns, (0, self.pad_size))

        # Split into N subvectors and project each (Eq. 2)
        heads = []
        for j in range(self.num_heads):
            start = j * self.split_dim
            end = start + self.split_dim
            heads.append(self.head_projs[j](e_ns[:, start:end]))

        return torch.stack(heads, dim=1)  # (B, N, D)


class TaskNetwork(nn.Module):
    """Task-specific prediction head for multi-task CTR prediction.

    A simple MLP that maps the fused representation to a scalar logit.

    Args:
        input_dim: Input dimension (N * D).
        hidden_dims: Tuple of hidden layer dimensions.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h_d
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) fused representation.
        Returns:
            (B,) logits for binary classification.
        """
        return self.net(x).squeeze(-1)


class MixFormer(nn.Module):
    """MixFormer: unified Transformer-style Large Recommender Model.

    Jointly models sequential behaviors and feature interactions within
    a single backbone through shared parameterization (Section 3).

    Architecture:
        1. Feature Embedding & Splitting (Section 3.2)
        2. L stacked MixFormer Blocks (Section 3.3)
        3. Task-specific Networks

    Args:
        config: MixFormerConfig instance with model hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Feature embedding layers (Section 3.2)
        self.seq_embedding = SequentialEmbedding(
            config.seq_feature_specs, config.total_dim
        )
        self.non_seq_embedding = NonSequentialEmbedding(
            config.non_seq_feature_specs, config.num_heads, config.head_dim
        )

        # L MixFormer blocks (Section 3.3)
        self.blocks = nn.ModuleList([
            MixFormerBlock(
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                seq_ffn_hidden_dim=config.seq_ffn_hidden_dim,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final normalization before task networks
        self.final_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

        # Multi-task prediction heads
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
            seq_mask: (B, T) boolean mask (True=valid, False=padding).
        Returns:
            logits: List of (B,) tensors, one per task (raw logits).
        """
        # Embed features (Section 3.2)
        x = self.non_seq_embedding(non_seq_features)  # (B, N, D)
        seq = self.seq_embedding(seq_features)  # (B, T, N*D)

        # Pass through L MixFormer blocks
        # Note: seq is the original sequence input reused at each layer,
        # each block has its own per-layer FFN for sequence transformation
        for block in self.blocks:
            x = block(x, seq, seq_mask)

        # Final normalization and flatten
        x = self.final_norm(x)  # (B, N, D)
        x_flat = x.reshape(x.size(0), -1)  # (B, N*D)

        # Task-specific predictions
        logits = [task_net(x_flat) for task_net in self.task_networks]
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
