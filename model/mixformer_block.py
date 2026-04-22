"""
MixFormer Block implementation (Section 3.3).

Each MixFormer block consists of three core modules:
  1. Query Mixer   — replaces self-attention with HeadMixing + per-head FFN
  2. Cross Attention — aggregates sequence conditioned on high-order queries
  3. Output Fusion  — deep integration via per-head FFN

These correspond to Self-Attention, Cross-Attention, and FFN in a standard
Transformer decoder block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLUFFN, PerHeadSwiGLUFFN, head_mixing


class QueryMixer(nn.Module):
    """Query Mixer module (Section 3.3.1).

    Replaces self-attention with parameter-free HeadMixing for efficient
    cross-head information exchange on heterogeneous feature heads, followed
    by per-head SwiGLU FFN for head-specific transformations.

    Forward pass (Eq. 3-4):
        P = HeadMixing(RMSNorm(X)) + X
        Q_i = SwiGLUFFN_i(RMSNorm(P_i)) + P_i

    Args:
        num_heads: Number of feature heads (N).
        head_dim: Dimension per head (D).
        ffn_hidden_dim: Hidden dimension for per-head SwiGLU FFN.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Pre-norm for HeadMixing
        self.norm1 = RMSNorm(head_dim, eps=norm_eps)
        # Pre-norm for per-head FFN
        self.norm2 = RMSNorm(head_dim, eps=norm_eps)
        # Per-head SwiGLU FFN
        self.per_head_ffn = PerHeadSwiGLUFFN(num_heads, head_dim, ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) non-sequential feature heads.
        Returns:
            q: (B, N, D) query representations.
        """
        # Eq. 3: HeadMixing with pre-norm and residual
        p = head_mixing(self.norm1(x)) + x
        # Eq. 4: Per-head FFN with pre-norm and residual
        q = self.per_head_ffn(self.norm2(p)) + p
        return q


class CrossAttention(nn.Module):
    """Cross Attention module (Section 3.3.2).

    Aggregates user behavior sequences conditioned on high-order feature
    representations from the Query Mixer. Each query head serves as a
    semantically specialized sub-query attending to the sequence.

    The sequence is transformed by a per-layer SwiGLU FFN (independently
    parameterized at each layer), then projected to keys and values for
    multi-head cross attention.

    Forward pass (Eq. 5-8):
        h_t = SwiGLUFFN^(l)(RMSNorm(s_t)) + s_t       (per-layer seq FFN)
        h_t^i = h_t[iD : (i+1)D]                       (split into heads)
        k_t^i = W_k^i h_t^i;  v_t^i = W_v^i h_t^i     (per-head K,V proj)
        z_i = Σ_t softmax(q_i^T k_t^i / √D) v_t^i + q_i  (attention + residual)

    Args:
        num_heads: Number of heads (N).
        head_dim: Dimension per head (D).
        total_dim: Total dimension N*D for sequence FFN.
        seq_ffn_hidden_dim: Hidden dimension for sequence SwiGLU FFN.
        norm_eps: Epsilon for RMSNorm.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        total_dim: int,
        seq_ffn_hidden_dim: int,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Per-layer SwiGLU FFN for sequence transformation (Eq. 5)
        self.seq_norm = RMSNorm(total_dim, eps=norm_eps)
        self.seq_ffn = SwiGLUFFN(total_dim, seq_ffn_hidden_dim)

        # Per-head K, V projection matrices (Eq. 7)
        # W_k^i, W_v^i ∈ R^{D×D} for each head i
        self.w_k = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        self.w_v = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        self._init_kv_weights()

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _init_kv_weights(self):
        for p in [self.w_k, self.w_v]:
            for i in range(self.num_heads):
                nn.init.xavier_uniform_(p[i])

    def forward(
        self,
        q: torch.Tensor,
        seq: torch.Tensor,
        seq_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            q: (B, N, D) query from Query Mixer.
            seq: (B, T, N*D) original behavior sequence embeddings.
            seq_mask: (B, T) boolean mask (True = valid, False = padding).
        Returns:
            z: (B, N, D) feature-conditioned sequential aggregation.
        """
        B, T, _ = seq.shape

        # Eq. 5: Transform sequence with per-layer FFN
        h = self.seq_ffn(self.seq_norm(seq)) + seq  # (B, T, N*D)

        # Eq. 6: Split into N heads
        h = h.view(B, T, self.num_heads, self.head_dim)  # (B, T, N, D)

        # Eq. 7: Project to keys and values (per-head)
        k = torch.einsum('btnd,nde->btne', h, self.w_k)  # (B, T, N, D)
        v = torch.einsum('btnd,nde->btne', h, self.w_v)  # (B, T, N, D)

        # Eq. 8: Scaled dot-product cross attention
        # q: (B, N, D), k: (B, T, N, D) → attn: (B, N, T)
        attn = torch.einsum('bnd,btnd->bnt', q, k) * self.scale

        # Apply sequence padding mask
        if seq_mask is not None:
            # seq_mask: (B, T) → (B, 1, T) for broadcasting with (B, N, T)
            attn = attn.masked_fill(~seq_mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)  # (B, N, T)
        attn = self.attn_dropout(attn)

        # Weighted aggregation of values
        z = torch.einsum('bnt,btnd->bnd', attn, v)  # (B, N, D)

        # Residual connection to query (Eq. 8)
        z = z + q

        return z


class OutputFusion(nn.Module):
    """Output Fusion module (Section 3.3.3).

    Performs deep integration of sequential and non-sequential signals
    using per-head SwiGLU FFN. Each head is processed independently to
    preserve head-level specialization and avoid representational
    interference across heterogeneous feature subspaces.

    Forward pass (Eq. 9):
        o_i = SwiGLUFFN_i(RMSNorm(z_i)) + z_i

    Args:
        num_heads: Number of heads (N).
        head_dim: Dimension per head (D).
        ffn_hidden_dim: Hidden dimension for per-head SwiGLU FFN.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = RMSNorm(head_dim, eps=norm_eps)
        self.per_head_ffn = PerHeadSwiGLUFFN(num_heads, head_dim, ffn_hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, D) cross-attention output.
        Returns:
            o: (B, N, D) fused output representation.
        """
        o = self.per_head_ffn(self.norm(z)) + z
        return o


class MixFormerBlock(nn.Module):
    """Single MixFormer block (Section 3.3).

    Integrates three core components that conceptually correspond to a
    standard Transformer decoder block:
        Query Mixer     ←→  Self-Attention
        Cross Attention ←→  Cross-Attention
        Output Fusion   ←→  Feed-Forward Network

    The unified parameterization enables sequential and non-sequential
    signals to mutually enhance each other within a single block.

    Args:
        num_heads: Number of feature heads (N).
        head_dim: Dimension per head (D).
        ffn_hidden_dim: Hidden dim for per-head FFNs (Query Mixer & Output Fusion).
        seq_ffn_hidden_dim: Hidden dim for sequence FFN in Cross Attention.
        norm_eps: Epsilon for RMSNorm.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        seq_ffn_hidden_dim: int,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        total_dim = num_heads * head_dim

        self.query_mixer = QueryMixer(
            num_heads, head_dim, ffn_hidden_dim, norm_eps
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
            x: (B, N, D) non-sequential feature heads.
            seq: (B, T, N*D) behavior sequence embeddings.
            seq_mask: (B, T) boolean mask for valid positions.
        Returns:
            o: (B, N, D) output features for next block or task networks.
        """
        q = self.query_mixer(x)
        z = self.cross_attention(q, seq, seq_mask)
        o = self.output_fusion(z)
        return o
