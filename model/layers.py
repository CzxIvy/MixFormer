"""
Core building blocks for MixFormer.

Implements:
  - RMSNorm: Root Mean Square Layer Normalization
  - SwiGLUFFN: SwiGLU-activated Feed-Forward Network
  - PerHeadSwiGLUFFN: Independent SwiGLU FFN per head (batched)
  - head_mixing: Parameter-free cross-head information exchange
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used as pre-normalization throughout MixFormer blocks.
    Normalizes over the last dimension.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x_float / rms
        return (x_normed * self.weight).to(dtype)


class SwiGLUFFN(nn.Module):
    """SwiGLU-activated Feed-Forward Network.

    FFN(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down

    Args:
        in_dim: Input and output dimension.
        hidden_dim: Hidden layer dimension. Defaults to 4 * in_dim.
    """

    def __init__(self, in_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim * 4
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class PerHeadSwiGLUFFN(nn.Module):
    """Per-head SwiGLU FFN with efficient batched implementation.

    Each of the N heads has its own independent SwiGLU FFN parameters,
    enabling head-specific non-linear transformations for heterogeneous
    features. Uses einsum for efficient batched computation.

    Args:
        num_heads: Number of heads (N).
        head_dim: Dimension per head (D).
        hidden_dim: Hidden layer dimension. Defaults to 4 * head_dim.
    """

    def __init__(self, num_heads: int, head_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = head_dim * 4
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        # Batched parameters: (N, D, H) for gate/up, (N, H, D) for down
        self.w_gate = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.w_up = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.w_down = nn.Parameter(torch.empty(num_heads, hidden_dim, head_dim))
        self._init_weights()

    def _init_weights(self):
        for p in [self.w_gate, self.w_up, self.w_down]:
            for i in range(self.num_heads):
                nn.init.kaiming_uniform_(p[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input tensor with N heads of dimension D.
        Returns:
            (B, N, D) transformed tensor, each head processed independently.
        """
        gate = torch.einsum('bnd,ndh->bnh', x, self.w_gate)
        up = torch.einsum('bnd,ndh->bnh', x, self.w_up)
        hidden = F.silu(gate) * up
        return torch.einsum('bnh,nhd->bnd', hidden, self.w_down)


def head_mixing(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free HeadMixing operation (Section 3.3.1, Figure 1).

    Enables efficient cross-head information exchange by:
      1. Reshaping (B, N, D) → (B, N, N, D/N)
      2. Transposing the two N dimensions
      3. Reshaping back to (B, N, D)

    This is equivalent to a structured permutation that mixes information
    across heads without any learnable parameters.

    Args:
        x: (B, N, D) tensor where N is number of heads, D is head dimension.
           D must be divisible by N.
    Returns:
        (B, N, D) tensor with mixed cross-head information.
    """
    B, N, D = x.shape
    assert D % N == 0, (
        f"head_dim ({D}) must be divisible by num_heads ({N}) for HeadMixing"
    )
    sub_dim = D // N
    x = x.reshape(B, N, N, sub_dim)
    x = x.transpose(1, 2).contiguous()
    x = x.reshape(B, N, D)
    return x
