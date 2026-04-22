"""MixFormer typed configuration model.

Provides a dataclass-based configuration object used across model building,
YAML config loading, and manual Python instantiation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


FeatureSpec = Tuple[int, int]


@dataclass(slots=True)
class MixFormerConfig:
    """Structured model configuration for MixFormer and UI-MixFormer.

    Key hyperparameters from the paper:
      - MixFormer-small:  N=16, L=4, D=384  (paper reports D=386)
      - MixFormer-medium: N=16, L=4, D=768
    """

    num_heads: int = 16
    num_layers: int = 4
    head_dim: int = 384
    ffn_hidden_dim: Optional[int] = None
    seq_ffn_hidden_dim: Optional[int] = None
    max_seq_len: int = 512
    num_tasks: int = 2
    task_hidden_dims: Tuple[int, ...] = (256, 128)
    norm_eps: float = 1e-6
    dropout: float = 0.0
    non_seq_feature_specs: List[FeatureSpec] = field(default_factory=list)
    seq_feature_specs: List[FeatureSpec] = field(default_factory=list)
    user_feature_indices: Optional[List[int]] = None
    item_feature_indices: Optional[List[int]] = None

    def __post_init__(self):
        self.task_hidden_dims = tuple(self.task_hidden_dims)
        self.non_seq_feature_specs = list(self.non_seq_feature_specs)
        self.seq_feature_specs = list(self.seq_feature_specs)

        if self.user_feature_indices is not None:
            self.user_feature_indices = list(self.user_feature_indices)
        if self.item_feature_indices is not None:
            self.item_feature_indices = list(self.item_feature_indices)

        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = self.head_dim * 4
        if self.seq_ffn_hidden_dim is None:
            self.seq_ffn_hidden_dim = self.head_dim * 4

        self._validate()

    def _validate(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.head_dim % self.num_heads != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be divisible by num_heads "
                f"({self.num_heads}) for HeadMixing operation"
            )
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")
        if self.seq_ffn_hidden_dim <= 0:
            raise ValueError("seq_ffn_hidden_dim must be positive")
        if any(len(spec) != 2 for spec in self.non_seq_feature_specs):
            raise ValueError("non_seq_feature_specs must be a list of (vocab_size, embed_dim)")
        if any(len(spec) != 2 for spec in self.seq_feature_specs):
            raise ValueError("seq_feature_specs must be a list of (vocab_size, embed_dim)")

    @property
    def total_dim(self) -> int:
        """Total model dimension: N * D."""
        return self.num_heads * self.head_dim

    @property
    def non_seq_embed_dim(self) -> int:
        """Total embedding dimension for non-sequential features."""
        return sum(dim for _, dim in self.non_seq_feature_specs)

    @property
    def seq_embed_dim(self) -> int:
        """Total embedding dimension for sequential features."""
        return sum(dim for _, dim in self.seq_feature_specs)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Construct config from a plain dictionary.

        This is useful when converting parsed YAML or other serialized config
        data into the typed runtime configuration object.
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Export config as a plain dictionary."""
        return {
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "head_dim": self.head_dim,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "seq_ffn_hidden_dim": self.seq_ffn_hidden_dim,
            "max_seq_len": self.max_seq_len,
            "num_tasks": self.num_tasks,
            "task_hidden_dims": list(self.task_hidden_dims),
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
            "non_seq_feature_specs": list(self.non_seq_feature_specs),
            "seq_feature_specs": list(self.seq_feature_specs),
            "user_feature_indices": self.user_feature_indices,
            "item_feature_indices": self.item_feature_indices,
        }

    @classmethod
    def with_preset(cls, preset: str, **kwargs):
        """Create config from a named preset plus overrides."""
        presets = {
            "small": {"num_heads": 16, "num_layers": 4, "head_dim": 384},
            "medium": {"num_heads": 16, "num_layers": 4, "head_dim": 768},
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from small/medium.")
        defaults = presets[preset].copy()
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def small(cls, **kwargs):
        """MixFormer-small preset: N=16, L=4, D=384."""
        return cls.with_preset("small", **kwargs)

    @classmethod
    def medium(cls, **kwargs):
        """MixFormer-medium preset: N=16, L=4, D=768."""
        return cls.with_preset("medium", **kwargs)

    def __repr__(self):
        return (
            f"MixFormerConfig(N={self.num_heads}, L={self.num_layers}, "
            f"D={self.head_dim}, total_dim={self.total_dim}, "
            f"max_seq_len={self.max_seq_len}, num_tasks={self.num_tasks})"
        )
