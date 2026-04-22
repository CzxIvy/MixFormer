"""YAML configuration loader for MixFormer.

Loads YAML config files with inheritance: user config overrides default.yaml.
Parses feature definitions into the typed runtime config expected by models.
"""

import copy
from pathlib import Path

import yaml

from config import MixFormerConfig


CONFIGS_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "base" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists are replaced, not appended."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str) -> dict:
    """Load a single YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str = None) -> dict:
    """Load config with configs/base/default.yaml as base, then override it.

    Args:
        config_path: Path to user YAML config. If None, only default is loaded.
    Returns:
        Merged config dict.
    """
    base = load_yaml(DEFAULT_CONFIG_PATH)
    if config_path is not None:
        override = load_yaml(config_path)
        return _deep_merge(base, override)
    return base


def parse_feature_specs(cfg: dict):
    """Parse feature definitions from config dict.

    Converts the human-readable YAML feature list into the tuple format
    expected by MixFormerConfig:
      non_seq_specs: [(vocab_size, embed_dim), ...]
      seq_specs:     [(vocab_size, embed_dim), ...]
      user_indices:  [int, ...]
      item_indices:  [int, ...]

    Args:
        cfg: Full merged config dict.
    Returns:
        (non_seq_specs, seq_specs, user_indices, item_indices, feature_names)
    """
    feat_cfg = cfg.get("features", {})

    # Non-sequential features
    non_seq_list = feat_cfg.get("non_sequential", [])
    non_seq_specs = [(f["vocab_size"], f["embed_dim"]) for f in non_seq_list]

    # Sequential features
    seq_list = feat_cfg.get("sequential", [])
    seq_specs = [(f["vocab_size"], f["embed_dim"]) for f in seq_list]

    # User/Item group split for UI-MixFormer
    user_groups = set(feat_cfg.get("user_groups", ["user"]))
    item_groups = set(feat_cfg.get("item_groups", ["item", "context"]))

    user_indices = []
    item_indices = []
    for i, f in enumerate(non_seq_list):
        group = f.get("group", "item")
        if group in user_groups:
            user_indices.append(i)
        elif group in item_groups:
            item_indices.append(i)
        else:
            # Unknown group defaults to item side
            item_indices.append(i)

    # Feature names for logging
    non_seq_names = [f.get("name", f"non_seq_{i}") for i, f in enumerate(non_seq_list)]
    seq_names = [f.get("name", f"seq_{i}") for i, f in enumerate(seq_list)]

    return non_seq_specs, seq_specs, user_indices, item_indices, non_seq_names, seq_names


def build_model_config(cfg: dict) -> MixFormerConfig:
    """Build MixFormerConfig from parsed YAML config.

    Handles model size presets (small/medium) and custom configurations.

    Args:
        cfg: Full merged config dict.
    Returns:
        MixFormerConfig instance.
    """
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    non_seq_specs, seq_specs, user_indices, item_indices, _, _ = parse_feature_specs(cfg)

    # Common kwargs
    kwargs = dict(
        max_seq_len=data_cfg.get("max_seq_len", 64),
        num_tasks=data_cfg.get("num_tasks", 2),
        task_hidden_dims=tuple(model_cfg.get("task_hidden_dims", [256, 128])),
        norm_eps=model_cfg.get("norm_eps", 1e-6),
        dropout=model_cfg.get("dropout", 0.0),
        non_seq_feature_specs=non_seq_specs,
        seq_feature_specs=seq_specs,
        user_feature_indices=user_indices,
        item_feature_indices=item_indices,
    )

    # FFN hidden dims (null → auto)
    if model_cfg.get("ffn_hidden_dim") is not None:
        kwargs["ffn_hidden_dim"] = model_cfg["ffn_hidden_dim"]
    if model_cfg.get("seq_ffn_hidden_dim") is not None:
        kwargs["seq_ffn_hidden_dim"] = model_cfg["seq_ffn_hidden_dim"]

    # Model size
    size = model_cfg.get("size", "small")
    if size == "small":
        config = MixFormerConfig.small(**kwargs)
    elif size == "medium":
        config = MixFormerConfig.medium(**kwargs)
    elif size == "custom":
        kwargs["num_heads"] = model_cfg.get("num_heads", 16)
        kwargs["num_layers"] = model_cfg.get("num_layers", 4)
        kwargs["head_dim"] = model_cfg.get("head_dim", 384)
        config = MixFormerConfig.from_dict(kwargs)
    else:
        raise ValueError(f"Unknown model size: {size}. Choose from small/medium/custom.")

    return config


def print_config_summary(cfg: dict, config: MixFormerConfig):
    """Print a human-readable summary of the loaded configuration."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    wandb_cfg = cfg.get("wandb", {})
    sweep_cfg = cfg.get("sweep", {})
    _, _, _, _, non_seq_names, seq_names = parse_feature_specs(cfg)

    wandb_enabled = bool(wandb_cfg.get("enabled", False)) and wandb_cfg.get("mode", "online") != "disabled"

    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"  Model type:      {model_cfg.get('type', 'mixformer')}")
    print(f"  Model size:      {model_cfg.get('size', 'small')}")
    print(f"  Data source:     {data_cfg.get('source', 'synthetic')}")
    print(f"  Architecture:    N={config.num_heads}, L={config.num_layers}, D={config.head_dim}")
    print(f"  Total dim:       {config.total_dim}")
    print(f"  Max seq len:     {config.max_seq_len}")
    print(f"  Num tasks:       {config.num_tasks}")
    print(f"  FFN hidden dim:  {config.ffn_hidden_dim}")
    print(f"  Non-seq features ({len(non_seq_names)}): {', '.join(non_seq_names)}")
    print(f"  Seq features ({len(seq_names)}):     {', '.join(seq_names)}")
    print(f"  Optimizer:       {train_cfg.get('optimizer', 'adamw')}")
    print(f"  LR:              {train_cfg.get('lr', 1e-3)}")
    print(f"  Batch size:      {train_cfg.get('batch_size', 256)}")
    print(f"  Epochs:          {train_cfg.get('num_epochs', 5)}")
    print(f"  Device:          {train_cfg.get('device', 'auto')}")
    if sweep_cfg.get("max_parameters") not in (None, "", 0):
        print(f"  Param budget:    {int(sweep_cfg.get('max_parameters')):,}")
    print(
        f"  W&B:             "
        f"{'enabled' if wandb_enabled else 'disabled'} ({wandb_cfg.get('mode', 'online')})"
    )
    if wandb_enabled:
        print(f"  W&B project:     {wandb_cfg.get('project', 'MixFormer')}")
        if wandb_cfg.get('name'):
            print(f"  W&B run name:    {wandb_cfg.get('name')}" )
        if wandb_cfg.get('log_model', False):
            print(
                f"  W&B artifact:    best checkpoint by "
                f"{wandb_cfg.get('best_model_metric', 'task_0/AUC')} "
                f"({wandb_cfg.get('best_model_mode', 'max')})"
            )
            if wandb_cfg.get('save_local_best_checkpoint', True):
                print(
                    f"  Local output:    "
                    f"{wandb_cfg.get('local_checkpoint_dir', 'outputs')}"
                )
    if data_cfg.get('source', 'synthetic') == 'local_file':
        print(f"  Train path:      {data_cfg.get('train_path', '')}")
        print(f"  Eval path:       {data_cfg.get('eval_path', '')}")
        print(f"  File format:     {data_cfg.get('file_format', 'jsonl')}")
    print("=" * 60)
