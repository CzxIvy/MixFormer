"""
Training and evaluation script for MixFormer.

Supports:
  - MixFormer (unified) and UI-MixFormer (user-item decoupled) variants
  - Multi-task CTR prediction with binary cross-entropy loss
  - AUC and UAUC evaluation metrics
  - YAML-based configuration with presets and custom overrides
    - Optional Weights & Biases experiment tracking
  - Synthetic data generation for demonstration

Usage:
    # 使用预设配置文件
    python train.py --config configs/experiments/debug.yaml
    python train.py --config configs/experiments/mixformer_small.yaml
    python train.py --config configs/experiments/ui_mixformer_medium.yaml

    # 使用配置文件 + 命令行覆盖（命令行参数优先级更高）
    python train.py --config configs/experiments/mixformer_small.yaml --batch_size 128 --lr 5e-4

    # 不使用配置文件，纯命令行（兼容旧用法）
    python train.py --model_size small --num_epochs 3 --batch_size 256
"""

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from config_loader import load_config, build_model_config, parse_feature_specs, print_config_summary
from model import MixFormer, UIMixFormer
from data import LocalFileRecDataset, SyntheticRecDataset, rec_collate_fn
from utils import compute_auc, compute_uauc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device, num_tasks, grad_clip=None):
    """Train model for one epoch.

    Returns:
        Average loss over all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        non_seq, seq, seq_mask, labels, _ = batch
        non_seq = [x.to(device) for x in non_seq]
        seq = [x.to(device) for x in seq]
        seq_mask = seq_mask.to(device)
        labels = [l.to(device) for l in labels]

        optimizer.zero_grad()
        logits = model(non_seq, seq, seq_mask)

        # Multi-task BCE loss (averaged across tasks)
        loss = sum(
            F.binary_cross_entropy_with_logits(logits[i], labels[i])
            for i in range(num_tasks)
        ) / num_tasks

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def is_wandb_enabled(wandb_cfg: dict) -> bool:
    """Check whether W&B tracking should be enabled for this run."""
    return bool(wandb_cfg.get("enabled", False)) and wandb_cfg.get("mode", "online") != "disabled"


def init_wandb_run(
    cfg: dict,
    resolved_model_config,
    model,
    device,
    num_params: int,
    train_dataset,
    eval_dataset,
):
    """Initialize a Weights & Biases run if enabled."""
    wandb_cfg = cfg.get("wandb", {})
    if not is_wandb_enabled(wandb_cfg):
        return None

    if wandb is None:
        raise ImportError(
            "wandb is enabled in config but the package is not installed. "
            "Install it with `pip install wandb` or disable wandb in the config."
        )

    run = wandb.init(
        project=wandb_cfg.get("project", "MixFormer"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        tags=wandb_cfg.get("tags") or None,
        notes=wandb_cfg.get("notes"),
        mode=wandb_cfg.get("mode", "online"),
        dir=wandb_cfg.get("dir"),
        config=cfg,
        reinit="finish_previous" if wandb_cfg.get("finish_previous", False) else None,
    )

    run.config.update(
        {
            "resolved_model_config": resolved_model_config.to_dict(),
            "runtime": {
                "device": str(device),
                "num_parameters": num_params,
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(eval_dataset),
            },
        },
        allow_val_change=True,
    )

    if wandb_cfg.get("watch_model", False):
        wandb.watch(
            model,
            log=wandb_cfg.get("watch_log", "gradients"),
            log_freq=wandb_cfg.get("watch_log_freq", 100),
        )

    return run


def init_wandb_skip_run(cfg: dict, message: str, skip_type: str = "invalid_model_shape"):
    """Create a lightweight W&B run to mark a sweep trial as skipped."""
    wandb_cfg = cfg.get("wandb", {})
    if not is_wandb_enabled(wandb_cfg) or wandb is None:
        print(message)
        return

    run = wandb.init(
        project=wandb_cfg.get("project", "MixFormer"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        tags=wandb_cfg.get("tags") or None,
        notes=wandb_cfg.get("notes"),
        mode=wandb_cfg.get("mode", "online"),
        dir=wandb_cfg.get("dir"),
        config=cfg,
        reinit="finish_previous" if wandb_cfg.get("finish_previous", False) else None,
    )
    run.summary["skipped"] = True
    run.summary["skip_reason"] = message
    run.summary["skip_type"] = skip_type
    run.finish()
    print(message)


def log_wandb_epoch(run, epoch, train_loss, train_time, eval_results, eval_time, optimizer):
    """Log per-epoch metrics to W&B."""
    if run is None:
        return

    payload = {
        "epoch": epoch,
        "train/loss": train_loss,
        "train/time_sec": train_time,
        "eval/time_sec": eval_time,
        "train/learning_rate": optimizer.param_groups[0]["lr"],
    }
    payload.update({f"eval/{key}": value for key, value in eval_results.items()})
    run.log(payload, step=epoch)


def get_wandb_model_selection_config(wandb_cfg: dict):
    """Return the metric name and optimization mode for best-model selection."""
    metric_name = wandb_cfg.get("best_model_metric", "task_0/AUC")
    metric_mode = wandb_cfg.get("best_model_mode", "max")
    if metric_mode not in {"max", "min"}:
        raise ValueError("wandb.best_model_mode must be 'max' or 'min'")
    return metric_name, metric_mode


def resolve_selection_metric(eval_results: dict, metric_name: str) -> float:
    """Resolve a scalar selection metric from evaluation results.

    Supported values:
      - task_0/AUC
      - eval/task_0/AUC
      - mean/AUC
      - mean/UAUC
    """
    normalized_name = metric_name[5:] if metric_name.startswith("eval/") else metric_name

    if normalized_name == "mean/AUC":
        values = [value for key, value in eval_results.items() if key.endswith("/AUC")]
    elif normalized_name == "mean/UAUC":
        values = [value for key, value in eval_results.items() if key.endswith("/UAUC")]
    else:
        if normalized_name not in eval_results:
            raise KeyError(
                f"Best-model metric '{metric_name}' not found in eval results. "
                f"Available metrics: {sorted(eval_results.keys())}"
            )
        return float(eval_results[normalized_name])

    if not values:
        raise KeyError(
            f"Best-model metric '{metric_name}' could not be resolved from eval results."
        )
    return float(sum(values) / len(values))


def is_better_metric(candidate: float, current_best: float | None, mode: str) -> bool:
    """Compare two scalar metrics according to optimization mode."""
    if current_best is None:
        return True
    if mode == "max":
        return candidate > current_best
    return candidate < current_best


def sanitize_run_name(name: str) -> str:
    """Convert a run name into a filesystem-safe directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "run"


def get_local_best_checkpoint_path(run, wandb_cfg: dict):
    """Return local outputs path for best checkpoint, or None if disabled."""
    if not wandb_cfg.get("save_local_best_checkpoint", True):
        return None

    base_dir = Path(wandb_cfg.get("local_checkpoint_dir", "outputs"))
    run_name = sanitize_run_name(run.name or run.id or "run")
    filename = wandb_cfg.get("local_checkpoint_name", "best_model.pt")
    local_dir = base_dir / f"{run_name}-{run.id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir / filename


def maybe_save_best_wandb_checkpoint(
    run,
    cfg: dict,
    model,
    optimizer,
    resolved_model_config,
    eval_results: dict,
    epoch: int,
    best_artifact_info: dict,
):
    """Save a checkpoint when the configured best-model metric improves."""
    wandb_cfg = cfg.get("wandb", {})
    if run is None or not wandb_cfg.get("log_model", False):
        return best_artifact_info

    metric_name = best_artifact_info["metric_name"]
    metric_mode = best_artifact_info["metric_mode"]
    metric_value = resolve_selection_metric(eval_results, metric_name)

    if not is_better_metric(metric_value, best_artifact_info.get("metric_value"), metric_mode):
        return best_artifact_info

    checkpoint_path = Path(run.dir) / "best_model.pt"
    checkpoint_payload = {
        "epoch": epoch,
        "best_model_metric": metric_name,
        "best_model_metric_mode": metric_mode,
        "best_model_metric_value": metric_value,
        "eval_results": eval_results,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "resolved_model_config": resolved_model_config.to_dict(),
        "raw_config": cfg,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    local_checkpoint_path = get_local_best_checkpoint_path(run, wandb_cfg)
    if local_checkpoint_path is not None:
        torch.save(checkpoint_payload, local_checkpoint_path)

    best_artifact_info.update(
        {
            "metric_value": metric_value,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "local_checkpoint_path": str(local_checkpoint_path) if local_checkpoint_path else None,
        }
    )

    run.summary["best_model/metric_name"] = metric_name
    run.summary["best_model/metric_mode"] = metric_mode
    run.summary["best_model/metric_value"] = metric_value
    run.summary["best_model/epoch"] = epoch
    if local_checkpoint_path is not None:
        run.summary["best_model/local_checkpoint_path"] = str(local_checkpoint_path)

    return best_artifact_info


def finalize_wandb_run(
    run,
    cfg: dict,
    model,
    optimizer,
    resolved_model_config,
    best_metrics: dict,
    best_artifact_info: dict,
):
    """Write final summaries and optional model artifact to W&B."""
    if run is None:
        return

    wandb_cfg = cfg.get("wandb", {})
    for key, value in best_metrics.items():
        run.summary[f"best/{key}"] = value

    if wandb_cfg.get("log_model", False) and best_artifact_info.get("checkpoint_path"):
        artifact = wandb.Artifact(
            name=wandb_cfg.get("artifact_name", "mixformer-model"),
            type="model",
            metadata={
                "metric_name": best_artifact_info.get("metric_name"),
                "metric_mode": best_artifact_info.get("metric_mode"),
                "metric_value": best_artifact_info.get("metric_value"),
                "epoch": best_artifact_info.get("epoch"),
            },
        )
        artifact.add_file(best_artifact_info["checkpoint_path"], name="best_model.pt")
        run.log_artifact(artifact)

    run.finish()


def maybe_skip_invalid_model_shape(cfg: dict) -> bool:
    """Skip invalid custom head_dim/num_heads combinations during W&B sweeps.

    Outside sweep runs, invalid configurations still raise as normal later.
    """
    model_cfg = cfg.get("model", {})
    if model_cfg.get("size", "small") != "custom":
        return False

    num_heads = model_cfg.get("num_heads", 16)
    head_dim = model_cfg.get("head_dim", 384)
    if head_dim % num_heads == 0:
        return False

    if os.environ.get("WANDB_SWEEP_ID"):
        init_wandb_skip_run(
            cfg,
            (
                "Skipping invalid sweep configuration: "
                f"head_dim ({head_dim}) must be divisible by num_heads ({num_heads})."
            ),
            skip_type="invalid_model_shape",
        )
        return True

    return False


def maybe_skip_excessive_parameter_count(cfg: dict, num_params: int) -> bool:
    """Skip sweep runs whose parameter count exceeds the configured budget."""
    sweep_cfg = cfg.get("sweep", {})
    max_parameters = sweep_cfg.get("max_parameters")
    if max_parameters in (None, "", 0):
        return False

    if num_params <= int(max_parameters):
        return False

    message = (
        "Skipping configuration because parameter count exceeds budget: "
        f"{num_params:,} > {int(max_parameters):,}."
    )

    if os.environ.get("WANDB_SWEEP_ID"):
        init_wandb_skip_run(cfg, message, skip_type="parameter_budget")
        return True

    raise ValueError(message)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, num_tasks):
    """Evaluate model on validation/test data.

    Returns:
        Dictionary with AUC and UAUC for each task.
    """
    model.eval()
    all_logits = [[] for _ in range(num_tasks)]
    all_labels = [[] for _ in range(num_tasks)]
    all_user_ids = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        non_seq, seq, seq_mask, labels, user_ids = batch
        non_seq = [x.to(device) for x in non_seq]
        seq = [x.to(device) for x in seq]
        seq_mask = seq_mask.to(device)

        logits = model(non_seq, seq, seq_mask)

        for i in range(num_tasks):
            all_logits[i].append(logits[i].cpu())
            all_labels[i].append(labels[i])
        all_user_ids.append(user_ids)

    # Concatenate
    user_ids_np = torch.cat(all_user_ids).numpy()

    results = {}
    auc_values = []
    uauc_values = []
    for i in range(num_tasks):
        preds = torch.sigmoid(torch.cat(all_logits[i])).numpy()
        targets = torch.cat(all_labels[i]).numpy()

        auc = compute_auc(preds, targets)
        uauc = compute_uauc(preds, targets, user_ids_np)

        auc_values.append(auc)
        uauc_values.append(uauc)

        results[f"task_{i}/AUC"] = auc
        results[f"task_{i}/UAUC"] = uauc

    results["mean/AUC"] = float(sum(auc_values) / len(auc_values))
    results["mean/UAUC"] = float(sum(uauc_values) / len(uauc_values))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MixFormer Training")

    # Config file (highest priority source)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (e.g. configs/experiments/debug.yaml). "
             "Overrides default.yaml, then CLI args override config file."
    )

    # Model (CLI overrides for config file values)
    parser.add_argument(
        "--model_size", type=str, default=None, choices=["small", "medium", "custom"],
        help="Model size preset"
    )
    parser.add_argument(
        "--model_type", type=str, default=None, choices=["mixformer", "ui"],
        help="Model variant: 'mixformer' (unified) or 'ui' (user-item decoupled)"
    )
    parser.add_argument("--num_layers", type=int, default=None, help="Override number of layers")
    parser.add_argument("--num_heads", type=int, default=None, help="Override number of heads")
    parser.add_argument("--head_dim", type=int, default=None, help="Override head dimension")

    # Data
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--train_samples", type=int, default=None, help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=None, help="Number of evaluation samples")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of prediction tasks")

    # Training
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "rmsprop", "sgd"])
    parser.add_argument(
        "--max_parameters", type=int, default=None,
        help="Maximum allowed trainable parameters; over-budget sweep runs are skipped"
    )

    # W&B
    wandb_switch = parser.add_mutually_exclusive_group()
    wandb_switch.add_argument(
        "--wandb", dest="wandb_enabled", action="store_true",
        help="Enable Weights & Biases logging"
    )
    wandb_switch.add_argument(
        "--no_wandb", dest="wandb_enabled", action="store_false",
        help="Disable Weights & Biases logging"
    )
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--wandb_mode", type=str, default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode"
    )
    parser.add_argument(
        "--wandb_tags", type=str, default=None,
        help="Comma-separated W&B tags"
    )
    parser.add_argument("--wandb_group_name", type=str, default=None, help="W&B group name")

    # System
    parser.add_argument("--device", type=str, default=None, help="Device: auto/cpu/cuda")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args) -> dict:
    """Apply command-line arguments as overrides to the loaded YAML config.

    Only non-None CLI values override the YAML config.
    """
    # Model overrides
    if args.model_type is not None:
        cfg["model"]["type"] = args.model_type
    if args.model_size is not None:
        cfg["model"]["size"] = args.model_size
    if args.num_layers is not None:
        cfg["model"]["num_layers"] = args.num_layers
        if cfg["model"].get("size") in ("small", "medium"):
            cfg["model"]["size"] = "custom"
    if args.num_heads is not None:
        cfg["model"]["num_heads"] = args.num_heads
        if cfg["model"].get("size") in ("small", "medium"):
            cfg["model"]["size"] = "custom"
    if args.head_dim is not None:
        cfg["model"]["head_dim"] = args.head_dim
        if cfg["model"].get("size") in ("small", "medium"):
            cfg["model"]["size"] = "custom"
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout

    # Training overrides
    if args.num_epochs is not None:
        cfg["train"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.optimizer is not None:
        cfg["train"]["optimizer"] = args.optimizer
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.device is not None:
        cfg["train"]["device"] = args.device

    # Sweep / search budget overrides
    cfg.setdefault("sweep", {})
    if args.max_parameters is not None:
        cfg["sweep"]["max_parameters"] = args.max_parameters

    # W&B overrides
    cfg.setdefault("wandb", {})
    if args.wandb_enabled is not None:
        cfg["wandb"]["enabled"] = args.wandb_enabled
    if args.wandb_project is not None:
        cfg["wandb"]["project"] = args.wandb_project
    if args.wandb_run_name is not None:
        cfg["wandb"]["name"] = args.wandb_run_name
    if args.wandb_mode is not None:
        cfg["wandb"]["mode"] = args.wandb_mode
    if args.wandb_tags is not None:
        cfg["wandb"]["tags"] = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    if args.wandb_group_name is not None:
        cfg["wandb"]["group"] = args.wandb_group_name

    # Data overrides
    if args.max_seq_len is not None:
        cfg["data"]["max_seq_len"] = args.max_seq_len
    if args.train_samples is not None:
        cfg["data"]["train_samples"] = args.train_samples
    if args.eval_samples is not None:
        cfg["data"]["eval_samples"] = args.eval_samples
    if args.num_tasks is not None:
        cfg["data"]["num_tasks"] = args.num_tasks

    return cfg


def build_optimizer(model, train_cfg):
    """Build optimizer from config."""
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("lr", 1e-3)
    wd = train_cfg.get("weight_decay", 1e-5)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def main():
    args = parse_args()

    # Load YAML config (default.yaml as base, user config as override)
    cfg = load_config(args.config)

    # Apply CLI overrides on top of YAML config
    cfg = apply_cli_overrides(cfg, args)

    if maybe_skip_invalid_model_shape(cfg):
        return

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    # Set seed
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device_str = train_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Build MixFormerConfig from YAML
    config = build_model_config(cfg)

    # Print summary
    print_config_summary(cfg, config)
    print(f"Using device: {device}")

    # Build model
    model_type = model_cfg.get("type", "mixformer")
    if model_type == "mixformer":
        model = MixFormer(config).to(device)
    else:
        model = UIMixFormer(config).to(device)

    num_params = model.count_parameters()
    print(f"Model: {model_type}, Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    if maybe_skip_excessive_parameter_count(cfg, num_params):
        return

    # Parse feature specs for dataset creation
    non_seq_specs, seq_specs, _, _, non_seq_names, seq_names = parse_feature_specs(cfg)

    # Build datasets
    train_samples = data_cfg.get("train_samples", 10000)
    eval_samples = data_cfg.get("eval_samples", 2000)
    num_tasks = data_cfg.get("num_tasks", 2)
    max_seq_len = data_cfg.get("max_seq_len", 64)
    data_source = data_cfg.get("source", "synthetic")

    if data_source == "synthetic":
        print(f"Generating synthetic data: {train_samples} train, {eval_samples} eval")
        train_dataset = SyntheticRecDataset(
            num_samples=train_samples,
            non_seq_feature_specs=non_seq_specs,
            seq_feature_specs=seq_specs,
            max_seq_len=max_seq_len,
            num_tasks=num_tasks,
            seed=seed,
        )
        eval_dataset = SyntheticRecDataset(
            num_samples=eval_samples,
            non_seq_feature_specs=non_seq_specs,
            seq_feature_specs=seq_specs,
            max_seq_len=max_seq_len,
            num_tasks=num_tasks,
            seed=seed + 1,
        )
    elif data_source == "local_file":
        train_path = data_cfg.get("train_path")
        eval_path = data_cfg.get("eval_path")
        file_format = data_cfg.get("file_format", "jsonl")
        label_columns = data_cfg.get("label_columns", [])
        user_id_column = data_cfg.get("user_id_column", "user_id")
        sequence_separator = data_cfg.get("sequence_separator", " ")

        if not train_path or not eval_path:
            raise ValueError(
                "data.train_path and data.eval_path must be set when data.source=local_file"
            )
        if len(label_columns) != num_tasks:
            raise ValueError(
                f"data.num_tasks ({num_tasks}) must match len(data.label_columns) "
                f"({len(label_columns)})"
            )

        print(f"Loading local dataset: train={train_path}, eval={eval_path}")
        train_dataset = LocalFileRecDataset(
            file_path=train_path,
            non_seq_feature_names=non_seq_names,
            seq_feature_names=seq_names,
            label_columns=label_columns,
            user_id_column=user_id_column,
            max_seq_len=max_seq_len,
            file_format=file_format,
            sequence_separator=sequence_separator,
        )
        eval_dataset = LocalFileRecDataset(
            file_path=eval_path,
            non_seq_feature_names=non_seq_names,
            seq_feature_names=seq_names,
            label_columns=label_columns,
            user_id_column=user_id_column,
            max_seq_len=max_seq_len,
            file_format=file_format,
            sequence_separator=sequence_separator,
        )
    else:
        raise ValueError(
            f"Unknown data.source: {data_source}. Use synthetic or local_file."
        )

    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=rec_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=rec_collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer
    optimizer = build_optimizer(model, train_cfg)

    # W&B
    run = init_wandb_run(
        cfg=cfg,
        resolved_model_config=config,
        model=model,
        device=device,
        num_params=num_params,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Gradient clipping
    grad_clip = train_cfg.get("grad_clip_norm", None)

    # Training loop
    num_epochs = train_cfg.get("num_epochs", 5)
    best_metrics = {"train/loss": float("inf")}
    best_metric_name, best_metric_mode = get_wandb_model_selection_config(cfg.get("wandb", {}))
    best_artifact_info = {
        "metric_name": best_metric_name,
        "metric_mode": best_metric_mode,
        "metric_value": None,
        "epoch": None,
        "checkpoint_path": None,
        "local_checkpoint_path": None,
    }
    print(f"\n{'='*60}")
    print(f"Starting training: {num_epochs} epochs")
    print(f"{'='*60}")

    try:
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, num_tasks, grad_clip
            )
            train_time = time.time() - t0

            t0 = time.time()
            eval_results = evaluate(model, eval_loader, device, num_tasks)
            eval_time = time.time() - t0

            best_metrics["train/loss"] = min(best_metrics["train/loss"], train_loss)
            for key, value in eval_results.items():
                best_metrics[key] = max(best_metrics.get(key, float("-inf")), value)

            best_artifact_info = maybe_save_best_wandb_checkpoint(
                run,
                cfg,
                model,
                optimizer,
                config,
                eval_results,
                epoch,
                best_artifact_info,
            )

            log_wandb_epoch(run, epoch, train_loss, train_time, eval_results, eval_time, optimizer)

            # Print results
            metrics_str = "  ".join(
                f"{k}: {v:.4f}" for k, v in eval_results.items()
            )
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Loss: {train_loss:.4f} | {metrics_str} | "
                f"Train: {train_time:.1f}s | Eval: {eval_time:.1f}s"
            )
    finally:
        finalize_wandb_run(
            run,
            cfg,
            model,
            optimizer,
            config,
            best_metrics,
            best_artifact_info,
        )

    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
