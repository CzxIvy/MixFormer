# MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

PyTorch implementation of **MixFormer** — a unified Transformer-style architecture for recommender systems that jointly models sequential behaviors and feature interactions within a single backbone.

> **Paper**: *MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders*
> Xu Huang, Hao Zhang, Zhifang Fan, Yunwen Huang, Zhuoxing Wei, Zheng Chai, Jinan Ni, Yuchao Zheng, Qiwei Chen (ByteDance)
> arXiv:2602.14110

## Architecture Overview

MixFormer is an efficient decoder-only Transformer variant tailored for multi-task recommender systems:

```
Non-Sequential Input    Sequential Input
(user, item, context)   (behavior history)
        │                       │
   Embedding &              Embedding &
     Split                   Projection
        │                       │
        ▼                       │
  ┌─────────────┐               │
  │ Query Mixer │◄──────────────┤
  │ (HeadMixing │               │
  │ + Per-head  │               │
  │   FFN)      │               │
  └──────┬──────┘               │
         ▼                      │
  ┌─────────────┐               │
  │   Cross     │◄──────────────┘
  │  Attention  │  (per-layer seq FFN
  │             │   + K,V projection)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Output    │
  │   Fusion    │
  │ (Per-head   │
  │   FFN)      │
  └──────┬──────┘
         ▼
    × L blocks
         │
    Task Networks
```

### Key Components

| Module | Description | Equations |
|--------|-------------|-----------|
| **Feature Embedding & Split** | Embeds non-seq features, splits into N heads, projects each to D dims | Eq. 1-2 |
| **Query Mixer** | HeadMixing (parameter-free) + per-head SwiGLU FFN | Eq. 3-4 |
| **Cross Attention** | Per-layer seq FFN + multi-head cross attention | Eq. 5-8 |
| **Output Fusion** | Per-head SwiGLU FFN for deep fusion | Eq. 9 |
| **UI-MixFormer** | User-item decoupled variant with masked HeadMixing | Eq. 10-11 |

### Model Configurations

| Config | N (heads) | L (layers) | D (head_dim) | Paper Name |
|--------|-----------|------------|--------------|------------|
| Small  | 16        | 4          | 384          | MixFormer-small |
| Medium | 16        | 4          | 768          | MixFormer-medium |

## Project Structure

```
MixFormer/
├── config.py                  # 核心模型配置类 MixFormerConfig
├── config_loader.py           # YAML 配置加载与合并逻辑
├── train.py                   # 支持 YAML 配置和 CLI 覆盖的训练脚本
├── requirements.txt           # Python dependencies
├── configs/
│   ├── base/
│   │   └── default.yaml       # 全局默认配置模板
│   ├── experiments/
│   │   ├── debug.yaml         # CPU 快速调试配置
│   │   ├── mixformer_small.yaml
│   │   ├── mixformer_medium.yaml
│   │   ├── ui_mixformer_medium.yaml
│   │   ├── long_seq_scaling.yaml
│   │   └── minimal_features.yaml
│   ├── datasets/
│   │   └── amazon_electronics_x1.yaml
│   └── sweeps/
│       ├── bases/
│       │   └── amazon_electronics_x1.yaml
│       ├── lr_batch_num_heads.yaml
│       └── lr_batch_num_heads_head_dim.yaml
├── model/
│   ├── __init__.py
│   ├── layers.py              # RMSNorm, SwiGLUFFN, PerHeadSwiGLUFFN, HeadMixing
│   ├── mixformer_block.py     # QueryMixer, CrossAttention, OutputFusion, MixFormerBlock
│   ├── mixformer.py           # MixFormer full model with embeddings & task heads
│   └── ui_mixformer.py        # UI-MixFormer with user-item decoupling
├── data/
│   ├── __init__.py
│   └── dataset.py             # Synthetic dataset and collate function
└── utils/
    ├── __init__.py
    └── metrics.py             # AUC and UAUC metrics
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

如果你要在线记录实验到 W&B，先执行：

```bash
wandb login
```

### Training with Synthetic Data

```bash
# 推荐：使用 YAML 配置文件
python train.py --config configs/experiments/debug.yaml
python train.py --config configs/experiments/mixformer_small.yaml
python train.py --config configs/experiments/ui_mixformer_medium.yaml

# 配置文件 + 命令行覆盖（命令行优先级更高）
python train.py --config configs/experiments/mixformer_small.yaml --batch_size 128 --lr 5e-4

# 自定义特征选择示例
python train.py --config configs/experiments/minimal_features.yaml

# 开启 W&B 记录（命令行临时开启）
python train.py --config configs/experiments/debug.yaml --wandb --wandb_mode offline --wandb_project MixFormer

# 兼容旧用法：纯命令行启动
python train.py --model_size small --max_seq_len 128 --num_epochs 5 --lr 0.001
```

### Configuration System

当前项目的训练配置分为两层：

- `config.py`：定义核心配置类 `MixFormerConfig`，负责模型结构参数校验、small/medium 预设构建，以及给模型代码提供统一的 Python 对象。
- `configs/`：定义训练参数、数据参数、特征选择、模型类型等实验配置，按用途拆成多个子目录。

两者不是重复关系，而是分工关系：

- `config.py` 负责“模型内部配置对象”。
- `configs/` 负责“训练实验入口配置”。

配置优先级如下：

```text
configs/base/default.yaml
  -> 用户指定的 configs/experiments/*.yaml 或 configs/datasets/*.yaml 或其他 override
    -> 命令行参数覆盖
```

例如：

```bash
python train.py --config configs/experiments/mixformer_small.yaml --batch_size 64 --num_epochs 3
```

上面这条命令会先加载 `configs/base/default.yaml`，再加载 `configs/experiments/mixformer_small.yaml`，最后把 `batch_size` 和 `num_epochs` 用命令行值覆盖。

这里有一个重要约定：

- `configs/base/default.yaml` 只放“全项目通用默认值”，例如通用训练参数、W&B 默认项、基础特征模板
- 数据集私有信息不要回填到 `default.yaml`，例如你的 `train_path`、`eval_path`、标签列名、专属特征定义
- 针对具体数据集，应该新建一个 `configs/datasets/你的数据集.yaml` 作为 override 配置，只写和默认值不同的部分

也就是说，如果你用自己的数据集训练，正确做法不是“把数据集配置覆盖回 `default.yaml`”，而是：

```text
configs/base/default.yaml               # 全局默认模板
  + configs/datasets/你的数据集.yaml    # 数据路径 / 标签 / 特征 / 必要训练差异
  + 命令行临时覆盖           # 例如 batch_size、num_epochs、lr
```

### Feature Selection

现在特征选择已经从 `train.py` 中抽离到 YAML：

- `features.non_sequential`：定义非序列特征
- `features.sequential`：定义行为序列特征
- `features.user_groups` / `features.item_groups`：控制 UI-MixFormer 的 user/item 解耦分组

每个非序列特征格式如下：

```yaml
- { name: user_id, vocab_size: 100000, embed_dim: 32, group: user }
```

每个序列特征格式如下：

```yaml
- { name: action_item_id, vocab_size: 500000, embed_dim: 32 }
```

如果你只想快速删减特征做实验，直接复制并修改 [configs/experiments/minimal_features.yaml](configs/experiments/minimal_features.yaml) 即可。

### Real Dataset Path Configuration

真实数据集路径不在 `config.py` 里配，而是在 `configs/*.yaml` 的 `data` 段里配。

最关键的字段有这些：

```yaml
data:
  source: local_file
  file_format: jsonl
  train_path: datasets/AmazonElectronics_x1/train.jsonl
  eval_path: datasets/AmazonElectronics_x1/valid.jsonl
  user_id_column: user_id
  label_columns: [label]
  sequence_separator: " "
```

当前代码支持两种本地文件格式：

- `jsonl`：每行一个 JSON 对象，最适合带序列字段的数据
- `csv`：标量列正常存，序列列可以写成 JSON 数组字符串如 `[1,2,3]`，也可以写成空格分隔字符串如 `1 2 3`

训练时会按 `features.non_sequential` 和 `features.sequential` 里的 `name` 去数据文件里找同名列。

例如你在配置里声明了：

```yaml
features:
  non_sequential:
    - { name: user_id, vocab_size: 192404, embed_dim: 32, group: user }
    - { name: item_id, vocab_size: 63002, embed_dim: 32, group: item }
  sequential:
    - { name: hist_item_id, vocab_size: 63002, embed_dim: 32 }
```

那么你的 `train.jsonl` / `valid.jsonl` 每一行至少要包含这些列：`user_id`、`item_id`、`hist_item_id`，以及 `label_columns` 里声明的标签列。

### AmazonElectronics_x1 Example

仓库里已经加了一个可直接改的模板配置：[configs/datasets/amazon_electronics_x1.yaml](configs/datasets/amazon_electronics_x1.yaml)。

这个文件本身不是“全量配置”，而是一个数据集 override 示例：

- 它已经写了这份数据集自己的 `model`、`train`、`data`、`features`
- 没写出来的字段会自动继承 [configs/base/default.yaml](configs/base/default.yaml)
- 所以你不需要把它的内容抄回 [configs/base/default.yaml](configs/base/default.yaml)

如果你要接自己的数据集，最省事的方式通常是：

1. 复制 [configs/datasets/amazon_electronics_x1.yaml](configs/datasets/amazon_electronics_x1.yaml) 成你自己的数据集配置文件
2. 修改 `data.train_path`、`data.eval_path`、`data.label_columns`
3. 按你的真实列名和 `vocab_size` 修改 `features`
4. 如果训练策略要变，再只改你需要的 `train` 或 `model` 字段

启动方式示例：

```bash
python train.py --config configs/datasets/amazon_electronics_x1.yaml
```

这个模板默认假设你的目录结构类似：

```text
datasets/
  AmazonElectronics_x1/
    train.jsonl
    valid.jsonl
```

当前仓库已经附带了一份可直接运行的 toy 样例文件，路径就是上面的 `datasets/AmazonElectronics_x1/`。

并假设单任务二分类标签列名为 `label`。如果你的实际列名或文件格式不同，只需要改 YAML，不用改训练代码。

### Weights & Biases Tracking

当前训练脚本已经支持用 W&B 记录训练过程，配置入口在 `configs/*.yaml` 的 `wandb` 段。

最小配置示例：

```yaml
wandb:
  enabled: true
  project: MixFormer
  mode: online
  name: mixformer-small-run1
  tags: [baseline, small]
  watch_model: false
  log_model: false
```

如果你只是本地调试，不想联网，可以直接改成：

```yaml
wandb:
  enabled: true
  mode: offline
```

训练过程中会自动记录这些指标：

- `train/loss`
- `train/learning_rate`
- `train/time_sec`
- `eval/time_sec`
- 每个任务的 `eval/task_i/AUC`
- 每个任务的 `eval/task_i/UAUC`

并在 run summary 里写入每个指标的最佳值。

若 `wandb.log_model=true`，现在上传的是“最佳 checkpoint”，不是最终 epoch 权重。默认按 `task_0/AUC` 选最佳模型，也可以改成：

```yaml
wandb:
  log_model: true
  best_model_metric: task_0/AUC   # 或 mean/AUC, task_0/UAUC, eval/task_0/AUC
  best_model_mode: max
```

这意味着：

- 训练过程中每次该指标变好时，会覆盖保存本次最佳 checkpoint
- 训练结束时上传到 W&B artifact 的就是这个最佳 checkpoint
- artifact metadata 会包含最佳指标名、指标值和对应 epoch

另外，最佳 checkpoint 现在也会固定保存到本地 `outputs/` 目录。默认路径格式是：

```text
outputs/<run-name>-<run-id>/best_model.pt
```

如果你想改本地保存位置：

```yaml
wandb:
  log_model: true
  save_local_best_checkpoint: true
  local_checkpoint_dir: outputs
  local_checkpoint_name: best_model.pt
```

仓库里已经加了一个离线调试模板：[configs/experiments/debug_wandb_offline.yaml](configs/experiments/debug_wandb_offline.yaml)。

如果你想专门验证“最佳 AUC 对应的 checkpoint 会同时上传到 artifact，并落到本地 outputs/”，可以使用 [configs/experiments/debug_wandb_best_artifact.yaml](configs/experiments/debug_wandb_best_artifact.yaml)。

直接运行：

```bash
python train.py --config configs/experiments/debug_wandb_offline.yaml
```

### W&B Sweep

仓库里现在整理成两份更适合正式实验的 sweep 配置：

- [configs/sweeps/lr_batch_num_heads.yaml](configs/sweeps/lr_batch_num_heads.yaml)：主搜索配置，优先稳定探索学习率、batch、层数、head 数和 dropout
- [configs/sweeps/lr_batch_num_heads_head_dim.yaml](configs/sweeps/lr_batch_num_heads_head_dim.yaml)：结构探索配置，在主搜索基础上再放开 `head_dim`

这两份 sweep 现在都默认使用 [configs/sweeps/bases/amazon_electronics_x1.yaml](configs/sweeps/bases/amazon_electronics_x1.yaml) 作为基础训练配置，而不是调试用的 [configs/experiments/debug.yaml](configs/experiments/debug.yaml)。这意味着它们默认会直接使用仓库里的 AmazonElectronics_x1 示例数据，并采用更接近正式实验的训练轮数与 batch 设置。

如果你要换成自己的数据集，优先修改 [configs/sweeps/bases/amazon_electronics_x1.yaml](configs/sweeps/bases/amazon_electronics_x1.yaml) 里的：

- `data.train_path`
- `data.eval_path`
- `data.label_columns`
- `features.non_sequential`
- `features.sequential`

主搜索配置 [configs/sweeps/lr_batch_num_heads.yaml](configs/sweeps/lr_batch_num_heads.yaml) 当前会搜索：

- `lr`
- `batch_size`
- `num_layers`
- `num_heads`
- `dropout`

这个版本固定使用基线配置里的 `head_dim=128`，目的是先把主干搜索空间收紧，减少结构变量过多带来的噪声，更适合作为第一轮正式实验。

启动方式：

```bash
wandb login
wandb sweep configs/sweeps/lr_batch_num_heads.yaml
wandb agent <your-sweep-id>
```

主搜索当前默认优化目标是：

```text
eval/mean/AUC
```

这个指标对单任务和多任务都兼容；当前 AmazonElectronics_x1 示例是单任务，所以它会等价于该任务的 AUC。

主搜索还设置了参数量上限：

```yaml
parameters:
  max_parameters:
    value: 60000000
```

训练脚本会在模型构建后统计总参数量；如果超过上限：

- 普通训练会直接报错
- W&B sweep 模式下会自动把该 run 标记为 skipped，并把 `skip_type` 写成 `parameter_budget`

如果你想进一步放开结构搜索，可以使用 [configs/sweeps/lr_batch_num_heads_head_dim.yaml](configs/sweeps/lr_batch_num_heads_head_dim.yaml)。

它会同时搜索：

- `lr`
- `batch_size`
- `num_layers`
- `num_heads`
- `head_dim`
- `dropout`

这个版本适合第二轮实验：先用主搜索锁定大致训练区间，再用它做结构精调。由于 `head_dim=96` 与 `num_heads=16` 组合时不整除，这个配置会允许少量 run 被自动跳过；训练脚本会把这类 run 标记为 `invalid_model_shape`，不会进入正式训练。

结构搜索也启用了参数量预算限制：

```yaml
parameters:
  max_parameters:
    value: 70000000
```

启动方式：

```bash
wandb sweep configs/sweeps/lr_batch_num_heads_head_dim.yaml
wandb agent <your-sweep-id>
```

如果你要切到更稳定的多任务最佳模型选择，可以把基础配置里的 W&B 选择指标改成：

```yaml
wandb:
  best_model_metric: mean/AUC
  best_model_mode: max
```

如果你想直接验证“按最佳 AUC 上传 artifact”，可以使用 [configs/experiments/debug_wandb_best_artifact.yaml](configs/experiments/debug_wandb_best_artifact.yaml)。

### Using the Model in Code

#### Option 1: Manual Python Configuration

```python
from config import MixFormerConfig
from model import MixFormer, UIMixFormer

# Create MixFormer-small
config = MixFormerConfig.small(
    non_seq_feature_specs=[(1000, 16)] * 20,  # 20 features, vocab=1000, dim=16
    seq_feature_specs=[(10000, 32)] * 5,       # 5 seq features per action
    max_seq_len=512,
    num_tasks=2,
)
model = MixFormer(config)

# Forward pass
import torch
B, T = 4, 512
non_seq = [torch.randint(0, 1000, (B,)) for _ in range(20)]
seq = [torch.randint(0, 10000, (B, T)) for _ in range(5)]
seq_mask = torch.ones(B, T, dtype=torch.bool)

logits = model(non_seq, seq, seq_mask)  # list of (B,) tensors
print(f"Task 0 logits: {logits[0].shape}")  # (4,)
print(f"Task 1 logits: {logits[1].shape}")  # (4,)
```

#### Option 2: Load from YAML Configuration

```python
import torch

from config_loader import build_model_config, load_config
from model import MixFormer, UIMixFormer

# Load configs/base/default.yaml + user config
cfg = load_config("configs/experiments/mixformer_small.yaml")

# CLI-style overrides can also be applied manually before building the model
cfg["train"]["batch_size"] = 128
cfg["data"]["max_seq_len"] = 256

# Convert merged YAML dict into the typed MixFormerConfig object
model_config = build_model_config(cfg)

model_type = cfg["model"].get("type", "mixformer")
if model_type == "ui":
  model = UIMixFormer(model_config)
else:
  model = MixFormer(model_config)

B, T = 4, model_config.max_seq_len
non_seq = [
  torch.randint(0, vocab_size, (B,))
  for vocab_size, _ in model_config.non_seq_feature_specs
]
seq = [
  torch.randint(0, vocab_size, (B, T))
  for vocab_size, _ in model_config.seq_feature_specs
]
seq_mask = torch.ones(B, T, dtype=torch.bool)

logits = model(non_seq, seq, seq_mask)
print(model_config)
print([logit.shape for logit in logits])
```

这两种方式最终都会落到同一个 `MixFormerConfig` dataclass 对象上：

- 手动方式适合直接在 Python 代码里构建模型。
- YAML 方式适合做训练实验管理、参数切换和特征选择。

## Implementation Notes

### Faithfulness to Paper

- **HeadMixing** (Eq. 3): Parameter-free reshape-transpose-reshape operation for cross-head information exchange, replacing self-attention for heterogeneous feature heads.
- **Per-head FFN** (Eq. 4, 9): Independent SwiGLU FFN per head, implemented efficiently via batched einsum operations.
- **Per-layer Sequence FFN** (Eq. 5): Each MixFormer block has its own FFN for transforming the original sequence input — the sequence is not updated across layers.
- **Cross Attention** (Eq. 8): Query heads from the Query Mixer attend to the sequence with a residual connection.
- **Pre-RMSNorm**: All sub-layers use pre-normalization with RMSNorm (ablation shows this outperforms post-LayerNorm).
- **UI-MixFormer** (Eq. 10-11): Masked HeadMixing enforces unidirectional user→item information flow for Request Level Batching compatibility.

### Differences from Paper

- **Hidden dimensions**: The paper does not specify FFN hidden dimensions; we default to `4 × head_dim` for per-head FFNs and sequence FFNs.
- **Data**: The paper uses proprietary Douyin data (trillions of interactions, 300+ features). We provide synthetic data for demonstration.
- **Optimizer**: The paper uses RMSProp (dense) + Adagrad (sparse) in a hybrid distributed framework. We use AdamW for simplicity.
- **D=384 vs 386**: The paper reports D=386 for small; we use D=384 (divisible by N=16 for HeadMixing).

## Citation

```bibtex
@inproceedings{huang2026mixformer,
  title={MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders},
  author={Huang, Xu and Zhang, Hao and Fan, Zhifang and Huang, Yunwen and Wei, Zhuoxing and Chai, Zheng and Ni, Jinan and Zheng, Yuchao and Chen, Qiwei},
  year={2026},
  eprint={2602.14110},
  archivePrefix={arXiv}
}
```