"""
Synthetic recommendation dataset for MixFormer experiments.

Generates random categorical features for non-sequential (user/item/context)
and sequential (user behavior history) inputs, along with binary task labels.
"""

import ast
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SyntheticRecDataset(Dataset):
    """Synthetic recommendation dataset for testing MixFormer.

    Generates random data mimicking industrial recommendation scenarios:
      - Non-sequential features: categorical features (user, item, context)
      - Sequential features: user behavior history with categorical features
      - Binary labels for multi-task CTR prediction

    Args:
        num_samples: Number of samples to generate.
        non_seq_feature_specs: List of (vocab_size, embed_dim) for non-seq features.
        seq_feature_specs: List of (vocab_size, embed_dim) for sequential features.
        max_seq_len: Maximum sequence length.
        num_tasks: Number of binary prediction tasks.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int,
        non_seq_feature_specs: list,
        seq_feature_specs: list,
        max_seq_len: int = 512,
        num_tasks: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.num_tasks = num_tasks

        generator = torch.Generator().manual_seed(seed)

        # Generate non-sequential features: each (num_samples,)
        self.non_seq_data = []
        for vocab_size, _ in non_seq_feature_specs:
            self.non_seq_data.append(
                torch.randint(0, vocab_size, (num_samples,), generator=generator)
            )

        # Generate sequential features: each (num_samples, max_seq_len)
        self.seq_data = []
        for vocab_size, _ in seq_feature_specs:
            self.seq_data.append(
                torch.randint(
                    0, vocab_size, (num_samples, max_seq_len), generator=generator
                )
            )

        # Random sequence lengths (at least 1)
        self.seq_lengths = torch.randint(
            1, max_seq_len + 1, (num_samples,), generator=generator
        )

        # Binary labels for each task
        self.labels = []
        for _ in range(num_tasks):
            self.labels.append(
                torch.randint(0, 2, (num_samples,), generator=generator).float()
            )

        # User IDs for UAUC computation (random user assignment)
        num_users = max(num_samples // 20, 10)
        self.user_ids = torch.randint(0, num_users, (num_samples,), generator=generator)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        non_seq = [feat[idx] for feat in self.non_seq_data]
        seq = [feat[idx] for feat in self.seq_data]
        seq_len = self.seq_lengths[idx].item()

        # Create sequence mask: True for valid positions
        seq_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        seq_mask[:seq_len] = True

        labels = [task_labels[idx] for task_labels in self.labels]
        user_id = self.user_ids[idx]

        return non_seq, seq, seq_mask, labels, user_id


def _parse_scalar(value):
    """Parse a scalar value from CSV/JSONL into an integer-friendly form."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if value is None:
        return 0

    text = str(value).strip()
    if text == "":
        return 0
    return int(float(text))


def _parse_sequence(value, max_seq_len: int, separator: str = " "):
    """Parse a sequence column from JSON arrays, Python lists, or split strings."""
    if isinstance(value, list):
        seq = [_parse_scalar(v) for v in value]
    elif value is None:
        seq = []
    else:
        text = str(value).strip()
        if text == "":
            seq = []
        elif text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(text)
            seq = [_parse_scalar(v) for v in parsed]
        else:
            tokens = [token for token in text.split(separator) if token]
            seq = [_parse_scalar(token) for token in tokens]

    if len(seq) > max_seq_len:
        seq = seq[-max_seq_len:]

    seq_mask = torch.zeros(max_seq_len, dtype=torch.bool)
    if seq:
        seq_mask[:len(seq)] = True

    padded = seq + [0] * (max_seq_len - len(seq))
    return torch.tensor(padded, dtype=torch.long), seq_mask


def _load_records(file_path: str, file_format: str):
    """Load row-wise records from a local file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    normalized_format = file_format.lower()
    if normalized_format == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if normalized_format == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported file format: {file_format}. Use jsonl or csv.")


class LocalFileRecDataset(Dataset):
    """Recommendation dataset loaded from local JSONL or CSV files.

    Expected row schema:
      - each non-sequential feature name appears as a scalar column
      - each sequential feature name appears as a list column or split string
      - label_columns contains one or more binary labels
      - user_id_column identifies the user for UAUC computation

    Example JSONL row:
      {
        "user_id": 12,
        "item_id": 301,
        "category_id": 7,
        "hist_item_id": [1, 9, 20],
        "hist_category_id": [4, 4, 7],
        "label": 1
      }
    """

    def __init__(
        self,
        file_path: str,
        non_seq_feature_names: list,
        seq_feature_names: list,
        label_columns: list,
        user_id_column: str,
        max_seq_len: int,
        file_format: str = "jsonl",
        sequence_separator: str = " ",
    ):
        super().__init__()
        self.records = _load_records(file_path, file_format)
        self.non_seq_feature_names = list(non_seq_feature_names)
        self.seq_feature_names = list(seq_feature_names)
        self.label_columns = list(label_columns)
        self.user_id_column = user_id_column
        self.max_seq_len = max_seq_len
        self.sequence_separator = sequence_separator

        if not self.records:
            raise ValueError(f"Dataset file is empty: {file_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]

        non_seq = [
            torch.tensor(_parse_scalar(row.get(name, 0)), dtype=torch.long)
            for name in self.non_seq_feature_names
        ]

        seq = []
        seq_mask = None
        for name in self.seq_feature_names:
            seq_tensor, current_mask = _parse_sequence(
                row.get(name, []),
                max_seq_len=self.max_seq_len,
                separator=self.sequence_separator,
            )
            seq.append(seq_tensor)
            if seq_mask is None:
                seq_mask = current_mask

        if seq_mask is None:
            seq_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        labels = [
            torch.tensor(float(row.get(name, 0)), dtype=torch.float32)
            for name in self.label_columns
        ]
        user_id = torch.tensor(_parse_scalar(row.get(self.user_id_column, 0)), dtype=torch.long)

        return non_seq, seq, seq_mask, labels, user_id


def rec_collate_fn(batch):
    """Custom collate function for recommendation dataset.

    Stacks features from list-of-lists into batched tensors.

    Returns:
        non_seq_batch: list of (B,) tensors, one per non-seq feature
        seq_batch: list of (B, T) tensors, one per seq feature
        seq_mask: (B, T) boolean tensor
        labels_batch: list of (B,) tensors, one per task
        user_ids: (B,) tensor
    """
    non_seq_lists = [item[0] for item in batch]
    seq_lists = [item[1] for item in batch]
    seq_masks = [item[2] for item in batch]
    labels_lists = [item[3] for item in batch]
    user_ids = [item[4] for item in batch]

    # Stack non-sequential features
    num_non_seq = len(non_seq_lists[0])
    non_seq_batch = [
        torch.stack([sample[i] for sample in non_seq_lists])
        for i in range(num_non_seq)
    ]

    # Stack sequential features
    num_seq = len(seq_lists[0])
    seq_batch = [
        torch.stack([sample[i] for sample in seq_lists])
        for i in range(num_seq)
    ]

    # Stack masks
    seq_mask = torch.stack(seq_masks)

    # Stack labels
    num_tasks = len(labels_lists[0])
    labels_batch = [
        torch.stack([sample[i] for sample in labels_lists])
        for i in range(num_tasks)
    ]

    # Stack user IDs
    user_ids = torch.stack(user_ids)

    return non_seq_batch, seq_batch, seq_mask, labels_batch, user_ids
