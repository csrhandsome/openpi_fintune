"""Episode-held-out supervised probes for frozen tactile encoders.

Labels are read from real LeRobot frame columns. The encoder is never trained
by this module. Feature extraction streams to disk, and only train statistics
are used to normalize regression targets. Validation selects the head; test
predictions are generated once after selection.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml

from force_encoder.data import LeRobotTactileDataset
from force_encoder.evaluation.artifacts import make_group_id
from force_encoder.evaluation.artifacts import make_sample_id
from force_encoder.model import ImageEncoder
from force_encoder.model import load_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/train/force_encoder_benchmark.yaml"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    for source in config["data"]["sources"]:
        source["root"] = str(_project_path(source["root"]))
    for key in ("checkpoint", "output_dir"):
        config[key] = str(_project_path(config[key]))
    return config


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_pretraining_split(checkpoint: dict, config: dict) -> None:
    """Reject probes whose test episodes participated in self-supervision."""
    previous = checkpoint.get("config", {})
    pretrain_data = previous.get("data", {})
    data = config["data"]
    if not data.get("test_fraction", 0) or not pretrain_data.get("test_fraction", 0):
        raise ValueError(
            "A held-out test split is required in both benchmark and pretraining. "
            "Old checkpoints without test_fraction cannot provide an uncontaminated benchmark; retrain JEPA."
        )
    if previous.get("seed") != config["seed"]:
        raise ValueError("Benchmark split seed must match the pretraining checkpoint")
    for key, default in (("val_fraction", 0.05), ("test_fraction", 0), ("max_episodes_per_source", None)):
        if pretrain_data.get(key, default) != data.get(key, default):
            raise ValueError(f"Benchmark {key} must match pretraining to prevent test leakage")
    sources = {source["name"]: source for source in pretrain_data.get("sources", [])}
    for source in data["sources"]:
        previous_source = sources.get(source["name"])
        if (
            previous_source is None
            or _project_path(previous_source["root"]).resolve() != _project_path(source["root"]).resolve()
        ):
            raise ValueError(f"Benchmark source {source['name']!r} must match a pretraining source")


class FrameLabelReader:
    """Read the selected episode only, with bounded parquet/episode caches."""

    def __init__(self, key: str, cache_size: int = 8):
        self.key = key
        self.cache_size = cache_size
        self._files: OrderedDict[str, Any] = OrderedDict()
        self._episodes: OrderedDict[tuple[str, int], dict[int, np.ndarray]] = OrderedDict()

    def __call__(self, metadata: dict) -> np.ndarray:
        import pyarrow.parquet as pq

        path, episode = str(metadata["data_path"]), int(metadata["episode_index"])
        cache_key = path, episode
        if cache_key not in self._episodes:
            if path not in self._files:
                parquet = pq.ParquetFile(path)
                required = {"episode_index", "frame_index", self.key}
                missing = required - set(parquet.schema_arrow.names)
                if missing:
                    parquet.close()
                    raise ValueError(
                        f"{path}: missing supervised columns {sorted(missing)}; labels are never fabricated"
                    )
                self._files[path] = parquet
                if len(self._files) > self.cache_size:
                    self._files.popitem(last=False)[1].close()
            self._files.move_to_end(path)
            parquet = self._files[path]
            episode_column = parquet.schema_arrow.get_field_index("episode_index")
            rows = {}
            for group in range(parquet.num_row_groups):
                stats = parquet.metadata.row_group(group).column(episode_column).statistics
                if stats is not None and stats.has_min_max and not stats.min <= episode <= stats.max:
                    continue
                columns = list(dict.fromkeys(("episode_index", "frame_index", self.key)))
                for row in parquet.read_row_group(group, columns=columns).to_pylist():
                    if int(row["episode_index"]) != episode:
                        continue
                    frame = int(row["frame_index"])
                    if frame in rows:
                        raise ValueError(f"{path}: duplicate frame_index {frame} in episode {episode}")
                    label = np.asarray(row[self.key], dtype=np.float32).reshape(-1)
                    if not label.size or not np.isfinite(label).all():
                        raise ValueError(f"{path}: invalid {self.key} for episode {episode}, frame {frame}")
                    rows[frame] = label
            self._episodes[cache_key] = rows
            if len(self._episodes) > self.cache_size:
                self._episodes.popitem(last=False)
        self._episodes.move_to_end(cache_key)
        try:
            return self._episodes[cache_key][int(metadata["frame_index"])].copy()
        except KeyError as error:
            raise ValueError(f"{path}: missing label for episode {episode}, frame {metadata['frame_index']}") from error

    def close(self) -> None:
        for parquet in self._files.values():
            parquet.close()
        self._files.clear()
        self._episodes.clear()


def selected_indices(length: int, cap: int | None) -> np.ndarray:
    """Deterministic coverage across the complete episode/camera index."""
    if cap is not None and cap < 1:
        raise ValueError("max_samples_per_split must be positive or null")
    count = min(length, cap) if cap is not None else length
    return np.linspace(0, length - 1, count, dtype=np.int64)


class _SelectedImages(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


@torch.inference_mode()
def extract_features(encoder: ImageEncoder, dataset: LeRobotTactileDataset, output: Path, config: dict) -> dict:
    """Write embeddings, raw targets and exact selected identities to disk."""
    output.mkdir(parents=True, exist_ok=True)
    cfg = config["training"]
    indices = selected_indices(len(dataset), cfg.get("max_samples_per_split"))
    if not len(indices):
        raise ValueError("Benchmark split contains no samples")
    reader = FrameLabelReader(config["task"]["target_key"])
    first = reader(dataset.sample_metadata(int(indices[0])))
    features = np.lib.format.open_memmap(
        output / "features.npy", mode="w+", dtype=np.float32, shape=(len(indices), encoder.embed_dim)
    )
    targets = np.lib.format.open_memmap(
        output / "targets.npy", mode="w+", dtype=np.float32, shape=(len(indices), len(first))
    )
    identity_hash = hashlib.sha256()
    groups: set[str] = set()
    try:
        with (output / "samples.jsonl").open("w") as stream:
            for row, index in enumerate(indices):
                metadata = dataset.sample_metadata(int(index))
                target = reader(metadata)
                if target.shape != first.shape:
                    raise ValueError("Target dimensionality changes between benchmark frames")
                targets[row] = target
                identity = {
                    "source": metadata["source"],
                    "episode_index": metadata["episode_index"],
                    "video_key": metadata["video_key"],
                    "frame_index": metadata["frame_index"],
                }
                line = json.dumps(identity, sort_keys=True)
                identity_hash.update((line + "\n").encode())
                stream.write(line + "\n")
                groups.add(make_group_id(identity["source"], identity["episode_index"]))
        device = next(encoder.parameters()).device
        workers = int(cfg.get("num_workers", 0))
        kwargs = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
        loader = DataLoader(
            _SelectedImages(dataset, indices),
            batch_size=cfg.get("feature_batch_size", 64),
            num_workers=workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
            **kwargs,
        )
        offset = 0
        for images in loader:
            encoded = encoder(images.to(device, non_blocking=True)).float().cpu().numpy()
            features[offset : offset + len(encoded)] = encoded
            offset += len(encoded)
        if offset != len(indices):
            raise RuntimeError("Incomplete feature extraction")
        features.flush()
        targets.flush()
    finally:
        reader.close()
        dataset.close()
    result = {
        "samples": len(indices),
        "available_samples": len(dataset),
        "target_dim": len(first),
        "feature_dim": encoder.embed_dim,
        "split_fingerprint": identity_hash.hexdigest(),
        "groups": sorted(groups),
        "split": dataset.split,
        "sample_cap": cfg.get("max_samples_per_split"),
        "data_summary": dataset.summary,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


class FeatureDataset(Dataset):
    def __init__(self, directory: Path, task: str, mean: np.ndarray, scale: np.ndarray, classes: list[int]):
        self.features = np.load(directory / "features.npy", mmap_mode="r")
        self.targets = np.load(directory / "targets.npy", mmap_mode="r")
        self.task, self.mean, self.scale = task, mean, scale
        self.classes = {label: index for index, label in enumerate(classes)}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = torch.from_numpy(np.array(self.features[index], copy=True))
        if self.task == "classification":
            label = torch.tensor(self.classes[int(self.targets[index, 0])], dtype=torch.long)
        else:
            label = torch.from_numpy(((self.targets[index] - self.mean) / self.scale).astype(np.float32))
        return feature, label


def make_head(input_dim: int, output_dim: int, config: dict) -> nn.Module:
    if config.get("type", "linear") == "linear":
        return nn.Linear(input_dim, output_dim)
    if config["type"] == "mlp":
        hidden = int(config.get("hidden_dim", 256))
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.0)),
            nn.Linear(hidden, output_dim),
        )
    raise ValueError("head.type must be linear or mlp")


@torch.inference_mode()
def _validation_loss(head, loader, device, criterion) -> float:
    head.eval()
    total = 0.0
    for features, labels in loader:
        loss = criterion(head(features.to(device)), labels.to(device))
        total += float(loss) * len(features)
    return total / len(loader.dataset)


@torch.inference_mode()
def _predict(head, loader, device, task, mean, scale, classes) -> np.ndarray:
    head.eval()
    predictions = []
    for features, _labels in loader:
        result = head(features.to(device)).cpu().numpy()
        predictions.append(result.argmax(axis=1) if task == "classification" else result * scale + mean)
    return np.concatenate(predictions)


def _target_statistics(directory: Path, task: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    targets = np.load(directory / "targets.npy", mmap_mode="r")
    if task == "classification":
        if targets.shape[1] != 1 or not np.equal(targets, np.floor(targets)).all():
            raise ValueError("Classification requires one integer-valued frame column, e.g. task_index")
        classes = sorted(int(value) for value in np.unique(targets))
        if len(classes) < 2:
            raise ValueError("Classification requires at least two classes in the training split")
        return np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32), classes
    if task != "regression":
        raise ValueError("task.type must be regression or classification")
    # Accumulate float64 moments in blocks without copying the full target array.
    total, squares = np.zeros(targets.shape[1]), np.zeros(targets.shape[1])
    for start in range(0, len(targets), 65536):
        block = np.asarray(targets[start : start + 65536], dtype=np.float64)
        total += block.sum(axis=0)
        squares += np.square(block).sum(axis=0)
    mean = total / len(targets)
    variance = np.maximum(squares / len(targets) - mean**2, 0)
    if np.all(variance <= 1e-12):
        raise ValueError(
            "All regression target dimensions are constant in train; check for placeholder labels before benchmarking"
        )
    scale = np.sqrt(variance)
    return mean.astype(np.float32), np.maximum(scale, 1e-6).astype(np.float32), []


def _identities(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    sample_ids, group_ids = [], []
    with (directory / "samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            group = make_group_id(row["source"], row["episode_index"])
            sample_ids.append(make_sample_id(row["source"], row["episode_index"], row["video_key"], row["frame_index"]))
            group_ids.append(group)
    return np.asarray(sample_ids), np.asarray(group_ids)


def train_head(config: dict, manifests: dict, output: Path, seed: int, metadata: dict) -> Path:
    from force_encoder.evaluation import compute_metrics
    from force_encoder.evaluation import save_evaluation_artifact

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cfg = config["training"]
    device = torch.device(cfg["device"])
    task = config["task"]["type"]
    cache = output / "features"
    mean, scale, classes = _target_statistics(cache / "train", task)
    datasets = {split: FeatureDataset(cache / split, task, mean, scale, classes) for split in ("train", "val", "test")}
    if task == "classification":
        for split in ("val", "test"):
            values = datasets[split].targets
            if values.shape[1] != 1 or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{split}: classification labels must be scalar integers")
            unknown = {int(value) for value in np.unique(values)} - set(classes)
            if unknown:
                raise ValueError(f"{split} contains classes absent from train: {sorted(unknown)}")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=split == "train",
            generator=torch.Generator().manual_seed(seed),
        )
        for split, dataset in datasets.items()
    }
    output_dim = len(classes) if classes else manifests["train"]["target_dim"]
    head = make_head(manifests["train"]["feature_dim"], output_dim, config["head"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.0))
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    run_dir = output / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    with (run_dir / "training.jsonl").open("w") as log:
        for epoch in range(cfg["epochs"]):
            head.train()
            total = 0.0
            for features, labels in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(features.to(device)), labels.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite supervised loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(features)
            val_loss = _validation_loss(head, loaders["val"], device, criterion)
            record = {"epoch": epoch + 1, "train_loss": total / len(datasets["train"]), "val_loss": val_loss}
            log.write(json.dumps(record) + "\n")
            print(json.dumps({"seed": seed, **record}), flush=True)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                torch.save(
                    {
                        "head": head.state_dict(),
                        "head_config": config["head"],
                        "input_dim": manifests["train"]["feature_dim"],
                        "output_dim": output_dim,
                        "target_mean": torch.from_numpy(mean),
                        "target_scale": torch.from_numpy(scale),
                        "classes": classes,
                        "task": config["task"],
                        "epoch": best_epoch,
                        "val_loss": best_loss,
                        "seed": seed,
                        "encoder": metadata,
                    },
                    run_dir / "best_head.pt",
                )
    selected = torch.load(run_dir / "best_head.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(selected["head"])
    # The test loader is consumed only after the head checkpoint is selected.
    predictions = _predict(head, loaders["test"], device, task, mean, scale, classes)
    targets = np.array(datasets["test"].targets)
    if task == "classification":
        lookup = {value: index for index, value in enumerate(classes)}
        targets = np.asarray([lookup[int(value)] for value in targets[:, 0]], dtype=np.int64)
    sample_id, group_id = _identities(cache / "test")
    names = config["task"].get("label_names")
    if names is None:
        names = (
            [str(value) for value in classes]
            if classes
            else [f"{config['task']['target_key']}[{index}]" for index in range(targets.shape[1])]
        )
    if task == "regression" and len(names) != targets.shape[1]:
        raise ValueError("task.label_names must match regression target dimension")
    (run_dir / "test_metrics.json").write_text(
        json.dumps(compute_metrics(task, targets, predictions, names), indent=2, allow_nan=False)
    )
    return save_evaluation_artifact(
        run_dir,
        task=task,
        sample_id=sample_id,
        group_id=group_id,
        y_true=targets,
        y_pred=predictions,
        label_names=names,
        dataset_fingerprint=metadata["dataset_fingerprint"],
        seed=seed,
        protocol={
            "name": "episode-held-out-train-validation-test",
            "train_group_ids": manifests["train"]["groups"],
            "validation_group_ids": manifests["val"]["groups"],
            "test_group_ids": manifests["test"]["groups"],
        },
        checkpoint=str(config["checkpoint"]),
        extra_manifest={
            **{key: value for key, value in metadata.items() if key != "dataset_fingerprint"},
            "target_key": config["task"]["target_key"],
            "units": config["task"].get("units"),
            "head": config["head"],
            "head_training": {
                "epochs": cfg["epochs"],
                "learning_rate": cfg["learning_rate"],
                "weight_decay": cfg.get("weight_decay", 0.0),
                "batch_size": cfg["batch_size"],
                "feature_normalization": "none; mean-pooled encoder output",
                "image_normalization": "ImageNet RGB mean/std; bicubic full-image resize",
                "target_normalization": "train-only per-dimension mean/std" if task == "regression" else "none",
            },
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "run_family": f"{config.get('encoder', {}).get('baseline', 'pretrained')}_{config['head'].get('type', 'linear')}",
            "original_class_ids": classes,
            "sample_cap": cfg.get("max_samples_per_split"),
        },
    )


def load_head(checkpoint: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Load a trained head; regression outputs require scale and mean below."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    head = make_head(payload["input_dim"], payload["output_dim"], payload["head_config"])
    head.load_state_dict(payload["head"])
    return head.to(device).eval().requires_grad_(requires_grad=False), payload


def run(config: dict) -> list[Path]:
    config = copy.deepcopy(config)
    if (
        config["task"]["type"] == "classification"
        and config["task"]["target_key"] == "task_index"
        and len(config["data"]["sources"]) != 1
    ):
        raise ValueError(
            "task_index classification requires exactly one source: task indices are source-local vocabularies"
        )
    cfg = config["training"]
    for key in ("epochs", "batch_size", "feature_batch_size"):
        if int(cfg.get(key, 64)) < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["learning_rate"] <= 0 or cfg.get("weight_decay", 0) < 0 or cfg.get("num_workers", 0) < 0:
        raise ValueError("Invalid optimizer or worker configuration")
    seeds = cfg.get("seeds", [0])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must contain unique seeds")
    payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    validate_pretraining_split(payload, config)
    device = torch.device(cfg["device"])
    encoder = load_encoder(config["checkpoint"], device)
    encoder_cfg = config.get("encoder", {})
    baseline = encoder_cfg.get("baseline", "pretrained")
    if baseline == "frozen_random":
        torch.manual_seed(encoder_cfg.get("random_seed", config["seed"]))
        encoder = ImageEncoder(**encoder.config).to(device).eval().requires_grad_(requires_grad=False)
    elif baseline != "pretrained":
        raise ValueError("encoder.baseline must be pretrained or frozen_random")
    output = Path(config["output_dir"])
    if (output / "config.yaml").exists():
        raise FileExistsError(f"{output} already contains a benchmark; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifests = {}
    for split in ("train", "val", "test"):
        dataset = LeRobotTactileDataset(
            **config["data"], image_size=encoder.image_size, split=split, seed=config["seed"]
        )
        manifests[split] = extract_features(encoder, dataset, output / "features" / split, config)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(manifests[left]["groups"]) & set(manifests[right]["groups"]):
            raise ValueError("Episode leakage between supervised splits")
    identity = []
    for source in config["data"]["sources"]:
        root = Path(source["root"])
        identity.append(
            {
                "name": source["name"],
                "info_sha256": _file_digest(root / "meta/info.json"),
                "episodes": [
                    (str(path.relative_to(root)), _file_digest(path))
                    for path in sorted((root / "meta/episodes").rglob("*.parquet"))
                ],
            }
        )
    metadata = {
        "baseline": baseline,
        "random_seed": encoder_cfg.get("random_seed", config["seed"]),
        "encoder_config": encoder.config,
        "checkpoint_sha256": _file_digest(config["checkpoint"]),
        "dataset_fingerprint": _digest(identity),
        "split_seed": config["seed"],
        "val_fraction": config["data"]["val_fraction"],
        "test_fraction": config["data"]["test_fraction"],
    }
    artifacts = [train_head(config, manifests, output, int(seed), metadata) for seed in seeds]
    (output / "artifacts.json").write_text(json.dumps([str(path) for path in artifacts], indent=2))
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--baseline", choices=("pretrained", "frozen_random"))
    args = parser.parse_args()
    config = read_config(args.config)
    for key in ("checkpoint", "output_dir"):
        if getattr(args, key):
            config[key] = str(_project_path(getattr(args, key)))
    for key in ("device", "epochs", "max_samples_per_split"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    if args.baseline:
        config.setdefault("encoder", {})["baseline"] = args.baseline
    print(json.dumps({"artifacts": [str(path) for path in run(config)]}), flush=True)


if __name__ == "__main__":
    main()
