"""Frozen pooled-data evaluator and reproducibility artifacts for alpha studies."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import PretrainedModel


def scenario_id(seed, dimension, poison_ratio, target):
    raw = json.dumps(
        {"seed": seed, "dimension": dimension, "poison_ratio": poison_ratio,
         "target": target}, sort_keys=True, separators=(",", ":")
    )
    return f"scenario_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def pooled_clean_arrays(records):
    X = np.concatenate([row["X"] for row in records]).astype(np.float32)
    clean_y = np.concatenate([row["clean_y"] for row in records]).astype(np.float32)
    observed_y = np.concatenate([row["observed_y"] for row in records]).astype(np.float32)
    agent_ids = np.concatenate([
        np.full(len(row["X"]), row["agent_id"], dtype=np.int64) for row in records
    ])
    poisoned = np.concatenate([
        np.full(len(row["X"]), row["is_poisoned"], dtype=bool) for row in records
    ])
    return X, clean_y, observed_y, agent_ids, poisoned


def deterministic_split(n_samples, seed, validation_fraction=0.15, test_fraction=0.15):
    if n_samples < 3:
        raise ValueError("External evaluator requires at least three samples.")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    indices = np.random.default_rng(seed).permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_fraction)))
    n_validation = max(1, int(round(n_samples * validation_fraction)))
    if n_test + n_validation >= n_samples:
        raise ValueError("Split leaves no pooled training samples.")
    return {
        "train": np.sort(indices[n_test + n_validation:]),
        "validation": np.sort(indices[n_test:n_test + n_validation]),
        "test": np.sort(indices[:n_test]),
    }


def _predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy().reshape(-1)


def synthetic_truth(X):
    X = np.asarray(X, dtype=float)
    values = np.sum(X, axis=-1)
    if X.shape[-1] > 1:
        values = values + np.sum(X[..., :-1] * X[..., 1:], axis=-1)
    return np.asarray(values, dtype=float)


def train_frozen_evaluator(X, y, n_features, seed, epochs=100, batch_size=64,
                           validation_fraction=0.15, test_fraction=0.15):
    """Train with validation checkpointing; test data never selects the checkpoint."""
    splits = deterministic_split(len(X), seed, validation_fraction, test_fraction)
    torch.manual_seed(seed)
    model = PretrainedModel(in_features=n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train = TensorDataset(torch.as_tensor(X[splits["train"]]),
                          torch.as_tensor(y[splits["train"]]).unsqueeze(1))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    best_state = None
    best_validation_mse = float("inf")
    best_epoch = None
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        validation_prediction = _predict(model, X[splits["validation"]])
        validation_mse = float(np.mean((validation_prediction - y[splits["validation"]]) ** 2))
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_prediction = _predict(model, X[splits["test"]])
    test_error = test_prediction - y[splits["test"]]
    test_truth_error = test_prediction - synthetic_truth(X[splits["test"]])
    test_mse = float(np.mean(test_error ** 2))
    hard_threshold = float(np.quantile(np.abs(y[splits["test"]]), 0.75))
    hard_mask = np.abs(y[splits["test"]]) >= hard_threshold
    hard_test_mse = float(np.mean(test_error[hard_mask] ** 2))
    test_truth_mse = float(np.mean(test_truth_error ** 2))
    hard_test_truth_mse = float(np.mean(test_truth_error[hard_mask] ** 2))
    metrics = {
        "best_epoch": best_epoch,
        "validation_mse": best_validation_mse,
        "test_mse": test_mse,
        "hard_test_mse": hard_test_mse,
        "test_truth_mse": test_truth_mse,
        "hard_test_truth_mse": hard_test_truth_mse,
        "hard_test_threshold_abs_y": hard_threshold,
        "train_samples": len(splits["train"]),
        "validation_samples": len(splits["validation"]),
        "test_samples": len(splits["test"]),
    }
    return model, splits, metrics


def save_scenario_artifacts(root, identity, agents, records, evaluator_epochs=100,
                            validation_fraction=0.15, test_fraction=0.15):
    artifact_dir = Path(root) / "artifacts" / scenario_id(**identity)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    X, clean_y, observed_y, agent_ids, poisoned = pooled_clean_arrays(records)
    evaluator_seed = int(identity["seed"]) + 104729
    evaluator, splits, metrics = train_frozen_evaluator(
        X, clean_y, identity["dimension"], evaluator_seed, evaluator_epochs,
        validation_fraction=validation_fraction, test_fraction=test_fraction,
    )
    test_x = X[splits["test"]]
    target_mask = np.abs(synthetic_truth(test_x) - identity["target"]) <= 0.25
    target_distance = np.abs(synthetic_truth(test_x) - identity["target"])
    target_errors = _predict(evaluator, test_x) - synthetic_truth(test_x)
    metrics["target_near_samples"] = int(target_mask.sum())
    metrics["target_near_truth_mse"] = float(np.mean(target_errors[target_mask]**2)) if target_mask.any() else None
    metrics["target_near_sparse"] = int(target_mask.sum()) < 30
    coverage_bins = (
        ("0_to_0p1", target_distance <= 0.1),
        ("0p1_to_0p25", (target_distance > 0.1) & (target_distance <= 0.25)),
        ("0p25_to_0p5", (target_distance > 0.25) & (target_distance <= 0.5)),
        ("0p5_to_1", (target_distance > 0.5) & (target_distance <= 1.0)),
        ("over_1", target_distance > 1.0),
    )
    for name, mask in coverage_bins:
        metrics[f"target_distance_{name}_samples"] = int(mask.sum())
        metrics[f"target_distance_{name}_truth_mse"] = (
            float(np.mean(target_errors[mask] ** 2)) if mask.any() else None
        )
        metrics[f"target_distance_{name}_sparse_lt30"] = int(mask.sum()) < 30
    np.savez_compressed(
        artifact_dir / "pooled_data.npz", X=X, clean_y=clean_y,
        observed_y=observed_y, agent_ids=agent_ids, is_poisoned=poisoned,
        train_indices=splits["train"], validation_indices=splits["validation"],
        test_indices=splits["test"],
    )
    torch.save({
        "n_features": identity["dimension"],
        "agents": {agent.agent_id: agent.model.state_dict() for agent in agents},
    }, artifact_dir / "local_models.pt")
    torch.save({
        "n_features": identity["dimension"], "state_dict": evaluator.state_dict(),
        "evaluator_seed": evaluator_seed, "metrics": metrics,
    }, artifact_dir / "global_evaluator.pt")
    metadata = {
        "schema_version": 1, "scenario": identity,
        "evaluator_seed": evaluator_seed, "metrics": metrics,
        "interpretation": "Offline frozen evaluator; never used by selection, alpha update, or pruning.",
        "data_file": "pooled_data.npz", "local_models_file": "local_models.pt",
        "global_evaluator_file": "global_evaluator.pt",
        "agent_data_profile": [
            {
                "agent_id": int(row["agent_id"]),
                "sample_count": int(row.get("sample_count", len(row["X"]))),
                "feature_shift": float(row.get("feature_shift", 0.0)),
                "noise_std": float(row.get("noise_std", 0.05)),
                "evaluator_noise_std": float(
                    row.get("evaluator_noise_std", row.get("noise_std", 0.05))
                ),
                "is_poisoned": bool(row["is_poisoned"]),
            }
            for row in records
        ],
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    return evaluator, artifact_dir, metrics


def score_solution(evaluator, solution, target):
    prediction = float(_predict(evaluator, np.asarray(solution, dtype=np.float32)[None, :])[0])
    return prediction, abs(prediction - target)


def load_evaluator_artifact(artifact_dir):
    payload = torch.load(Path(artifact_dir) / "global_evaluator.pt", map_location="cpu",
                         weights_only=True)
    model = PretrainedModel(in_features=payload["n_features"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
