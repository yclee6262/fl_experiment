import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def generate_agent_datasets(num_agents=20, n_samples_per_agent=500,
                            poison_ratio=0.4, n_features=2, seed=42,
                            sample_counts=None, feature_shifts=None,
                            noise_stds=None):
    """Return deterministic observed and clean synthetic data for every agent.

    ``observed_y`` is what the local agent receives. ``clean_y`` preserves the
    benign counterfactual needed to build an offline study evaluator without
    training that evaluator on poisoned labels.
    """
    np.random.seed(seed)
    records = []
    num_poisoned = int(num_agents * poison_ratio)
    sample_counts = (
        [int(n_samples_per_agent)] * num_agents
        if sample_counts is None else [int(value) for value in sample_counts]
    )
    feature_shifts = (
        [0.0] * num_agents
        if feature_shifts is None else [float(value) for value in feature_shifts]
    )
    noise_stds = (
        [0.05] * num_agents
        if noise_stds is None else [float(value) for value in noise_stds]
    )
    if not all(len(values) == num_agents for values in
               (sample_counts, feature_shifts, noise_stds)):
        raise ValueError("Per-agent controls must match num_agents.")
    if any(value < 3 for value in sample_counts):
        raise ValueError("Every agent requires at least three samples.")
    if any(abs(value) >= 1 for value in feature_shifts):
        raise ValueError("Feature shifts must be strictly between -1 and 1.")
    if any(value < 0 for value in noise_stds):
        raise ValueError("Noise standard deviations must be non-negative.")
    for i in range(num_agents):
        half_width = 1.0 - abs(feature_shifts[i])
        lower = feature_shifts[i] - half_width
        upper = feature_shifts[i] + half_width
        X = np.random.uniform(lower, upper, (sample_counts[i], n_features))
        clean_y = np.sum(X, axis=1)
        if n_features > 1:
            clean_y += np.sum(X[:, :-1] * X[:, 1:], axis=1)
        noise = np.random.normal(0, noise_stds[i], sample_counts[i])
        clean_y = clean_y + noise
        poisoned = i >= num_agents - num_poisoned
        if poisoned:
            observed_y = -3 * np.sum(X, axis=1)
            if n_features > 1:
                observed_y += 2 * np.sum(X[:, :-1] ** 2, axis=1)
            observed_y += noise
        else:
            observed_y = clean_y.copy()
        records.append({
            "agent_id": i + 1,
            "X": X,
            "clean_y": clean_y,
            "observed_y": observed_y,
            "is_poisoned": poisoned,
            "sample_count": sample_counts[i],
            "feature_shift": feature_shifts[i],
            "noise_std": noise_stds[i],
            "evaluator_noise_std": noise_stds[i],
        })
    return records

def generate_agent_dataloaders(num_agents=20, n_samples_per_agent=500,
                               poison_ratio=0.4, n_features=2, seed=42,
                               sample_counts=None, feature_shifts=None,
                               noise_stds=None):
    """
    生成 Agent 訓練用的模擬數據
    ⭐ 加入 n_features 參數，並動態切換物理公式
    """
    loaders = []
    
    print(f"\n[環境設定] 決策變數維度: {n_features}D")
    print(f"[真實函數] y = sum(x_i) + sum(x_i * x_i+1)")
    print("-" * 50)
    
    records = generate_agent_datasets(
        num_agents=num_agents,
        n_samples_per_agent=n_samples_per_agent,
        poison_ratio=poison_ratio,
        n_features=n_features,
        seed=seed,
        sample_counts=sample_counts,
        feature_shifts=feature_shifts,
        noise_stds=noise_stds,
    )
    for record in records:
        dataset = TabularDataset(record["X"], record["observed_y"])
        loaders.append(DataLoader(dataset, batch_size=32, shuffle=True))
        
    return loaders
