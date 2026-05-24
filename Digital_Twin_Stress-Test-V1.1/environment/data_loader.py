"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization
File: data_loader.py
Description:
    Topological Skeleton Loader for the Digital Twin simulation environment.
    Designed exclusively to load the fundamental bipartite graph (User-Item interactions)
    for the primary objective (BPR Loss).

    Note: Adversarial topological structures (e.g., 'conflict_edges.txt') and
    phase angle forcing mechanisms are strictly handled by 'conflict_injector.py'
    to maintain absolute separation between the natural physical manifold and
    artificial external interventions.
Date: 2026-05-16
"""

import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Set


class RecDataset(Dataset):
    """
    Standard Bipartite Graph Dataset with Uniform Negative Sampling.
    Forms the baseline topological manifold for the primary recommendation task.
    """

    def __init__(self, data: List[List[int]], item_dict: Dict[int, Set[int]], num_items: int, neg_count: int = 10):
        self.data = data
        self.item_dict = item_dict
        self.num_items = num_items
        self.neg_count = neg_count

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, int, List[int]]:
        user, pos_item = self.data[idx][0], self.data[idx][1]
        neg_items = []

        # Fast stochastic negative sampling
        while len(neg_items) < self.neg_count:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in self.item_dict.get(user, set()):
                neg_items.append(neg_item)

        return user, pos_item, neg_items


def collate_fn(batch: List[Tuple[int, int, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flattens the multiple negative samples into a 1D tensor format optimized
    for fast inner-product computations in the pure NCF backbone.
    """
    users, pos_items, neg_items_list = zip(*batch)
    expanded_users = []
    expanded_pos = []
    expanded_neg = []

    for u, pos, negs in zip(users, pos_items, neg_items_list):
        for neg in negs:
            expanded_users.append(u)
            expanded_pos.append(pos)
            expanded_neg.append(neg)

    return torch.LongTensor(expanded_users), torch.LongTensor(expanded_pos), torch.LongTensor(expanded_neg)


class DataProcessor:
    """
    Central Data Processor for Empirical Sandboxes ('data_mini_yelp' & 'yelp_sample_conflict').
    Strips away heavy engineering overhead (like GCN adj-matrix builds) to focus
    purely on providing a clean optimization manifold.
    """

    def __init__(self, data_path: str, batch_size: int = 2048, train_neg_count: int = 10):
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_neg_count = train_neg_count

        print(f"[*] Initializing Topological Skeleton from: {self.data_path}")

        # 1. Load topological interactions
        self.train_data = self._load_interactions(os.path.join(self.data_path, "train.txt"))
        self.test_data = self._load_interactions(os.path.join(self.data_path, "test.txt"))

        # Load or construct the Meta-Validation Set (Crucial for IM-Net's Look-ahead)
        meta_val_file = os.path.join(self.data_path, "meta_val.txt")
        if os.path.exists(meta_val_file):
            self.meta_val_data = self._load_interactions(meta_val_file)
            print(f"    - Loaded Explicit Meta-Validation Set: {len(self.meta_val_data)} edges.")
        else:
            # Fallback: Proxy Meta-Val partitioning if file doesn't exist
            # Slices the last 10% of training data as the meta-validation manifold
            split_idx = int(len(self.train_data) * 0.9)
            self.meta_val_data = self.train_data[split_idx:]
            self.train_data = self.train_data[:split_idx]
            print(f"    - [Fallback] Partitioned Proxy Meta-Validation Set: {len(self.meta_val_data)} edges.")

        # 2. Ascertain Global Graph Dimensions
        all_data = self.train_data + self.test_data + self.meta_val_data
        self.n_users, self.n_items = self._get_counts(all_data)
        print(f"    - Global Graph Boundaries -> Users: {self.n_users}, Items: {self.n_items}")

        # 3. Construct exclusion dictionaries for valid negative sampling
        self.train_dict = self._build_user_item_dict(self.train_data, as_set=True)
        self.test_dict = self._build_user_item_dict(self.test_data, as_set=False)

        # 4. Instantiate PyTorch DataLoaders
        self.train_loader = DataLoader(
            RecDataset(self.train_data, self.train_dict, self.n_items, neg_count=self.train_neg_count),
            batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn
        )

        self.meta_val_loader = DataLoader(
            RecDataset(self.meta_val_data, self.train_dict, self.n_items, neg_count=1),
            batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True, collate_fn=collate_fn
        )

        self.test_loader = DataLoader(
            RecDataset(self.test_data, self.train_dict, self.n_items, neg_count=1),
            batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn
        )

    def _load_interactions(self, file_path: str) -> List[List[int]]:
        """Reads bipartite edges from a text file."""
        data = []
        if not os.path.exists(file_path):
            return data

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append([int(parts[0]), int(parts[1])])
        return data

    def _get_counts(self, data: List[List[int]]) -> Tuple[int, int]:
        """Calculates strict upper bounds for embedding matrices to prevent index overflow."""
        if not data:
            return 0, 0
        arr = np.array(data)
        n_users = arr[:, 0].max() + 1
        n_items = arr[:, 1].max() + 1
        return int(n_users), int(n_items)

    def _build_user_item_dict(self, data: List[List[int]], as_set: bool = True):
        """Constructs adjacency mappings for O(1) sampling verification."""
        interaction_dict = defaultdict(set if as_set else list)
        for u, i in data:
            if as_set:
                interaction_dict[u].add(i)
            else:
                interaction_dict[u].append(i)
        return dict(interaction_dict)
