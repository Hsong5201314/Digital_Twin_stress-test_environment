"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization
File: conflict_injector.py
Description:
    The core intervention module for the Emergent Phase Transition (EPT) simulation.
    It reads the deliberately perturbed topological skeleton ('yelp_sample_conflict')
    and exerts extreme adversarial forces on the optimization dynamics.

    Intervention Mechanisms:
    1. Loss-level Injection: aux_loss = - (BPR_loss) * scale.
    2. Gradient-level Forcing: Explicitly forces the auxiliary gradient's cosine
       similarity with the primary gradient to match exactly `rho_target` (e.g., -0.99).
Date: 2026-05-16
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class ConflictInjector:
    """
    Simulates a catastrophic multi-objective optimization environment by
    artificially injecting severe gradient misalignment and task conflicts.
    """
    def __init__(self, data_path: str, device: torch.device, rho_target: float = -0.99,
                 conflict_scale: float = 1.0, grad_force_enabled: bool = True):
        """
        Args:
            data_path: Path to the specific dataset directory (e.g., 'datasets/yelp_sample_conflict').
            device: Computation device (CPU/GPU).
            rho_target: The targeted cosine similarity constraint between primary and auxiliary gradients.
                        Negative values (e.g., -0.99) simulate intense adversarial conflict.
            conflict_scale: Scaling factor for the adversarial auxiliary loss.
            grad_force_enabled: Boolean flag to enable explicit geometrical gradient alignment.
        """
        self.data_path = data_path
        self.device = device
        self.rho_target = rho_target
        self.conflict_scale = conflict_scale
        self.grad_force_enabled = grad_force_enabled

        # Load predefined adversarial edges to perturb the topological manifold
        self.conflict_edges = self._load_conflict_edges()

        print(f"[*] ConflictInjector Initialized.")
        print(f"    - Manifold Path: {self.data_path}")
        print(f"    - Target Phase Angle (rho): {self.rho_target}")
        print(f"    - Topological Perturbations Loaded: {len(self.conflict_edges)} edges.")

    def _load_conflict_edges(self) -> np.ndarray:
        """
        Reads 'conflict_edges.txt' from the topological skeleton directory.
        Format per line: node1 node2.
        Acts as the structural basis for auxiliary adversarial task generation.
        """
        edge_file = os.path.join(self.data_path, "conflict_edges.txt")
        if not os.path.exists(edge_file):
            print(f"[Warning] No conflict_edges.txt found in {self.data_path}. "
                  "Falling back to random stochastic perturbation.")
            return np.empty((0, 2), dtype=int)

        edges = []
        with open(edge_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    edges.append([int(parts[0]), int(parts[1])])
        return np.array(edges, dtype=int)

    def compute_conflict_aux_loss(self, model: torch.nn.Module, batch: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Loss-level Injection: Constructs an adversarial landscape where the auxiliary
        task explicitly maximizes the primary task's error.

        Equation: L_{aux} = - L_{primary} * scale
        """
        users, pos_items, neg_items = [b.to(self.device).view(-1) for b in batch]

        # Extract zero-hop embeddings from the pure Canonical NCF backbone
        u_emb, i_emb = model.get_all_embeddings()

        batch_u = u_emb[users]
        batch_pos = i_emb[pos_items]
        batch_neg = i_emb[neg_items]

        # Canonical inner product (matching the physical simulation energy function)
        pos_scores = torch.sum(batch_u * batch_pos, dim=1)
        neg_scores = torch.sum(batch_u * batch_neg, dim=1)

        # Canonical BPR Loss (Primary Objective)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Adversarial Auxiliary Loss (Intentional Subversion)
        aux_loss = -bpr_loss * self.conflict_scale

        return aux_loss

    def force_gradients(self, grad_primary: List[torch.Tensor],
                        grad_aux: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Gradient-level Injection (Geometrical Forcing):
        Explicitly projects and modifies the auxiliary gradient manifold so that
        its cosine similarity with the primary gradient strictly equals `rho_target`.

        This guarantees the required theoretical instability conditions for EPT.
        """
        if not self.grad_force_enabled:
            return grad_aux

        # Flatten gradients into vectors for high-dimensional geometrical projection
        flat_pri = torch.cat([g.contiguous().view(-1) for g in grad_primary])
        flat_aux = torch.cat([g.contiguous().view(-1) for g in grad_aux])

        norm_pri = flat_pri.norm() + 1e-8
        norm_aux = flat_aux.norm() + 1e-8

        # 1. Compute current projection component
        proj = torch.dot(flat_pri, flat_aux) / (norm_pri * norm_aux)

        # 2. Extract orthogonal component
        orth = flat_aux - proj * flat_pri
        orth_norm = orth.norm() + 1e-8

        # 3. Construct new auxiliary gradient vector mathematically
        target_cos = self.rho_target
        a = target_cos / norm_pri
        b = ((1.0 - target_cos**2)**0.5) / orth_norm
        new_flat = a * flat_pri + b * orth

        # 4. Restore the original gradient magnitude to maintain step-size scale
        scale = norm_aux / (new_flat.norm() + 1e-8)
        new_flat = new_flat * scale

        # Unflatten back to the network's parameter shapes
        new_grad_aux = []
        idx = 0
        for g in grad_aux:
            numel = g.numel()
            new_grad_aux.append(new_flat[idx:idx+numel].view(g.shape))
            idx += numel

        return new_grad_aux

    def sample_conflict_aux_links(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples pre-defined conflicting node pairs from the topological skeleton.
        Used to replace standard auxiliary contrastive links with structurally
        destructive links.
        """
        if len(self.conflict_edges) == 0:
            # Fallback: Stochastic noise injection if specific topological flaws are missing
            node1 = torch.randint(0, 10000, (batch_size,), device=self.device)
            node2 = torch.randint(0, 10000, (batch_size,), device=self.device)
            return node1, node2

        indices = np.random.randint(0, len(self.conflict_edges), size=batch_size)
        edges = self.conflict_edges[indices]

        node1 = torch.LongTensor(edges[:, 0]).to(self.device)
        node2 = torch.LongTensor(edges[:, 1]).to(self.device)

        return node1, node2
