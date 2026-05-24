import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Canonical Matrix Factorization Backbone (Minimalist Version).

    Designed specifically for the Digital Twin Stress-Test (RQ4) and EPT Simulations.
    This architecture is deliberately stripped of Graph Convolutions (GNNs),
    deep MLPs, and Data Augmentation noises.

    Purpose: To completely eliminate structural inductive biases, ensuring that the
    observed gradient conflicts and symmetry breaking stem solely from the
    topological manifold (the dataset) and the optimization dynamics.
    """

    def __init__(self, num_users, num_items, embed_dim=64):
        """
        Args:
            num_users: Number of users in the topological skeleton.
            num_items: Number of items in the topological skeleton.
            embed_dim: The dimensionality of the latent space (default: 64 to save compute).
        """
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim

        # The canonical parameter space (acting as the physical manifold in our simulation)
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        # Strict normal initialization is crucial for dynamical stability in simulations.
        # Xavier can cause large initial variances, leading to immediate thermal blowups.
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def get_all_embeddings(self, graph=None):
        """
        Returns the pure zero-hop embeddings.
        Note: The 'graph' argument is kept solely for API compatibility with
        the data_loader, but is rigorously ignored to prevent GNN message passing.
        """
        return self.user_embedding.weight, self.item_embedding.weight

    def forward(self, users, pos_items, neg_items=None, return_embs=False, graph=None):
        """
        Standard forward pass via Dot Product (Canonical BPR-MF style).
        Included for completeness, though RQ4 directly computes via get_all_embeddings().
        """
        u_emb = self.user_embedding(users)
        pos_i_emb = self.item_embedding(pos_items)

        # Canonical inner product matching the physical simulation energy function
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=-1)

        neg_scores = None
        neg_i_emb = None
        if neg_items is not None:
            neg_i_emb = self.item_embedding(neg_items)
            neg_scores = torch.sum(u_emb * neg_i_emb, dim=-1)

        if return_embs:
            return pos_scores, neg_scores, u_emb, pos_i_emb, neg_i_emb
        else:
            return pos_scores, neg_scores

