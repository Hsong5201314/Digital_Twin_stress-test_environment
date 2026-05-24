"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization
File: run_rq4_stress_test.py
Description:
    Digital Twin Stress-Test Environment (RQ4).
    Simulates a Canonical Coupled Oscillator system on the pure NCF backbone to verify
    dynamical symmetry breaking, proactive meta-weighting, and spectral regularization.
Date: 2026-05-16
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# ⭐️ Import lightweight components exclusive to the twin environment
from environment.data_loader import DataProcessor
from environment.backbone_ncf import NCF
from environment.imnet_core import IMNet

# ⭐️Import theoretical observation instruments (verify Proposition 1)
from theory_utils.hessian_tracker import SpectralDynamicsTracker


def set_seed(seed=2026):
    """Solidify initial random state of physical sandbox (initial thermodynamic microstate)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_flat_grads(loss, model, retain_graph=True):
    """Compute flattened gradient of a single loss (tangent vector of high-dimensional manifold)"""
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=retain_graph, allow_unused=True)
    flat_grads = [g.view(-1) for g in grads if g is not None]
    if not flat_grads:
        return torch.zeros(1, device=loss.device)
    return torch.cat(flat_grads)


@torch.no_grad()
def compute_gradient_alignment(loss1, loss2, model, retain_graph=True):
    """Calculate cosine phase angle (\(\rho\)) between tangent vectors"""
    g1 = get_flat_grads(loss1, model, retain_graph=retain_graph)
    g2 = get_flat_grads(loss2, model, retain_graph=retain_graph)
    return F.cosine_similarity(g1, g2, dim=0).item()


@torch.no_grad()
def evaluate_manifold_stability(model, test_loader, device):
    """
    [Lightweight Evaluation for Physical Simulation]
    Calculate Manifold AUC (pos_score > neg_score ratio only, no full sorting)
    100x faster, ideal for dynamic evolution observation
    """
    model.eval()
    correct = 0
    total = 0
    for batch in test_loader:
        users, pos_items, neg_items = [b.to(device).view(-1) for b in batch]
        u_emb, i_emb = model.get_all_embeddings()

        pos_scores = torch.sum(u_emb[users] * i_emb[pos_items], dim=1)
        neg_scores = torch.sum(u_emb[users] * i_emb[neg_items], dim=1)

        correct += (pos_scores > neg_scores).sum().item()
        total += len(users)

    return correct / (total + 1e-8)


def run_stress_test(use_full_dataset=False):
    print("=" * 85)
    print("🔬 Starting RQ4: Digital Twin Controlled Stress Test on Manifold Optimization")
    print("=" * 85)

    set_seed(2026)  # Lock random force field
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Warning] Thermodynamics simulation running on CPU – expected slow phase evolution.")

    # Force pointing to the sandbox dataset
    dataset_name = 'datasets/data_mini_yelp' if not use_full_dataset else 'datasets/yelp_processed_for_meta'
    max_epochs = 200 if not use_full_dataset else 500
    print(f"[Environment] Loaded Skeleton: {dataset_name} | Max Epochs: {max_epochs}")

    dp = DataProcessor(dataset_name)

    # ========== Kinetic Test Configuration ==========
    config = {
        'lr': 0.001,
        'meta_lr': 0.005,
        'embed_dim': 64,
        'hvp_eps': 1e-2,
        'hvp_max_eps': 0.01,
        'meta_loss_weight': 50.0,
    }

    variants = ['scalarization', 'meta']
    variant_names = {'scalarization': 'Reactive Baseline', 'meta': 'IM-Net (Proactive)'}
    results = []

    # Extract meta-validation state (Meta-Validation State)
    meta_val_batches = []
    val_batch_size = 16
    for i, batch in enumerate(dp.train_loader):
        if i >= val_batch_size: break
        meta_val_batches.append(batch)

    for variant in variants:
        print(f">>>>> Initiating Phase Evolution for: {variant_names[variant]} <<<<<")
        model = NCF(dp.n_users, dp.n_items, config['embed_dim']).to(device)
        optimizer_model = torch.optim.Adam(model.parameters(), lr=config['lr'])

        if variant == 'meta':
            # Note: If your core IMNet takes 5-dimensional input, pass a 5D tensor here as instructed earlier.
            # This assumes you use the dual-task simplified IMNet (num_tasks=2) for the sandbox.
            imnet = IMNet(num_tasks=2).to(device)
            optimizer_meta = torch.optim.Adam(imnet.parameters(), lr=config['meta_lr'])

        # ⭐️ Initialize Hessian spectral radius tracker (verify Theorem 1)
        tracker = SpectralDynamicsTracker(num_iter=5, tolerance=1e-6)

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_alignments = []
            epoch_effective_align = []
            epoch_aux_weights = []

            pbar = tqdm(dp.train_loader, desc=f"Epoch {epoch} [{variant_names[variant]}]", leave=False)
            for batch in pbar:
                users, pos_items, neg_items = [b.to(device).view(-1) for b in batch]
                u_emb, i_emb = model.get_all_embeddings()

                pos_scores = torch.sum(u_emb[users] * i_emb[pos_items], dim=1)
                neg_scores = torch.sum(u_emb[users] * i_emb[neg_items], dim=1)

                main_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                conflict_loss = -F.logsigmoid(neg_scores - pos_scores).mean()  # Injected conflict

                if variant == 'scalarization':
                    alignment = compute_gradient_alignment(main_loss, conflict_loss, model, retain_graph=True)
                    epoch_alignments.append(alignment)

                    final_loss = main_loss + conflict_loss
                    optimizer_model.zero_grad()
                    final_loss.backward()
                    optimizer_model.step()

                    aux_weight = 1.0
                    effective_alignment = alignment

                elif variant == 'meta':
                    meta_input = torch.stack([main_loss.detach(), conflict_loss.detach()])
                    raw_weights, weights = imnet(meta_input)
                    weights = torch.sigmoid(weights) * 0.99 + 0.005

                    model_loss = weights[0] * main_loss + weights[1] * conflict_loss

                    # 对齐监控
                    model_grad = get_flat_grads(model_loss, model, retain_graph=True)
                    main_grad = get_flat_grads(main_loss, model, retain_graph=True)
                    effective_alignment = F.cosine_similarity(model_grad, main_grad, dim=0).item()
                    epoch_effective_align.append(effective_alignment)

                    raw_alignment = compute_gradient_alignment(weights[0] * main_loss, weights[1] * conflict_loss,
                                                               model,
                                                               retain_graph=True)
                    epoch_alignments.append(raw_alignment)

                    # ---------- 前瞻式虚拟位移 (Look-ahead Virtual Step) ----------
                    saved_params = [p.data.clone() for p in model.parameters()]
                    train_grads = torch.autograd.grad(model_loss, model.parameters(), retain_graph=True,
                                                      allow_unused=True)
                    train_grads = [g if g is not None else torch.zeros_like(p) for g, p in
                                   zip(train_grads, model.parameters())]

                    grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in train_grads]), 2)
                    safe_eps = torch.clamp(config['hvp_eps'] / (grad_norm + 1e-8), min=1e-6, max=config['hvp_max_eps'])

                    with torch.no_grad():
                        for param, g in zip(model.parameters(), train_grads):
                            param.add_(safe_eps * g)

                    # 探测虚拟态能量 (Validation Energy After Step)
                    v_main_after = sum(-F.logsigmoid(
                        torch.sum(model.get_all_embeddings()[0][b[0].to(device)] * model.get_all_embeddings()[1][
                            b[1].to(device)], dim=1) -
                        torch.sum(model.get_all_embeddings()[0][b[0].to(device)] * model.get_all_embeddings()[1][
                            b[2].to(device)], dim=1)
                    ).mean() for b in meta_val_batches) / len(meta_val_batches)

                    # Roll back physical state
                    with torch.no_grad():
                        for param, saved in zip(model.parameters(), saved_params):
                            param.data = saved

                    v_main_before = sum(-F.logsigmoid(
                        torch.sum(model.get_all_embeddings()[0][b[0].to(device)] * model.get_all_embeddings()[1][
                            b[1].to(device)], dim=1) -
                        torch.sum(model.get_all_embeddings()[0][b[0].to(device)] * model.get_all_embeddings()[1][
                            b[2].to(device)], dim=1)
                    ).mean() for b in meta_val_batches) / len(meta_val_batches)

                    # Meta loss backward
                    rel_improvement = (v_main_before - v_main_after) / (v_main_before + 1e-8)
                    meta_loss = -rel_improvement

                    dynamic_lambda = config['meta_loss_weight'] * (1 + epoch / 100)
                    weight_penalty = weights[1] + weights[1] ** 2
                    final_loss = model_loss + dynamic_lambda * (meta_loss + weight_penalty)

                    optimizer_model.zero_grad()
                    optimizer_meta.zero_grad()
                    final_loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(imnet.parameters(), max_norm=1.0)

                    optimizer_model.step()
                    optimizer_meta.step()

                    aux_weight = weights[1].item()
                    epoch_aux_weights.append(aux_weight)

                pbar.set_postfix({'ρ_eff': f"{effective_alignment:.2f}", 'W_harm': f"{aux_weight:.2f}"})

            # ========== ⭐️ Theoretical Verification: Probe Optimization Manifold Curvature (Hessian Spectral Radius) ==========
            model.train()
            # Use the first batch to compute representative curvature, greatly saving computation time
            b_users, b_pos, b_neg = [b.to(device).view(-1) for b in meta_val_batches[0]]
            b_u, b_i = model.get_all_embeddings()

            rep_loss = -F.logsigmoid(
                torch.sum(b_u[b_users] * b_i[b_pos], dim=1) -
                torch.sum(b_u[b_users] * b_i[b_neg], dim=1)
            ).mean()

            lambda_max = tracker.compute_spectral_radius(model, rep_loss)

            # ========== Evaluation and Logging ==========
            # Adopt ultra-lightweight Manifold AUC evaluation tailored for physical engines!
            manifold_auc = evaluate_manifold_stability(model, dp.test_loader, device)

            avg_alignment = np.mean(epoch_alignments) if epoch_alignments else 0.0
            if variant == 'scalarization':
                avg_effective = avg_alignment
                avg_aux_weight = 1.0
            else:
                avg_effective = np.mean(epoch_effective_align) if epoch_effective_align else 0.0
                avg_aux_weight = np.mean(epoch_aux_weights) if epoch_aux_weights else 0.0

            print(
                f"Epoch {epoch:02d} | Stability(AUC): {manifold_auc:.4f} | Eff ρ: {avg_effective:.3f} | AuxW: {avg_aux_weight:.3f} | Curvature(λ_max): {lambda_max:.4f}")

            results.append({
                'Method': variant_names[variant],
                'Epoch': epoch,
                'Manifold_AUC': manifold_auc,
                'GradientAlignment': avg_alignment,
                'EffectiveAlignment': avg_effective,
                'AuxWeight': avg_aux_weight,
                'SpectralRadius': lambda_max
            })

    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/rq4_stress_test_data.csv', index=False)
    plot_stress_test_figures(df)


def plot_stress_test_figures(df):
    """
    Generate side-by-side plots of the four core metrics.
    Intuitively illustrate the complete logical chain ranging from macroscopic stability,
    alignment degree and intervention weight down to physical manifold curvature (Hessian).
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.5))
    colors = {'Reactive Baseline': '#B0BEC5', 'IM-Net (Proactive)': '#E53935'}

    # (a) Topological Manifold Stability
    sns.lineplot(data=df, x='Epoch', y='Manifold_AUC', hue='Method', palette=colors, lw=2.5, ax=axes[0], legend=False)
    axes[0].set_title('a. Topological Manifold Stability', fontweight='bold')
    axes[0].set_ylabel('Manifold AUC (Pos > Neg %)', fontweight='bold')
    axes[0].set_ylim(0.4, 1.0)

    # (b) Effective Gradient Alignment Trajectory
    align_col = 'EffectiveAlignment' if 'EffectiveAlignment' in df.columns else 'GradientAlignment'
    ylabel = r'Phase Angle $\rho(g_{model}, g_{main})$'
    sns.lineplot(data=df, x='Epoch', y=align_col, hue='Method', palette=colors, lw=2.5, ax=axes[1])
    axes[1].set_title('b. Effective Geometric Alignment', fontweight='bold')
    axes[1].set_ylabel(ylabel, fontweight='bold')
    axes[1].axhline(0.0, color='grey', linestyle='--', alpha=0.5)
    axes[1].axhline(-1.0, color='lightgrey', linestyle=':', alpha=0.5)
    axes[1].legend(loc='lower right')

    # (c) Harmful Task Weight Trajectory
    df_imnet = df[df['Method'] == 'IM-Net (Proactive)']
    if not df_imnet.empty:
        axes[2].plot(df_imnet['Epoch'], df_imnet['AuxWeight'], color=colors['IM-Net (Proactive)'], lw=2.5)
        axes[2].set_title('c. Proactive Kinetic Intervention', fontweight='bold')
        axes[2].set_ylabel('Weight of Adversarial Vector', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].axhline(0.01, color='black', linestyle=':', alpha=0.7, label='Lower limit')
        axes[2].legend(loc='upper right')

    # (d) ⭐️ Optimization Manifold Curvature (Hessian Spectral Radius)
    sns.lineplot(data=df, x='Epoch', y='SpectralRadius', hue='Method', palette=colors, lw=2.5, ax=axes[3], legend=False)
    axes[3].set_title('d. Hessian Spectral Regularization (Prop. 1)', fontweight='bold')
    axes[3].set_ylabel(r'Spectral Radius ($\lambda_{max}$)', fontweight='bold')
    axes[3].set_yscale('log')
    axes[3].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('results/Fig_RQ4_Stress_Test_Complete.png', dpi=300)
    plt.savefig('results/Fig_RQ4_Stress_Test_Complete.pdf', bbox_inches='tight')
    print("[INFO] Comprehensive 1x4 figures saved to results/Fig_RQ4_Stress_Test_Complete.png")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Digital Twin Stress Test for Multi-Objective Dynamics")
    parser.add_argument('--full', action='store_true', help='Use full Yelp dataset')
    args = parser.parse_args()
    run_stress_test(use_full_dataset=args.full)