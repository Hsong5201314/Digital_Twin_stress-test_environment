import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# ================= Global style (top‑journal standard) =================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

os.makedirs('figures', exist_ok=True)

# ================= Load data =================
df_meta = pd.read_csv('simulation_logs/sim_metrics_yelp_meta.csv')
df_fixed = pd.read_csv('simulation_logs/sim_metrics_yelp_fixed_weights.csv')
res_meta = pd.read_csv('results_yelp_LightGCN_meta.csv')
res_fixed = pd.read_csv('results_yelp_LightGCN_fixed_weights.csv')

# Epoch‑level aggregation (first row of each epoch)
df_meta_epoch = df_meta.groupby('epoch').first().reset_index()
df_fixed_epoch = df_fixed.groupby('epoch').first().reset_index()

# Best performance metrics
best_meta_ndcg = res_meta['ndcg'].max()
best_fixed_ndcg = res_fixed['ndcg'].max()
best_meta_epoch = res_meta.loc[res_meta['ndcg'].idxmax(), 'epoch']
best_fixed_epoch = res_fixed.loc[res_fixed['ndcg'].idxmax(), 'epoch']
improvement = (best_meta_ndcg - best_fixed_ndcg) / best_fixed_ndcg * 100

# ================= Figure 5: Phase Transition Dynamics =================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(10, 4))

# 5a: NDCG@20 convergence curves
ax5a.plot(res_fixed['epoch'], res_fixed['ndcg'], color='#4C72B0', linestyle='--',
          linewidth=2, label='Static')
ax5a.plot(res_meta['epoch'], res_meta['ndcg'], color='#DD8452', linestyle='-',
          linewidth=2, label='IM-Net')
ax5a.scatter(best_fixed_epoch, best_fixed_ndcg, color='#4C72B0', s=50,
             edgecolor='black', zorder=5)
ax5a.scatter(best_meta_epoch, best_meta_ndcg, color='#DD8452', s=50,
             edgecolor='black', zorder=5)
ax5a.set_xlabel('Epoch')
ax5a.set_ylabel('NDCG@20')
ax5a.set_ylim(0.05, 0.08)
ax5a.grid(True, alpha=0.3, linestyle=':')
ax5a.legend()
ax5a.set_title('(a)', loc='left')
ax5a.text(0.6, 0.06, f'Improvement: {improvement:.1f}%', transform=ax5a.transAxes,
          fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 5b: Interference energy evolution (log scale) – alignment phase diagram
ax5b.plot(df_fixed_epoch['epoch'], df_fixed_epoch['interference_energy'],
          color='#4C72B0', linestyle='--', linewidth=2, label='Static')
ax5b.plot(df_meta_epoch['epoch'], df_meta_epoch['interference_energy'],
          color='#DD8452', linestyle='-', linewidth=2, label='IM-Net')
ax5b.set_xlabel('Epoch')
ax5b.set_ylabel('Interference Energy')
ax5b.set_yscale('log')
ax5b.grid(True, alpha=0.3, linestyle=':')
ax5b.legend()
ax5b.set_title('(b)', loc='left')

# Highlight phase transition region (Epoch 8–12)
ax5b.axvspan(8, 12, alpha=0.2, color='gray', label='Phase Transition')
ax5b.text(10, 1e-4, 'Conflict\nRegion', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('figures/Fig5_Phase_Transition.pdf')
plt.savefig('figures/Fig5_Phase_Transition.png')
plt.close()

# ================= Figure 6: Conflict & Alignment (Interference + Gradient Similarity) =================
fig6, ax6 = plt.subplots(figsize=(5, 3.5))
ax6_twin = ax6.twinx()

# Left axis: interference energy (linear, highlight the spike)
ax6.plot(df_meta_epoch['epoch'], df_meta_epoch['interference_energy'],
         color='#2C7FB8', linestyle='-', linewidth=2, label='Interference Energy')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Interference Energy (linear)', color='#2C7FB8')
ax6.tick_params(axis='y', labelcolor='#2C7FB8')

# Right axis: gradient cosine similarity
ax6_twin.plot(df_meta_epoch['epoch'], df_meta_epoch['grad_cos_sim'],
              color='#DD8452', linestyle='--', linewidth=2, label='Gradient Cosine Similarity')
ax6_twin.set_ylabel('Cosine Similarity', color='#DD8452')
ax6_twin.tick_params(axis='y', labelcolor='#DD8452')
ax6_twin.set_ylim(-1.0, 0.2)

# Mark the peak point (epoch 11)
peak_idx = df_meta_epoch['interference_energy'].idxmax()
peak_epoch = df_meta_epoch.loc[peak_idx, 'epoch']
peak_interf = df_meta_epoch.loc[peak_idx, 'interference_energy']
peak_cos = df_meta_epoch.loc[peak_idx, 'grad_cos_sim']
ax6.scatter(peak_epoch, peak_interf, color='#2C7FB8', s=50, zorder=5)
ax6_twin.scatter(peak_epoch, peak_cos, color='#DD8452', s=50, zorder=5)

# Annotate with relative offset
ax6.annotate(f'Epoch {peak_epoch}',
             xy=(peak_epoch, peak_interf),
             xytext=(15, -15),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=8, ha='left', va='top')

ax6.grid(True, alpha=0.3, linestyle=':')

# Combine legends
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('figures/Fig6_Conflict_Alignment.pdf')
plt.savefig('figures/Fig6_Conflict_Alignment.png')
plt.close()

# ================= Figure 7: Optimization Stability & Landscape Geometry =================
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(10, 4))

# 7a: Training loss curves (static vs. IM-Net)
ax7a.plot(res_fixed['epoch'], res_fixed['loss'], color='#4C72B0', linestyle='--',
          linewidth=2, label='Static')
ax7a.plot(res_meta['epoch'], res_meta['loss'], color='#DD8452', linestyle='-',
          linewidth=2, label='IM-Net')
ax7a.set_xlabel('Epoch')
ax7a.set_ylabel('Training Loss')
ax7a.set_yscale('log')
ax7a.grid(True, alpha=0.3, linestyle=':')
ax7a.legend()
ax7a.set_title('(a)', loc='left')

# Annotate static loss spike
peak_loss_epoch = res_fixed.loc[res_fixed['loss'].idxmax(), 'epoch']
ax7a.annotate('Loss spike', xy=(peak_loss_epoch, res_fixed['loss'].max()),
              xytext=(peak_loss_epoch - 15, 0.5),
              arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

# 7b: Adaptive auxiliary weight and interference energy (dual axis)
ax7b_twin = ax7b.twinx()

ax7b.plot(df_meta_epoch['epoch'], df_meta_epoch['w_aux'],
          color='#2C7FB8', linestyle='-', linewidth=2.5,
          label='$w_{\\mathrm{aux}}$ (IM-Net)')
ax7b_twin.plot(df_meta_epoch['epoch'], df_meta_epoch['interference_energy'],
               color='#DD8452', linestyle=':', linewidth=2.5,
               label='Interference (IM-Net)')

ax7b.set_xlabel('Epoch')
ax7b.set_ylabel('Auxiliary Weight', color='#2C7FB8')
ax7b_twin.set_ylabel('Interference Energy', color='#DD8452')
ax7b_twin.set_yscale('log')

ax7b.legend(loc='upper left', frameon=False)
ax7b_twin.legend(loc='upper right', frameon=False)

# Text annotation for final auxiliary weight
final_waux = df_meta_epoch['w_aux'].iloc[-1]
ax7b.annotate(f'$w_{{\\mathrm{{aux}}}} \\rightarrow {final_waux:.3f}$',
              xy=(80, final_waux), xytext=(80, final_waux - 0.005),
              fontsize=9, ha='center', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

# Adjust axis limits to leave room for annotation
ax7b.set_ylim(0.08, 0.20)
ax7b_twin.set_ylim(1e-6, 1e-4)
ax7b.set_title('(b)', loc='left')

plt.tight_layout()
plt.savefig('figures/Fig7_Landscape_Smoothing.pdf')
plt.savefig('figures/Fig7_Landscape_Smoothing.png')
plt.close()

print("All figures (Fig5, Fig6, Fig7) have been generated in the 'figures/' directory.")
