# 📢 Notice for Reviewers
**Thank you for reviewing our paper! Please note the following repository structure:**

**1. Exact Paper Reproduction (Original Submission):**
> To reproduce the exact results, tables, and figures presented in the submitted PDF, you may refer to the original code in the ../EPT/ folder (under the Digital_Twin_stress-test_environment repository) and the ../IM-Net/RQ4/ folder (under the IM-Net repository).

>For a more integrated and improved version, all experimental content and code from EPT and RQ4 have been consolidated into Digital_Twin_Stress-Test-V1.1/ within the Digital_Twin_stress-test_environment repository.

**2. Post-Submission Improvements (Digital Twin Stress Test):**
> Science is an ongoing process! After the submission deadline, we have continued to refine our framework. We have recently introduced a highly robust Digital Twin Stress Test, which further validates the proactive nature of IM-Net under extreme gradient interference.
> 
> The new code for this stress test is located strictly in this current directory (`Digital_Twin_Stress-Test`).
> The updated, more comprehensive visualization results will be dynamically generated in the `results/` folder upon execution.
> These new results will be included in the appendix of our camera-ready version.

<br>

**Code Availability for:** *Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization*

---

# 📖 Overview
Welcome to the **Digital Twin Stress-Test Sandbox**.

To rigorously verify the core optimization logic independent of complex architectural inductive biases (such as deep Graph Neural Networks), we isolated our meta-controller (IM-Net) into this minimalist simulation environment. This sandbox acts as a "Canonical Coupled Oscillator", providing a controlled thermodynamical environment to observe how conflicting gradients interact on a pure neural manifold.

Here, we inject extreme boundary conditions (Forced Severe Conflict, $\rho \approx -0.99$) to push the system into a theoretical "worst-case" scenario, allowing us to explicitly observe:

1. **Dynamical Symmetry Breaking** under severe catastrophic interference.
2. **Anticipatory Control** via proactive meta-weighting.
3. **Loss Landscape Smoothing** validated through Hessian Spectral Regularization (Proposition 1).

## 🧬 Data Environment Design
To rigorously analyze the optimization dynamics of IM-Net without the confounding hardware bottlenecks of massive graphs, this Digital Twin environment employs two carefully designed dataset modalities:

* **`data_mini_yelp`** *(For RQ4 - Algorithmic Mechanism Verification)*:
  * **Role:** A miniature empirical sandbox.
  * **Purpose:** Serves as an unperturbed, down-sampled manifold of the real world. We use this in RQ4 to cleanly observe how IM-Net coordinates auxiliary and recommendation tasks under normal conditions, ensuring the algorithmic logic remains robust on smaller scales.

* **`yelp_sample_conflict`** *(For EPT - Emergent Phase Transition Simulation)*:
  * **Role:** A conflict-injected topological skeleton.
  * **Purpose:** To observe the "Symmetry Breaking" and Phase Transitions discussed in the paper, standard datasets rarely reach the required thermal instability naturally. Therefore, we sample a dense subgraph (`yelp_sample_conflict`) and pair it with our `conflict_injector.py` to artificially force extreme gradient misalignment. This acts as a true "stress test", proving that IM-Net can stabilize the system even under catastrophic optimization conditions.

## ⚙️ Design Rationale & Core Mechanisms
### 1. Simulation Intervention Mechanism (`conflict_injector.py`)
To empirically validate the Emergent Phase Transition (EPT) triggered by extreme gradient misalignments, we implemented an external intervention module. Unlike standard noisy data augmentation, our `ConflictInjector` explicitly forces the high-dimensional auxiliary gradient manifold to maintain a strictly controlled phase angle ($\rho \approx -1.0$) relative to the primary task. This geometrically guarantees a thermodynamically unstable environment for our baseline testing.

### 2. Why a minimalist NCF (Pure MF) backbone for RQ4?
In modern Recommender Systems, deep GNNs (like LightGCN) or non-linear MLPs introduce strong structural inductive biases and complex Hessian geometries. To strictly isolate and verify the meta-optimization mechanism of IM-Net, we deployed a canonical inner-product backbone (Pure MF) in our Digital Twin environment. By stripping away graph message passing and MLP non-linearities, we ensure that the phase transitions and gradient alignments observed in RQ4 are pure manifestations of the underlying optimization dynamics, free from structural artifacts.

## 🏗️ Sandbox Architecture
The repository is structured to separate "physical laws" from "topological manifolds" and "observation instruments":

```text
Digital_Twin_Stress-Test/
│
├── datasets/                   # 🌌 Topological Manifolds (Data structures)
│   ├── data_mini_yelp/         # Miniature empirical sandbox for RQ4 mechanisms
│   └── yelp_sample_conflict/   # Conflict-injected skeleton for Emergent Phase Transitions (EPT)
│
├── environment/                # ⚙️ Physical Laws of the Simulation
│   ├── backbone_ncf.py         # Stripped-down Neural Collaborative Filtering (Pure baseline)
│   ├── imnet_core.py           # Lightweight, pure-MLP Meta-Controller (IM-Net)
│   ├── conflict_injector.py    # Logic to enforce extreme adversarial gradients ($\rho \approx -0.99$)
│   └── data_loader.py          # Unified topological data pipeline
│
├── theory_utils/               # 🔭 Theoretical Observation Instruments (For Reviewers)
│   └── hessian_tracker.py      # Tracks spectral radius ($\lambda_{max}$) & optimization manifold curvature
│
├── run_rq4_mechanisms.py       # Execution script: RQ4 Dynamical Stress Test
├── run_ept_simulation.py       # Execution script: Emergent Phase Transitions (Anticipatory Control)
└── requirements.txt            # Minimal dependencies
```

## 🚀 Experiments & Reproduction

**Experiment I: Dynamical Mechanisms Stress Test (RQ4)**

**Goal:** Observe how static weighting succumbs to gradient cancellation (Deadlock) and how proactive control breaks this symmetry.

```bash
python run_rq4_mechanisms.py
```

**Output:** (`results/Fig_RQ4_Stress_Test_Complete.pdf`):

- **(a)** Topological Manifold Stability: System capability recovery.
- **(b)** Effective Geometric Alignment: The phase angle ($\rho$) rebounding from $-1.0$.
- **(c)** Proactive Kinetic Intervention: Active suppression of adversarial vectors.
- **(d)** Hessian Spectral Regularization: Real-time logging of the spectral radius ($\lambda_{max}$).

**Experiment II: Emergent Phase Transitions (EPT) Simulation**

**Goal:** Visualize Anticipatory Control, proving the meta-controller absorbs conflicts pre-emptively, smoothing the loss landscape.

```bash
python run_ept_simulation.py
```

**Output:** Automatically generates the theoretical figures from the paper:

- `results/Fig_5_Optimization_Dynamics.pdf` (NDCG & Interference Energy tracking).
- `results/Fig_6_Conflict_Alignment.pdf` (Energy vs. Cosine Similarity Phase diagram).
- `results/Fig_7_Landscape_Smoothing.pdf` (Demonstrating anticipatory up-weighting and sharp minima avoidance).

## 📐 Key Theoretical Metrics Explained

> For transparency, the mathematical instruments utilized in this sandbox are defined as follows:
> * **Interference Energy:** Defined as the magnitude of the auxiliary gradient projected onto the negative direction of the primary gradient.
> * **Hessian Spectral Radius ($\lambda_{max}$):** Approximated using power iteration (`theory_utils/hessian_tracker.py`). It represents the sharpest curvature of the loss landscape. A lower Hessian Spectral Radius indicates implicit regularization and landscape smoothing (Proposition 1).
> * **Effective Alignment ($\rho$):** Cosine similarity between the final aggregated gradient and the primary task gradient.

## 🛠️ Installation
> Designed to be ultra-lightweight and CPU-friendly (GPU recommended for faster Hessian computation).

```bash
conda create -n digital_twin python=3.10
conda activate digital_twin
pip install -r requirements.txt
```

> * (Only standard libraries: `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm` are required for strict cross-platform reproducibility.)*

> * Developed for rigorous theoretical verification. For the full-scale recommendation system experiments on LightGCN/SimGCL (Amazon/Yelp), please refer to the main engineering repository.*
