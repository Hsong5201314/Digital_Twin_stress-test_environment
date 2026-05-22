# Digital Twin Stress-Test Environment: IM-Net Optimization Dynamics

This repository contains the early-version release of the **Digital Twin Stress-Test Environment**, designed to rigorously verify the optimization logic and anticipatory control mechanisms of **IM-Net**.

By utilizing a Canonical Coupled Oscillator Simulation environment based on a LightGCN backbone and the yelp_sample_conflict dataset, this setup serves as a proxy for complex physical systems governed by competing constraints (e.g., conflicting forces in molecular dynamics or fluid simulations).

## 🔬 Theoretical Background

To evaluate the meta-controller independent of architectural inductive biases, we establish a "Severe Conflict" boundary condition. In this stress-test, the auxiliary gradient is mathematically forced to directly oppose the primary task gradient (ρ≈−0.99). This establishes a theoretical "worst-case" dynamical system.

Under these conditions, we observe distinct emergent phase transitions:

> Phase I (Exploration, Epochs 0–8): Conflict is disabled (λ=0) for warm-up. IM-Net assigns initial non-zero auxiliary weights (0.09o0.13), allowing shared feature exploration with near-zero interference energy.

> Phase II (Stabilization, Epochs 9–100): The severe conflict (ρ≈−0.99) is triggered. While static weighting suffers from destructive interference (peaks up to 1.6imes10 −4) and loss spikes, IM-Net demonstrates anticipatory control. It permits a transient interference peak around Epoch 11, rapidly dissipates it, and continues to up-weight the auxiliary task (0.13o0.16), extracting synergies while keeping interference bounded (<4imes10 −5).

Key Outcomes: IM-Net achieves a superior NDCG@20 of 0.0711 (a 10.7% relative improvement over the static weighting baseline of 0.0644) and a smooth, monotonic loss decrease.


## 📂 Repository Structure

The simulation is modularized into the following core components:

> main_gpu.py: The main entry point for running the simulation. Initializes the dataset, model, and training loop.

> data_loaderMeta.py: Handles the loading and specialized processing of the yelp_sample_conflict dataset, setting up the graph structures and task splits.

> backbone.py: Implementation of the canonical LightGCN model. It acts as the physical substrate (coupled oscillators) for the optimization dynamics.

> imnet.py: The core meta-controller network. Implements the anticipatory weight adjustment logic.

> train_engine.py: The training loop where the "Severe Conflict" mathematical boundary conditions are enforced. It explicitly tracks interference energy, gradient cosine similarity, and auxiliary weights.

> utils.py: Utility functions for logging, metric calculation (NDCG, Recall), and visualization of the optimization trajectory.

## ⚙️ Installation & Requirements

Ensure you have a modern PyTorch environment (GPU recommended).

```bash
# Clone the repository
git clone <repository_url>
cd Digital_Twin_stress-test_environment

# Install dependencies (example)
pip install torch pandas numpy matplotlib seaborn scipy
```

## 🚀 How to Run the Simulation

To execute the digital twin stress-test and reproduce the optimization dynamics:

```bash
python main_gpu.py   --dataset yelp --model_name LightGCN --mode meta  --data_path  ./yelp_sample_conflict  --simulation  --epochs 100
```

```bash
python main_gpu.py   --dataset yelp --model_name LightGCN --mode fixed_weights  --data_path  ./yelp_sample_conflict  --simulation  --epochs 100
```

Note: You can toggle between IM-Net and the Static Weighting baseline within the configuration dict located in main_gpu.py.

## 📊 Expected Outputs & Visual Verification

Running the simulation will generate tracking logs and plot data that correspond to the observations detailed in the study:


Optimization Convergence (Ref: Fig. 5a & 7a):
You will observe the training loss of the Static Baseline spike around Epoch 9, while IM-Net maintains a smooth, monotonic decrease ("loss landscape smoothing").

Interference Energy vs. Gradient Alignment (Ref: Fig. 5b & 6):
Logs will show a sharp interference peak at Epoch 11 exactly when gradient cosine similarity reaches ≈−0.88. Both metrics will quickly recover under IM-Net control.

Anticipatory Weight Evolution (Ref: Fig. 7b):
The tracked auxiliary weight (w_aux ) will demonstrate preemptive rising (starting at Epoch 8) before the conflict peak hits at Epoch 11, proving the long-horizon synergy strategy of IM-Net.

Results will be saved in the results/ directory as CSV logs and PDF/PNG plots.

## 📝 Disclaimer

This is an early-version release focused specifically on validating the phase transitions and dynamical stability under forced conflict conditions. Future versions will scale this environment to real-world, unforced multi-task topologies.
