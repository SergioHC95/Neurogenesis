# Grow When Needed: Adaptive Capacity Expansion in Neural Networks

## 🧠 Project Overview

**Goal:** Design MLPs that start small and *grow adaptively* during training, based on internal signals that reflect learning saturation and bottlenecks. We aim to boost *sample efficiency*, reduce *computational cost*, and maintain strong generalization.

This project explores how internal training signals—such as mutual information, gradient norms, and activation dynamics—can be used to **diagnose capacity limits** and **trigger model growth**.

## 🔬 Motivation

Modern networks are often overparameterized from the outset. But this is wasteful. We hypothesize that:
> A model should only grow **when it needs to**—not before.

This principle can:
- Save compute and memory
- Adapt to different dataset complexities
- Offer interpretability through clear growth triggers

## 🧪 Key Research Questions

1. What internal signals reliably detect saturation in learning dynamics?
2. Does adaptive growth improve compute/sample efficiency?
3. What growth strategies are most effective (depth-first, width-first)?
4. How should newly added neurons/layers be initialized?

## 🧭 Project Structure

```
grow-when-needed/
│
├── configs/           # YAML or JSON experiment configurations
├── notebooks/         # Analysis & development notebooks
├── src/               # Main codebase
│   ├── models/        # MLPs (static and dynamic)
│   ├── diagnostics/   # Mutual info, gradient norms, activation stats
│   ├── data/          # Loaders and synthetic datasets
│   ├── training/      # Trainer, growth triggers, callbacks
│   ├── evaluation/    # Metrics and analysis
│   └── utils/         # Logging, seeding, helpers
│
├── experiments/       # Run scripts for main experiments
├── results/           # Logs, plots, checkpoints
├── tests/             # Unit tests
├── paper/             # Draft paper and figures
│
├── requirements.txt   # Dependencies
├── setup.py           # Install as package
├── LICENSE
└── README.md
```

## 📊 Planned Experiments

- MNIST, Fashion-MNIST, Parity Task, CIFAR-10 (flattened)
- Baselines: small/fixed MLPs, dropout, early stopping
- Ablations: MI vs gradient triggers, growth type, layer init

## 📦 Deliverables

- Modular PyTorch codebase
- Diagnostic toolkit
- Paper-ready writeup
- Visualizations and benchmark results

---

Let the model grow *only when it needs to*. 🌱
