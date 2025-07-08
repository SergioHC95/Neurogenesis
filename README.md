# Online Neurogenesis Through Synaptic Plasticity Signals  
## A Framework for Adaptive Neural Network Growth During Training

### Project Overview

**Goal:**  
Develop MLPs that start small and grow adaptively during training, guided by internal signals indicating capacity bottlenecks or saturation. The aim is to improve sample efficiency, reduce compute costs, and maintain or improve generalization.

This framework uses internal diagnostics—such as mutual information, gradient norms, and activation dynamics—to decide when and how the model should expand during training.

---

### Motivation

Modern neural networks are typically overparameterized from the start—wasting compute and memory on capacity that may not be needed.

We propose a different principle:

> Grow only when necessary.

This approach promises to:
- Save computation and memory during early training
- Adaptively match model complexity to data
- Provide interpretability via explicit growth triggers

---

### Core Research Questions

1. Which internal signals most reliably indicate capacity saturation?
2. Does online model growth improve training efficiency or generalization?
3. What growth strategies are most effective (depth-first, width-first, hybrid)?
4. How should new neurons or layers be initialized to integrate seamlessly?

---

### Planned Experiments

- Tasks: MNIST, Fashion-MNIST, Parity Task, flattened CIFAR-10  
- Baselines: fixed-size MLPs, dropout, early stopping  
- Ablations: growth trigger type (MI vs gradients), growth direction, init schemes  

---

### Deliverables

- A modular, extensible PyTorch codebase  
- Built-in diagnostic toolkit  
- Benchmark results and visualizations  
- Draft paper summarizing methods and findings  

---

Let the model grow *only when it needs to*.  
