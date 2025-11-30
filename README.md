# LLM From Scratch – Muon vs Adam Optimizer (Implementation Project)

This repository contains my full implementation of a Mixture-of-Experts LLM, following a research-driven approach.  
All core components are coded manually in modular form, trained end-to-end, and executed on Google Colab GPU.

---

### What This Project Includes (Completed Work)

1. **Transformer Architecture**
   - Token & Positional Embeddings (RoPE)
   - Multi-Head Self Attention
   - Feedforward (Expert) Network
   - Mixture-of-Experts with Top-K Router
   - Auxiliary Load-Balancing Loss

2. **Training & Optimization**
   - Adam optimizer coded manually
   - Muon optimizer coded and integrated
   - Gradient updates, loss computation
   - Config-based training launch

3. **Full Pipeline**
   - Data preprocessing + batching
   - Model training script
   - Experiment runner for multiple configs
   - Modular design (layers, trainer, utils, routing)

4. **Execution**
   - Developed locally in VS Code
   - Colab GPU training verified & logged
   - Source managed through GitHub repo

---

### Goals of this Project
- Learn how LLMs are implemented internally (not black-box)
- Implement Muon vs Adam optimizers and compare behaviour
- Build a modular, research-style codebase
- Run training on GPU and validate convergence
