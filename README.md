# Physics-Informed Neural Networks for Inverse Permeability Estimation in Enhanced Geothermal Systems (1D)

This repository contains the full implementation and results of a **Physics-Informed Neural Network (PINN)** framework for **inverse permeability estimation** in a **1D Enhanced Geothermal System (EGS)** benchmark problem. The project demonstrates the strengths and limitations of inverse PINNs under **sparse, noisy data** using a rigorously controlled numerical setup.

**Author:** Nehor Some  
**Affiliation:** Department of Mathematical Sciences, University of Texas at El Paso (UTEP)  
**Email:** nsome@miners.utep.edu  

---

## Problem Overview

We consider a **1D nonlinear diffusion equation** modeling pressure evolution in a geothermal fracture:

\[
u_t = \frac{\partial}{\partial x}\left( k(x)\, u_x \right), \quad x \in [0,1],\; t \in [0,1]
\]

with boundary and initial conditions:

\[
u(0,t)=0, \quad u(1,t)=1, \quad u(x,0)=0.
\]

The **true permeability field** is given by:

\[
k_{\text{true}}(x) = 1 + 0.5 \sin(2\pi x)
\]

The goal is to **infer the unknown permeability field \(k(x)\)** using **only sparse, noisy pressure measurements**.

---

## Project Objectives

- Generate **high-fidelity synthetic data** using the **Finite Element Method (FEM)**.
- Train a **dual-network PINN** to jointly learn:
  - The pressure field \(u(t,x)\)
  - The permeability field \(k(x)\)
- Quantify:
  -  **Forward accuracy**
  -  **Inverse instability**
- Demonstrate the **ill-posed nature of inverse diffusion problems**.

---

## PINN Architecture

- **Solution Network:**  
  \(N_u: (t,x) \mapsto u\)  
  - 3 hidden layers × 96 neurons  
  - `tanh` activation  

- **Permeability Network:**  
  \(N_k: x \mapsto k\)  
  - 3 hidden layers × 96 neurons  
  - `tanh` activation  
  - Positivity enforced via:
    \[
    k = \text{softplus}(z) + 0.1
    \]

- **Optimization:** Adam Optimizer  
- **Framework:** PyTorch  
- **Automatic Differentiation:** Used for PDE residuals

---

## Loss Function

\[
L = \lambda_{\text{data}}L_{\text{data}} + \lambda_{\text{pde}}L_{\text{pde}}
+ \lambda_{\text{bc}}L_{\text{bc}} + \lambda_{\text{ic}}L_{\text{ic}}
\]

With **curriculum training** on the PDE residual term.

---

## 📊 Results Summary

| Quantity | Metric | Value |
|----------|--------|-------|
| Pressure \(u\) | RMSE | **0.032** |
| Pressure \(u\) | Relative L2 | **6.58%** |
| Permeability \(k\) | RMSE | **0.445** |
| Permeability \(k\) | Relative L2 | **42.0%** |
| PDE Residual | RMS | **0.0951** |

Forward solution is highly accurate  
Inverse permeability recovery is unstable due to **ill-posedness**

---

## Key Insight

Inverse diffusion problems are **mathematically ill-posed**:  
small data noise leads to large oscillations in the recovered coefficient.  
This project explicitly **quantifies and demonstrates this instability**.

---

## Repository Structure

