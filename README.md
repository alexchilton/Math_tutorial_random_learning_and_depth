# Mathematical Foundations for Machine Learning
**Self-Study Tutorial Series**

A structured deep-dive into the mathematics underlying modern ML and AI — from statistical estimation theory through to measure theory, variational calculus, and theoretical physics. Each notebook combines rigorous derivations with practical implementations and visualisations.

---

## Notebooks

| # | Topic | Key Concepts |
|---|-------|-------------|
| 01 | [Statistical Foundations](01_statistical_foundations.ipynb) | Law of Large Numbers, estimators, Bessel's correction, OLS regression, correlation |
| 02 | [Information Theory & XAI](02_information_theory_and_xai.ipynb) | Shannon entropy, KL divergence, mutual information, explainable AI applications |
| 03 | [Matrix & Tensor Calculus](03_matrix_tensor_calculus_regression.ipynb) | Linear transformations, eigendecomposition, SVD, covariance matrices |
| 03a | [Advanced Matrix Calculus I](03a_advanced_matrix_calculus.ipynb) | Fréchet derivatives, Jacobians, forward/reverse-mode AD, Kronecker products |
| 03b | [Advanced Matrix Calculus II](03b_advanced_matrix_calculus_applications.ipynb) | Hessians, Newton's method, Gauss-Newton, reparameterisation trick, Euler-Lagrange |
| 04 | [Hypothesis Testing](04_p_values_t_statistics_confidence_intervals.ipynb) | t-statistics, p-values, confidence intervals, Type I/II errors, power analysis |
| 05 | [Measure Theory & Probability](05_measure_theory_and_probability.ipynb) | σ-algebras, probability spaces, Radon-Nikodym, L^p spaces, Fubini's theorem |
| 06 | [Quantum Mechanics](06_quantum_mechanics_and_field_theory.ipynb) | Hilbert spaces, Schrödinger equation, path integrals, second quantisation |
| 07 | [General Relativity](07_general_relativity.ipynb) | Differential geometry, metric tensor, Einstein field equations, Schwarzschild solution |

---

## Progression

```
Statistical Intuition → Linear Algebra → Theoretical Probability → Advanced Optimisation → Physics
      01, 04               03, 03a, 03b            05                  03b, 02              06, 07
```

The series is designed to build from practitioner-level statistics toward the mathematical foundations that make modern deep learning (VAEs, diffusion models, transformers) formally rigorous.

---

## Highlights

**Notebook 03a/03b — Advanced Matrix Calculus**
Goes beyond standard deep learning courses: Fréchet derivatives as coordinate-free operators, why schoolbook calculus breaks for matrix-valued functions, forward vs reverse-mode autodiff from first principles, and connections to variational calculus (brachistochrone, minimal surfaces).

**Notebook 05 — Measure Theory**
Rigorous treatment of why continuous random variables have P(X=x)=0, convergence theorems, and the Radon-Nikodym theorem — the theoretical backbone of probability that most ML courses skip.

**Notebook 02 — Information Theory & XAI**
Bridges KL divergence (core to VAE loss functions) with applied explainability using real datasets (Titanic, California Housing).

---

## Visualisations

31 original diagrams covering regression geometry, eigendecomposition, Newton vs gradient descent convergence, ODE sensitivity, reparameterisation, Radon-Nikodym, brachistochrone curves, and more.

---

## Stack

`NumPy` · `SciPy` · `scikit-learn` · `statsmodels` · `matplotlib` · `seaborn`

---

## Related Work

This mathematical groundwork underpins the applied projects in the [CAS Advanced Machine Learning](https://github.com/alexchilton/CAS_AML_Final_Project) series — particularly the VAE derivations, gradient-based optimisation in latent space, and probabilistic generation.
