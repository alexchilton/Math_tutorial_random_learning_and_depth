# Mathematical Foundations for Machine Learning
**Self-Study Tutorial Series**

A structured deep-dive into the mathematics underlying modern ML and AI — from statistical estimation theory through to measure theory, variational calculus, and theoretical physics. Each notebook combines rigorous derivations with practical implementations and visualisations.

---

## Notebooks

| # | Topic | Key Concepts |
|---|-------|-------------|
| 01 | [Statistical Foundations](01_statistical_foundations.ipynb) | Law of Large Numbers, estimators, Bessel's correction, OLS regression, correlation |
| 02 | [Information Theory & XAI](02_information_theory_and_xai.ipynb) | Shannon entropy, KL divergence, mutual information, explainable AI applications |
| 02a | [Approximating KL Divergence](02a_approximating_kl_divergence.ipynb) | Monte-Carlo KL estimators k1/k2/k3, bias vs variance, f-divergences, control variates, Bregman divergence, PPO/RLHF |
| 03 | [Matrix & Tensor Calculus](03_matrix_tensor_calculus_regression.ipynb) | Linear transformations, eigendecomposition, SVD, covariance matrices |
| 03a | [Advanced Matrix Calculus I](03a_advanced_matrix_calculus.ipynb) | Fréchet derivatives, Jacobians, forward/reverse-mode AD, Kronecker products |
| 03b | [Advanced Matrix Calculus II](03b_advanced_matrix_calculus_applications.ipynb) | Hessians, Newton's method, Gauss-Newton, reparameterisation trick, Euler-Lagrange |
| 04 | [Hypothesis Testing](04_p_values_t_statistics_confidence_intervals.ipynb) | t-statistics, p-values, confidence intervals, Type I/II errors, power analysis |
| 05 | [Measure Theory & Probability](05_measure_theory_and_probability.ipynb) | σ-algebras, probability spaces, Radon-Nikodym, L^p spaces, Fubini's theorem |
| 06 | [Quantum Mechanics](06_quantum_mechanics_and_field_theory.ipynb) | Hilbert spaces, Schrödinger equation, path integrals, second quantisation |
| 07 | [General Relativity](07_general_relativity.ipynb) | Differential geometry, metric tensor, Einstein field equations, Schwarzschild solution |
| 08a | [First Steps: Topology & Infinite Dimensions](08a_first_steps_topology_and_infinite_dimensions.ipynb) **← start here** | Topology vs topological space, functions *as* vectors, the five things that break in infinite dimensions |
| 08 | [Metric, Normed, Banach & Hilbert Spaces](08_metric_normed_banach_hilbert_spaces.ipynb) | Parallelogram law, completeness, projection theorem, orthonormal bases, RKHS & the kernel trick, Lie algebras |
| 09 | [Lie Groups & Lie Algebras](09_lie_groups_and_lie_algebras.ipynb) | Manifolds vs groups, deriving so(3), exp/log & Rodrigues, commutators & BCH, structure constants, optimisation on manifolds, SU(2) double cover |
| 10 | [Manifolds, Charts & the Axioms](10_manifolds_charts_and_the_axioms.ipynb) | Charts & transition maps, quotients, projective space, Hausdorff & second-countable counterexamples, partitions of unity, orientability, smooth structures, Riemannian metrics |

---

## Progression

```
Statistical Intuition → Linear Algebra → Theoretical Probability → Advanced Optimisation → Physics
      01, 04               03, 03a, 03b            05                  03b, 02              06, 07

Spaces & geometry:   08 (metric → Hilbert)
                     10 (manifolds) ─┬─ + group structure ──→ 09 (Lie groups & algebras)
                                     └─ + metric on T_pM ───→ 07 (Riemannian / GR)
```

The series is designed to build from practitioner-level statistics toward the mathematical foundations that make modern deep learning (VAEs, diffusion models, transformers) formally rigorous.

---

## Highlights

**Notebook 03a/03b — Advanced Matrix Calculus**
Goes beyond standard deep learning courses: Fréchet derivatives as coordinate-free operators, why schoolbook calculus breaks for matrix-valued functions, forward vs reverse-mode autodiff from first principles, and connections to variational calculus (brachistochrone, minimal surfaces).

**Notebook 10 — Manifolds, Charts & the Axioms**
Foundational to both 07 and 09 despite the number. Every condition in the definition of a
manifold is justified by the specific monster it excludes, computed rather than described:
the line with two origins where 1/n converges to two different points, the figure-eight whose
crossing splits a punctured neighbourhood into four pieces instead of two, a cone that is a
topological manifold but has no tangent plane, and R with the x^3 chart giving an incompatible
smooth structure. Ends with a partition of unity assembling a global integral from four local
ones, and a Riemannian metric making the great circle measurably shorter than the latitude.
Computational companion to The Bright Side of Mathematics' 59-lecture Manifolds series.

**Notebook 09 — Lie Groups & Lie Algebras**
Why you cannot average two rotations, and what to do instead. Derives so(3) from R^T R = I
rather than asserting it, verifies the commutator really is the infinitesimal
non-commutativity of the group, recovers the Levi-Civita symbol as structure constants, and
runs gradient descent over SO(3) in the Lie algebra to match the closed-form Kabsch solution.
Also pins down where manifolds, tangent spaces and Riemannian structure sit relative to
notebooks 07 and 08.

**Notebook 02a — Approximating KL Divergence**
A beginner's walk through John Schulman's [kl-approx](http://joschu.net/blog/kl-approx.html)
post, with every claim recomputed. Starts from KL on a two-outcome coin, builds bias vs
variance from scratch, motivates control variates on a toy problem with no KL in it, and
arrives at k3 = (r-1) - log r. Reproduces both of the post's tables. Adds two things the post
does not: the p/q convention trap (its prose and its code disagree, invisibly for
equal-variance Gaussians), and the condition for these estimators to work at all -- E_q[r^2]
must be finite, which for Gaussians means sigma_p < sqrt(2) sigma_q.

**Notebook 08a — First Steps** *(gentler prequel to 08)*
Four questions, slowly, with numbers: what a topology actually is and why "open sets", in what
sense a function *is* a vector (the dot product literally becoming an integral as the index
goes continuous), what infinite-dimensional means, and the five conveniences of R^n that fail
without one — completeness, norm equivalence, compactness, continuity of linear maps, and
angles. Includes two corrections to commonly-taught sloppiness: continuity depends on the
topology at *both* ends, and best sup-norm approximation by polynomials is unique
(Chebyshev equioscillation), so "Banach can't do best approximation" is too strong.

**Notebook 08 — Metric to Hilbert Spaces**
Answers the question the textbooks bury: *what does a Hilbert space actually buy you over
ordinary linear algebra?* Built by breaking things — the parallelogram law tested numerically
across p-norms (only p=2 survives), a Cauchy sequence of continuous functions escaping its own
space, non-unique nearest points in L1 and L-infinity, Gibbs phenomenon as L2-vs-uniform
convergence, and the kernel trick shown to be literally an inner product. Closes with where
Lie algebras sit relative to all of it.

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
