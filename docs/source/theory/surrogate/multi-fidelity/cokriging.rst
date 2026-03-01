Co-Kriging
==========

Co-Kriging is a multi-fidelity extension of Gaussian process regression.
It combines inexpensive low-fidelity evaluations with expensive
high-fidelity data to improve sample efficiency.

In mfbo, Co-Kriging is used when two levels of fidelity are available
and are correlated.

Autoregressive Multi-Fidelity Model
------------------------------------

Let:

- :math:`f_c(\mathbf{x})` denote the low-fidelity function,
- :math:`f_e(\mathbf{x})` denote the high-fidelity function.

The classical autoregressive formulation assumes:

.. math::

   f_e(\mathbf{x})
   =
   \rho\, f_c(\mathbf{x})
   +
   \delta(\mathbf{x}),

where:

- :math:`\rho \in \mathbb{R}` is a scaling parameter,
- :math:`\delta(\mathbf{x})` is an independent Gaussian process
  modeling the discrepancy between fidelities.

This formulation captures linear correlation between fidelity levels
while allowing structured deviations.

Training Data
-------------

Let the combined dataset be:

.. math::

   \mathbf{X}
   =
   \begin{bmatrix}
   \mathbf{X}_c \\
   \mathbf{X}_e
   \end{bmatrix},
   \qquad
   \mathbf{y}
   =
   \begin{bmatrix}
   \mathbf{y}_c \\
   \mathbf{y}_e
   \end{bmatrix},

where:

- :math:`\mathbf{X}_c \in \mathbb{R}^{n_c \times d}` are low-fidelity inputs,
- :math:`\mathbf{X}_e \in \mathbb{R}^{n_e \times d}` are high-fidelity inputs,
- :math:`\mathbf{y}_c`, :math:`\mathbf{y}_e` are corresponding outputs.

Kernel Structure
----------------

Let:

- :math:`k_c(\cdot, \cdot)` denote the low-fidelity kernel,
- :math:`k_d(\cdot, \cdot)` denote the discrepancy kernel.

Define kernel blocks:

.. math::

   K_{cc} = k_c(\mathbf{X}_c, \mathbf{X}_c),

.. math::

   K_{ce} = k_c(\mathbf{X}_c, \mathbf{X}_e),

.. math::

   K_{ec} = k_c(\mathbf{X}_e, \mathbf{X}_c),

.. math::

   K_{ee}
   =
   \rho^2 k_c(\mathbf{X}_e, \mathbf{X}_e)
   +
   k_d(\mathbf{X}_e, \mathbf{X}_e).

The full covariance matrix becomes:

.. math::

   K
   =
   \begin{bmatrix}
   k_c(\mathbf{X}_c, \mathbf{X}_c)
   &
   \rho\, k_c(\mathbf{X}_c, \mathbf{X}_e)
   \\
   \rho\, k_c(\mathbf{X}_e, \mathbf{X}_c)
   &
   \rho^2 k_c(\mathbf{X}_e, \mathbf{X}_e)
   +
   k_d(\mathbf{X}_e, \mathbf{X}_e)
   \end{bmatrix}.

This block structure encodes cross-fidelity correlations.

Posterior Prediction
--------------------

For a test point :math:`\mathbf{x}`, define the cross-covariance vector:

.. math::

   \mathbf{c}
   =
   \begin{bmatrix}
   k_c(\mathbf{x}, \mathbf{X}_c)
   \\
   \rho\, k_c(\mathbf{x}, \mathbf{X}_e)
   +
   k_d(\mathbf{x}, \mathbf{X}_e)
   \end{bmatrix}.

The predictive mean is:

.. math::

   \hat{f}(\mathbf{x})
   =
   \mu
   +
   \mathbf{c}^T
   K^{-1}
   (\mathbf{y} - \mu \mathbf{1}),

where :math:`\mu` is the common prior mean.

The predictive variance is:

.. math::

   \hat{\sigma}^2(\mathbf{x})
   =
   \rho^2 \sigma_c^2
   +
   \sigma_d^2
   -
   \mathbf{c}^T K^{-1} \mathbf{c}
   +
   \frac{(1 - \mathbf{1}^T K^{-1} \mathbf{c})^2}
        {\mathbf{1}^T K^{-1} \mathbf{1}}.

The mean provides the high-fidelity prediction,
and the variance quantifies epistemic uncertainty.

Hyperparameter Estimation
--------------------------

Model parameters include:

- Kernel hyperparameters for :math:`k_c` and :math:`k_d`,
- Scaling coefficient :math:`\rho`,
- Noise variances (if present).

These parameters are typically estimated by maximizing
the joint log marginal likelihood.

Numerical Considerations
------------------------

Co-Kriging requires inversion of the combined covariance matrix:

- Computational complexity scales as
  :math:`\mathcal{O}((n_c + n_e)^3)`.
- Memory scales as
  :math:`\mathcal{O}((n_c + n_e)^2)`.

Practical recommendations:

- Add jitter for numerical stability.
- Normalize both fidelity outputs consistently.
- Ensure sufficient correlation between fidelities.

Strengths
---------

Co-Kriging is effective when:

- Low-fidelity data is abundant and inexpensive.
- High-fidelity evaluations are costly.
- The fidelity gap is structured and correlated.

It can significantly reduce the number of expensive evaluations required.

Limitations
-----------

Potential challenges include:

- Cubic scaling with total dataset size.
- Reduced benefit if fidelities are weakly correlated.
- Linear scaling assumption may be restrictive.

For nonlinear fidelity relationships, neural multi-fidelity models
may provide improved flexibility.

Summary
-------

Co-Kriging extends Gaussian process regression to multi-fidelity
settings using an autoregressive structure.

It provides:

- Closed-form predictive mean and variance,
- Explicit cross-fidelity correlation modeling,
- Strong performance when fidelities are well aligned.

It is a classical and widely used baseline for multi-fidelity
Bayesian optimization in mfbo.