Multi-Fidelity Neural Network (MFNN)
====================================

The Multi-Fidelity Neural Network (MFNN) is a neural surrogate model
designed to learn nonlinear correlations between low-fidelity and
high-fidelity data sources.

Unlike Co-Kriging, which assumes a linear autoregressive relationship
between fidelities, MFNN provides a flexible neural architecture that
can represent both linear and nonlinear cross-fidelity mappings.

In mfbo, MFNN is used when:

- A low-fidelity function :math:`f_L(\mathbf{x})` is available,
- The relationship between fidelities may be nonlinear,
- Scalability beyond Gaussian processes is required.

Model Formulation
-----------------

Let:

- :math:`f_L(\mathbf{x})` denote the low-fidelity evaluation,
- :math:`f_H(\mathbf{x})` denote the high-fidelity objective.

The MFNN surrogate is defined as

.. math::

   \hat{f}(\mathbf{x}, f_L(\mathbf{x}))
   =
   \alpha\, F_l(\mathbf{x}, f_L(\mathbf{x}))
   +
   (1 - \alpha)\, F_{nl}(\mathbf{x}, f_L(\mathbf{x})),
   \qquad
   \alpha \in [0,1].

Here:

- :math:`F_l` is a linear mapping component,
- :math:`F_{nl}` is a nonlinear mapping component,
- :math:`\alpha` is a learnable gating parameter.

The coefficient :math:`\alpha` balances the relative contribution
of the two subnetworks and is optimized jointly with all network parameters.

Linear Subnetwork
-----------------

The linear component :math:`F_l` consists of a single affine transformation:

.. math::

   F_l(\mathbf{x}, f_L(\mathbf{x}))
   =
   \mathbf{w}^T
   \begin{bmatrix}
   \mathbf{x} \\
   f_L(\mathbf{x})
   \end{bmatrix}
   +
   b.

This subnetwork captures:

- Linear correlations between fidelities,
- Scale shifts,
- Bias corrections.

Because it has no hidden layers or activation functions,
it acts as a parametric linear correction model.

Nonlinear Subnetwork
--------------------

The nonlinear component :math:`F_{nl}` is implemented as a multilayer
perceptron (MLP):

.. math::

   \mathbf{a}^{(n+1)}
   =
   \sigma\!\left(
   \mathbf{W}^{(n)} \mathbf{a}^{(n)}
   +
   \mathbf{b}^{(n)}
   \right).

This network receives the concatenated input

.. math::

   \begin{bmatrix}
   \mathbf{x} \\
   f_L(\mathbf{x})
   \end{bmatrix}

and learns higher-order interactions and nonlinear discrepancy
structures between fidelity levels.

The nonlinear branch allows MFNN to represent complex mappings
that cannot be captured by linear autoregressive models.

Gating Mechanism
----------------

The scalar parameter :math:`\alpha` controls the mixture of the two experts.

During training:

- :math:`\alpha` is treated as a trainable parameter.
- It is typically constrained to :math:`[0,1]` (e.g., via sigmoid).

When:

- :math:`\alpha \approx 1`, the model behaves primarily as a linear mapping.
- :math:`\alpha \approx 0`, nonlinear corrections dominate.

This adaptive blending improves robustness across problems
with varying cross-fidelity structures.

Training Objective
------------------

Given high-fidelity training data

.. math::

   \mathcal{D}_H
   =
   \{(\mathbf{x}_i, f_H(\mathbf{x}_i))\}_{i=1}^{n_H},

the MFNN parameters are optimized to minimize a regression loss:

.. math::

   \mathcal{L}
   =
   \frac{1}{n_H}
   \sum_{i=1}^{n_H}
   \left(
   \hat{f}(\mathbf{x}_i, f_L(\mathbf{x}_i))
   -
   f_H(\mathbf{x}_i)
   \right)^2.

If low-fidelity data is abundant and inexpensive,
:math:`f_L(\mathbf{x})` may be evaluated directly.

Uncertainty Estimation via Ensembles
------------------------------------

A single MFNN is deterministic.
To estimate epistemic uncertainty, mfbo uses deep ensembles.

Let:

.. math::

   \{\hat{f}^{(m)}(\mathbf{x})\}_{m=1}^{M}

denote independently initialized MFNN models.

The ensemble predictive mean is:

.. math::

   \mu(\mathbf{x})
   =
   \frac{1}{M}
   \sum_{m=1}^{M}
   \hat{f}^{(m)}(\mathbf{x}),

and predictive variance is estimated as:

.. math::

   \sigma^2(\mathbf{x})
   =
   \frac{1}{M}
   \sum_{m=1}^{M}
   \left(
   \hat{f}^{(m)}(\mathbf{x})
   -
   \mu(\mathbf{x})
   \right)^2.

This empirical variance is used within acquisition functions
for uncertainty-aware optimization.

Computational Properties
------------------------

Compared to Co-Kriging:

- No kernel matrix inversion is required.
- Training scales approximately linearly with dataset size.
- Larger datasets can be handled more efficiently.

However:

- Training requires iterative gradient-based optimization.
- Hyperparameter tuning may be more involved.

Strengths
---------

MFNN is effective when:

- Cross-fidelity relationships are nonlinear.
- Large datasets are available.
- Gaussian process cubic scaling is prohibitive.
- Flexible representation is required.

Limitations
-----------

Potential challenges include:

- Sensitivity to network architecture and hyperparameters.
- Lack of closed-form uncertainty.
- Risk of overfitting with small high-fidelity datasets.

Using deep ensembles mitigates some uncertainty limitations
but increases computational cost.

Summary
-------

MFNN extends neural surrogate modeling to multi-fidelity settings
by combining linear and nonlinear mappings through a learnable gate.

It provides:

- Flexible cross-fidelity modeling,
- Scalability to larger datasets,
- Compatibility with ensemble-based uncertainty estimation.

It serves as a neural alternative to classical Co-Kriging
within the mfbo multi-fidelity framework.