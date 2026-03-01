Dual-Adaptive Multi-Fidelity Neural Network (Ada2MF)
====================================================

The Dual-Adaptive Multi-Fidelity Neural Network (Ada2MF) is a neural
multi-fidelity surrogate designed to improve cross-fidelity modeling
through an additive decomposition and adaptive weighting strategy.

Compared to classical MFNN architectures, Ada2MF introduces:

- A residual correction branch,
- Smooth adaptive gating of subnetworks,
- A principled additive structure that enhances flexibility.

In mfbo, Ada2MF is used as a scalable neural alternative to
Gaussian-process-based multi-fidelity models when nonlinear
cross-fidelity relationships are present.

Model Formulation
-----------------

Let:

- :math:`f_L(\mathbf{x})` denote the low-fidelity function,
- :math:`f_H(\mathbf{x})` denote the high-fidelity function.

The Ada2MF surrogate is defined as:

.. math::

   \hat{f}(\mathbf{x}, f_L(\mathbf{x}))
   =
   \tanh(\alpha_1)\, F_l(\mathbf{x}, f_L(\mathbf{x}))
   +
   \tanh(\alpha_2)\, F_{nl}(\mathbf{x}, f_L(\mathbf{x}))
   +
   \tanh(\alpha_3)\, F_{\mathrm{res}}(\mathbf{x}),

where:

- :math:`F_l` is a linear mapping subnetwork,
- :math:`F_{nl}` is a nonlinear mapping subnetwork,
- :math:`F_{\mathrm{res}}` is a residual correction subnetwork,
- :math:`\alpha_i \in \mathbb{R}` are learnable gating parameters.

The hyperbolic tangent ensures smooth bounded weighting of each component.

Additive Decomposition
----------------------

Unlike MFNN, which combines only linear and nonlinear mappings,
Ada2MF explicitly introduces a residual branch:

.. math::

   F_{\mathrm{res}}(\mathbf{x})
   \approx
   f_H(\mathbf{x}) - f_L(\mathbf{x}).

This residual component captures discrepancies that:

- Are independent of low-fidelity structure,
- Cannot be represented through joint transformations,
- Require direct modeling in the original input space.

The additive structure increases expressive capacity while maintaining
interpretability of components.

Subnetwork Structure
--------------------

Linear Component
^^^^^^^^^^^^^^^^

The linear branch performs an affine mapping:

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

It captures:

- Linear correlations,
- Scale shifts,
- Low-order fidelity relationships.

Nonlinear Component
^^^^^^^^^^^^^^^^^^^

The nonlinear branch is implemented as a multilayer perceptron (MLP)
receiving the concatenated input:

.. math::

   \begin{bmatrix}
   \mathbf{x} \\
   f_L(\mathbf{x})
   \end{bmatrix}.

Forward propagation follows:

.. math::

   \mathbf{a}^{(n+1)}
   =
   \sigma\!\left(
   \mathbf{W}^{(n)} \mathbf{a}^{(n)}
   +
   \mathbf{b}^{(n)}
   \right).

This branch models higher-order nonlinear cross-fidelity mappings.

Residual Component
^^^^^^^^^^^^^^^^^^

The residual branch receives only the original input :math:`\mathbf{x}`.
It models discrepancies not directly explained by the transformed
low-fidelity output.

This decoupling improves flexibility when low-fidelity information
is partially informative but incomplete.

Adaptive Weighting
------------------

The coefficients :math:`\alpha_i` regulate the relative importance
of each branch.

The smooth gating:

- Prevents abrupt dominance of any component,
- Allows gradual adaptation during training,
- Improves numerical stability compared to hard switching.

During training:

- :math:`\alpha_i` are optimized jointly with network parameters.
- The model automatically emphasizes the most informative components.

Simplified Training in mfbo
---------------------------

The original Ada2MF framework includes:

- An adaptive multi-fidelity (AMF) module,
- An adaptive fast weighting (AFW) mechanism for balancing losses.

In mfbo, when:

- The low-fidelity function :math:`f_L(\mathbf{x})` is directly available,
- No separate low-fidelity surrogate is required,

the architecture is simplified:

- The AMF low-fidelity surrogate branch is omitted.
- Only the high-fidelity regression loss is optimized.
- The AFW mechanism is unnecessary.

The core additive decomposition and gating structure remain intact.

Training Objective
------------------

Given high-fidelity data:

.. math::

   \mathcal{D}_H
   =
   \{(\mathbf{x}_i, f_H(\mathbf{x}_i))\}_{i=1}^{n_H},

parameters are optimized to minimize:

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

Alternative robust losses (e.g., Huber loss) may also be used.

Uncertainty via Ensembles
-------------------------

A single Ada2MF model is deterministic.
To estimate epistemic uncertainty, mfbo employs deep ensembles.

Let:

.. math::

   \{\hat{f}^{(m)}(\mathbf{x})\}_{m=1}^{M}

denote independently trained Ada2MF models.

The ensemble mean and variance are computed as:

.. math::

   \mu(\mathbf{x})
   =
   \frac{1}{M}
   \sum_{m=1}^{M}
   \hat{f}^{(m)}(\mathbf{x}),

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

This enables integration into Bayesian optimization acquisition functions.

Computational Properties
------------------------

Compared to Co-Kriging:

- No kernel inversion is required.
- Training scales with network size and dataset size.
- Suitable for larger datasets.

Compared to MFNN:

- Additional residual branch increases representational capacity.
- Slightly higher computational cost.
- Improved modeling of complex discrepancy structures.

Strengths
---------

Ada2MF is effective when:

- Cross-fidelity relationships are nonlinear,
- Residual discrepancies are structured,
- Gaussian-process assumptions are too restrictive,
- Scalability beyond cubic GP complexity is needed.

Limitations
-----------

Potential challenges include:

- Increased architectural complexity,
- Hyperparameter sensitivity,
- Need for careful regularization in small datasets.

Deep ensembles improve robustness but increase training cost.

Summary
-------

Ada2MF extends neural multi-fidelity modeling by:

- Introducing additive decomposition,
- Including an explicit residual correction branch,
- Employing smooth adaptive weighting of components.

It provides enhanced flexibility and expressive power
for complex multi-fidelity relationships within mfbo.