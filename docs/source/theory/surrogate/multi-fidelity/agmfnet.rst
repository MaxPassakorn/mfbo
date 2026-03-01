Adaptive Gating Multi-Fidelity Neural Network (AGMF-Net)
=========================================================

AGMF-Net is a neural multi-fidelity surrogate designed to model
complex nonlinear relationships between low- and high-fidelity data
using a gated mixture-of-experts architecture.

Unlike classical autoregressive models and fixed-weight neural
multi-fidelity approaches, AGMF-Net introduces:

- Input-dependent expert weighting,
- Explicit residual discrepancy modeling,
- Adaptive task-level loss balancing during training.

In mfbo, AGMF-Net serves as a high-capacity multi-fidelity surrogate
for problems where fidelity relationships vary across the domain.

Model Architecture
------------------

Let:

- :math:`f_L(\mathbf{x})` denote the low-fidelity function,
- :math:`f_H(\mathbf{x})` denote the high-fidelity function.

AGMF-Net defines three expert subnetworks:

- :math:`F_l(\mathbf{x}, f_L(\mathbf{x}))` — linear expert,
- :math:`F_{nl}(\mathbf{x}, f_L(\mathbf{x}))` — nonlinear expert,
- :math:`F_{\mathrm{res}}(\mathbf{x})` — residual expert.

The prediction is given by:

.. math::

   \hat{f}(\mathbf{x}, f_L(\mathbf{x}))
   =
   w_1(\mathbf{x}) F_l(\mathbf{x}, f_L(\mathbf{x}))
   +
   w_2(\mathbf{x}) F_{nl}(\mathbf{x}, f_L(\mathbf{x}))
   +
   w_3(\mathbf{x}) F_{\mathrm{res}}(\mathbf{x}),

where the mixture weights satisfy:

.. math::

   w_i(\mathbf{x}) \ge 0,
   \qquad
   \sum_{i=1}^{3} w_i(\mathbf{x}) = 1.

Gating Network
--------------

The weights are produced by a softmax gating network:

.. math::

   w(\mathbf{x})
   =
   \mathrm{softmax}\!\left(
   G(\mathbf{x}, f_L(\mathbf{x}))
   \right),

where :math:`G` is a learnable mapping.

This input-dependent gating enables:

- Spatially varying fidelity fusion,
- Smooth transitions between modeling regimes,
- Avoidance of subtractive cancellation.

When:

- The fidelity relationship is nearly linear,
  the linear expert may dominate.
- Nonlinear discrepancies appear,
  the nonlinear or residual expert receives higher weight.

Expert Subnetworks
------------------

Linear Expert
^^^^^^^^^^^^^

The linear branch performs an affine transformation of the concatenated input:

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

It captures linear scaling and low-order corrections.

Nonlinear Expert
^^^^^^^^^^^^^^^^

The nonlinear branch is implemented as a multilayer perceptron (MLP):

.. math::

   \mathbf{a}^{(n+1)}
   =
   \sigma\!\left(
   \mathbf{W}^{(n)} \mathbf{a}^{(n)}
   +
   \mathbf{b}^{(n)}
   \right).

It models higher-order nonlinear cross-fidelity interactions.

Residual Expert
^^^^^^^^^^^^^^^

The residual branch depends only on the original input:

.. math::

   F_{\mathrm{res}}(\mathbf{x})
   \approx
   f_H(\mathbf{x}) - f_L(\mathbf{x}).

This explicitly models discrepancies independent of low-fidelity structure.

Training Objective
------------------

Three complementary Huber losses guide learning:

Full Mixture Loss
^^^^^^^^^^^^^^^^^

.. math::

   \mathcal{L}_H
   =
   \mathrm{Huber}\!\left(
   \hat{f}(\mathbf{x}, f_L(\mathbf{x})),
   f_H(\mathbf{x})
   \right).

This trains the full gated model.

Linear+Nonlinear Alignment Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \mathcal{L}_{LH}
   =
   \mathrm{Huber}\!\left(
   F_l(\mathbf{x}, f_L(\mathbf{x}))
   +
   F_{nl}(\mathbf{x}, f_L(\mathbf{x})),
   f_H(\mathbf{x})
   \right).

This aligns transformation-based experts with high-fidelity behavior.

Residual Discrepancy Loss
^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \mathcal{L}_R
   =
   \mathrm{Huber}\!\left(
   F_{\mathrm{res}}(\mathbf{x}),
   f_H(\mathbf{x}) - f_L(\mathbf{x})
   \right).

This explicitly trains the residual branch.

The Huber loss improves robustness to bias and local mismatch
compared to pure mean-squared error.

Adaptive Task Weighting
-----------------------

To balance the three objectives, AGMF-Net uses adaptive task weights:

.. math::

   \mathcal{L}
   =
   \sum_{k \in \{H, LH, R\}}
   w_k^{(\mathrm{task})}
   \mathcal{L}_k,
   \qquad
   w^{(\mathrm{task})}
   =
   \mathrm{softmax}(u),

where :math:`u \in \mathbb{R}^3` are learnable logits.

After each iteration, the logits are updated using relative
log-loss improvements:

.. math::

   \ell^{(r)}
   =
   \begin{bmatrix}
   \log \mathcal{L}_H^{(r)} \\
   \log \mathcal{L}_{LH}^{(r)} \\
   \log \mathcal{L}_R^{(r)}
   \end{bmatrix},
   \qquad
   \Delta \ell^{(r)}
   =
   \ell^{(r)} - \ell^{(r+1)}.

The update rule is:

.. math::

   u^{(r+1)}
   =
   u^{(r)}
   -
   \eta
   \left[
   \mathrm{diag}(w^{(r)})
   -
   w^{(r)} (w^{(r)})^T
   \right]
   \Delta \ell^{(r)}.

This mechanism:

- Increases weight on slowly improving components,
- Prevents premature domination by fast-converging losses,
- Mitigates negative transfer.

Inference and Stability
-----------------------

At inference time:

- The gating network produces convex expert combinations.
- Predictions remain smooth across the domain.
- No task-weight updates are required.

Inputs and outputs are standardized during training
and de-standardized for reporting.

Uncertainty Estimation
----------------------

AGMF-Net is deterministic.
To estimate epistemic uncertainty, mfbo uses deep ensembles.

Given independently trained models:

.. math::

   \{\hat{f}^{(m)}(\mathbf{x})\}_{m=1}^{M},

the ensemble mean and variance are:

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
   \hat{f}^{(m)}(\mathbf{x}) - \mu(\mathbf{x})
   \right)^2.

Computational Characteristics
-----------------------------

Compared to Co-Kriging:

- No matrix inversion is required.
- Scales with network size and dataset size.
- Suitable for larger multi-fidelity datasets.

Compared to MFNN and Ada2MF:

- More expressive due to input-dependent gating.
- More stable due to convex mixture weighting.
- Higher training complexity.

Strengths
---------

AGMF-Net is particularly effective when:

- Cross-fidelity relationships vary spatially,
- Low-fidelity bias is non-uniform,
- Residual discrepancy structure is complex,
- High flexibility is required.

Limitations
-----------

Potential challenges include:

- Increased architectural complexity,
- Longer training time,
- Sensitivity to hyperparameter tuning.

Because task weights evolve during training,
longer training schedules are typically required
for stable convergence.

Summary
-------

AGMF-Net combines:

- Mixture-of-experts modeling,
- Input-dependent gating,
- Explicit residual correction,
- Adaptive task-level weighting.

It provides a highly flexible and robust
multi-fidelity neural surrogate within mfbo,
particularly suited for complex engineering problems.