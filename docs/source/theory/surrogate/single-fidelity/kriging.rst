Kriging (Gaussian Process Regression)
======================================

Kriging is a probabilistic surrogate modeling technique based on
Gaussian process (GP) regression. It provides both a predictive mean
and a predictive variance, making it particularly well suited for
Bayesian optimization.

In mfbo, Kriging serves as a single-fidelity surrogate model for
expensive deterministic or noisy objective functions.

Gaussian Process Prior
-----------------------

Let :math:`f : \mathcal{X} \to \mathbb{R}` be an unknown function.
A Gaussian process defines a prior distribution over functions:

.. math::

   f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')),

where:

- :math:`m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]` is the mean function,
- :math:`k(\mathbf{x}, \mathbf{x}')` is the covariance (kernel) function.

The covariance function determines smoothness, correlation length,
and anisotropy properties of the surrogate.

Kernel Function
---------------

A commonly used kernel in Kriging is the generalized exponential form:

.. math::

   k(\mathbf{x}, \mathbf{x}')
   =
   \exp\!\left(
   -\sum_{j=1}^{d}
   \theta_j
   \left\lVert x_j - x'_j \right\rVert^{p_j}
   \right),

where:

- :math:`\theta_j > 0` are length-scale parameters,
- :math:`p_j \in (0, 2]` control smoothness,
- :math:`d` is the input dimension.

Special cases include:

- Squared exponential (RBF) kernel when :math:`p_j = 2`
- Exponential kernel when :math:`p_j = 1`

Large :math:`\theta_j` implies stronger sensitivity to changes in
dimension :math:`j`.

Training Data
-------------

Given training data

.. math::

   \mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n},

we construct the kernel matrix

.. math::

   K =
   \left[
   k(\mathbf{x}_i, \mathbf{x}_j)
   \right]_{i,j=1}^{n}.

For noisy observations, a diagonal noise term
:math:`\sigma_n^2 I` may be added.

Posterior Prediction
--------------------

For a test point :math:`\mathbf{x}_*`, define the covariance vector

.. math::

   \mathbf{k}_*
   =
   \left[
   k(\mathbf{x}_*, \mathbf{x}_1),
   \dots,
   k(\mathbf{x}_*, \mathbf{x}_n)
   \right]^T.

The posterior predictive mean is

.. math::

   \mu(\mathbf{x}_*)
   =
   \mathbf{k}_*^T
   (K + \sigma_n^2 I)^{-1}
   \mathbf{y},

and the predictive variance is

.. math::

   \sigma^2(\mathbf{x}_*)
   =
   k(\mathbf{x}_*, \mathbf{x}_*)
   -
   \mathbf{k}_*^T
   (K + \sigma_n^2 I)^{-1}
   \mathbf{k}_*.

The mean provides the surrogate prediction,
and the variance quantifies epistemic uncertainty.

Hyperparameter Estimation
--------------------------

Kernel parameters :math:`\theta_j`, smoothness parameters :math:`p_j`,
and noise variance (if applicable) are typically estimated by
maximizing the log marginal likelihood:

.. math::

   \log p(\mathbf{y} \mid \mathbf{X})
   =
   -\frac{1}{2}
   \mathbf{y}^T
   K^{-1}
   \mathbf{y}
   -
   \frac{1}{2}
   \log |K|
   -
   \frac{n}{2}
   \log(2\pi).

This balances model fit and model complexity automatically.

Numerical Considerations
------------------------

Because Kriging requires inversion of the kernel matrix:

- Computational cost scales as :math:`\mathcal{O}(n^3)`
- Memory cost scales as :math:`\mathcal{O}(n^2)`

Practical recommendations:

- Add a small jitter term to the diagonal for stability.
- Normalize inputs and outputs before training.
- Monitor kernel conditioning in high-dimensional settings.

Strengths
---------

Kriging is particularly effective when:

- The number of evaluations is limited.
- The design dimension is moderate.
- Smoothness assumptions are reasonable.
- Uncertainty quantification is critical.

Limitations
-----------

Potential challenges include:

- Cubic scaling in dataset size.
- Sensitivity to kernel choice.
- Reduced performance in very high-dimensional spaces.

In such settings, neural surrogate models may provide
better scalability.

Summary
-------

Kriging provides a probabilistic surrogate with:

- Closed-form posterior mean and variance
- Natural uncertainty quantification
- Strong performance in low- to moderate-dimensional problems

It remains a foundational surrogate model in Bayesian optimization
and is fully supported within mfbo.