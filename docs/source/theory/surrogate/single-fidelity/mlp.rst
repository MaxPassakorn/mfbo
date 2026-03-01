Multilayer Perceptron (MLP) Surrogate
=====================================

A multilayer perceptron (MLP) is a feedforward neural network used as a
deterministic surrogate model. Unlike Gaussian processes, MLPs do not
provide uncertainty estimates in closed form, but they offer strong
scalability and expressive power for nonlinear regression.

In mfbo, MLPs are used as single-fidelity neural surrogates and can
be combined into ensembles to estimate epistemic uncertainty.

Network Architecture
--------------------

An MLP consists of:

- An input layer
- A sequence of fully connected hidden layers
- An output layer

Each layer performs an affine transformation followed by a nonlinear activation.
Let :math:`\mathbf{a}^{(n)}` denote the activation vector at layer :math:`n`.
Forward propagation is defined as:

.. math::

   \mathbf{a}^{(n+1)}
   =
   \sigma\!\left(
   \mathbf{W}^{(n)} \mathbf{a}^{(n)}
   +
   \mathbf{b}^{(n)}
   \right),

where:

- :math:`\mathbf{W}^{(n)}` is the weight matrix,
- :math:`\mathbf{b}^{(n)}` is the bias vector,
- :math:`\sigma(\cdot)` is a nonlinear activation applied elementwise.

Stacking affine and nonlinear transformations enables the network
to approximate highly nonlinear functions.

Universal Approximation
-----------------------

MLPs possess the universal approximation property:
given sufficient hidden units, they can approximate any continuous
function on a compact domain to arbitrary accuracy.

This makes them flexible surrogates for complex engineering objectives,
especially when smoothness assumptions of Gaussian processes
are too restrictive.

Activation Function
-------------------

To enhance gradient flow and representational smoothness,
mfbo employs the Mish activation function:

.. math::

   \mathrm{Mish}(x)
   =
   x \cdot \tanh(\mathrm{softplus}(x))
   =
   x \cdot \tanh\!\left(\ln(1 + e^x)\right).

Mish is smooth and non-monotonic, which may improve optimization
dynamics compared to piecewise-linear activations such as ReLU.

Other activation functions may be used depending on application needs.

Parameter Initialization
------------------------

Proper initialization is important for stable deep network training.

mfbo uses Kaiming (He) normal initialization for weight matrices:

.. math::

   W_{ij}
   \sim
   \mathcal{N}\!\left(
   0,
   \frac{2}{\mathrm{fan\_in}}
   \right),
   \qquad
   \mathbf{b}^{(n)} = \mathbf{0},

where ``fan_in`` denotes the number of input units to the layer.

This initialization preserves activation variance across layers,
mitigating vanishing or exploding gradients.

Training Objective
------------------

Given training data

.. math::

   \mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n,

the network parameters are optimized to minimize a regression loss,
commonly mean squared error:

.. math::

   \mathcal{L}
   =
   \frac{1}{n}
   \sum_{i=1}^n
   \left(
   \hat{f}(\mathbf{x}_i) - y_i
   \right)^2.

For robust training, alternative losses such as Huber loss
may also be used.

Optimization is typically performed using stochastic gradient-based methods.

Uncertainty via Deep Ensembles
------------------------------

Unlike Gaussian processes, a single MLP does not provide predictive variance.
To estimate epistemic uncertainty, mfbo uses **deep ensembles**.

An ensemble consists of independently initialized networks:

.. math::

   \{ \hat{f}^{(m)}(\mathbf{x}) \}_{m=1}^{M}.

Each model is trained separately with different random initializations
(and optionally shuffled training batches).

The ensemble predictive mean is:

.. math::

   \mu(\mathbf{x})
   =
   \frac{1}{M}
   \sum_{m=1}^M
   \hat{f}^{(m)}(\mathbf{x}),

and the predictive variance is estimated as:

.. math::

   \sigma^2(\mathbf{x})
   =
   \frac{1}{M}
   \sum_{m=1}^M
   \left(
   \hat{f}^{(m)}(\mathbf{x}) - \mu(\mathbf{x})
   \right)^2.

This empirical variance provides a measure of epistemic uncertainty
that can be used in acquisition functions.

Computational Properties
------------------------

Compared to Kriging:

- Training scales approximately linearly with dataset size.
- No matrix inversion is required.
- Memory usage scales with network size rather than data size.

MLPs are therefore more scalable to large datasets.

Strengths
---------

MLP surrogates are effective when:

- The dataset is moderately large.
- The objective exhibits complex nonlinear structure.
- High-dimensional inputs are present.
- Scalability is more important than exact uncertainty modeling.

Limitations
-----------

Potential limitations include:

- Lack of closed-form uncertainty.
- Sensitivity to hyperparameters (depth, width, learning rate).
- Risk of overfitting with small datasets.

Deep ensembles mitigate some of these issues but increase computational cost.

Summary
-------

MLPs provide flexible nonlinear surrogate models suitable for
moderate- to high-dimensional optimization problems.

When combined with deep ensembles, they provide uncertainty estimates
that enable their integration into Bayesian optimization workflows
within mfbo.