Bayesian Optimization
=====================

Bayesian optimization (BO) is a sample-efficient strategy for optimizing
expensive black-box functions. In mfbo, BO is used to iteratively construct
a probabilistic surrogate model and select new evaluation points through
acquisition maximization.

This section introduces the mathematical formulation, the optimization loop,
and the stopping criteria used throughout the library.

Problem Formulation
-------------------

Let :math:`\mathcal{X} \subset \mathbb{R}^k` denote a bounded design domain,
and let

.. math::

   f : \mathcal{X} \rightarrow \mathbb{R}

be an expensive objective function. We adopt a **minimization convention**
throughout the documentation. The global optimum is defined as

.. math::

   \mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x}),
   \qquad
   f^* = f(\mathbf{x}^*).

For maximization problems, one may equivalently minimize :math:`-f`.

Because direct evaluations of :math:`f` are assumed to be computationally
expensive, Bayesian optimization replaces repeated evaluations with a
probabilistic surrogate model.

Surrogate Approximation
-----------------------

Given a dataset of evaluated points

.. math::

   \mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n},
   \qquad y_i = f(\mathbf{x}_i),

a surrogate model constructs a posterior predictive distribution

.. math::

   f(\mathbf{x}) \mid \mathcal{D}_n
   \sim
   p\!\left(f(\mathbf{x}) \mid \mathcal{D}_n\right).

From this distribution we extract:

- Posterior mean:
  :math:`\mu(\mathbf{x}) = \mathbb{E}[f(\mathbf{x}) \mid \mathcal{D}_n]`
- Posterior variance:
  :math:`\sigma^2(\mathbf{x})`

The mean approximates the objective function, while the variance quantifies
**epistemic uncertainty** due to limited sampling density.

In mfbo, both single-fidelity and multi-fidelity surrogate models may be used.
Regardless of model type, the BO loop only requires access to predictive mean
and uncertainty estimates.

Bayesian Optimization Loop
--------------------------

Bayesian optimization proceeds iteratively.

Initialization
^^^^^^^^^^^^^^

1. Define bounded domain :math:`\mathcal{X}`.
2. Generate an initial design (e.g., Latin hypercube or uniform sampling).
3. Evaluate :math:`f` at initial points to obtain :math:`\mathcal{D}_0`.

Iterative Update
^^^^^^^^^^^^^^^^

At iteration :math:`n`:

1. **Fit surrogate model** to :math:`\mathcal{D}_n`.
2. **Construct acquisition function** :math:`\alpha(\mathbf{x})`.
3. **Maximize acquisition**:

   .. math::

      \mathbf{x}_{\text{cand}}
      =
      \arg\max_{\mathbf{x} \in \mathcal{X}}
      \alpha(\mathbf{x})

4. Evaluate :math:`f(\mathbf{x}_{\text{cand}})`.
5. Augment dataset:

   .. math::

      \mathcal{D}_{n+1}
      =
      \mathcal{D}_n
      \cup
      \{(\mathbf{x}_{\text{cand}}, f(\mathbf{x}_{\text{cand}}))\}.

6. Repeat until stopping condition is satisfied.

Acquisition Maximization
------------------------

The acquisition function encodes the exploration–exploitation trade-off.
It is optimized at each BO iteration to determine the next evaluation point.

In mfbo:

- Single-objective problems typically use gradient-based
  L-BFGS-B with multiple random restarts.
- Multi-objective problems may use evolutionary strategies
  when appropriate.

Because acquisition functions are generally non-convex,
multi-start strategies are recommended to mitigate local optima.

Stopping Criteria
-----------------

Bayesian optimization may terminate based on:

- Maximum number of evaluations
- Budget constraints (e.g., high-fidelity cost)
- Convergence of acquisition values
- Error-based criteria when ground truth is available

For benchmarking settings where the true optimum is known,
relative error metrics such as Mean Absolute Relative Error (MARE)
may be used to define convergence thresholds.

Recommended Solution
--------------------

After termination, a final recommended design may be obtained by:

1. Minimizing the surrogate model

   .. math::

      \hat{\mathbf{x}}^*
      =
      \arg\min_{\mathbf{x} \in \mathcal{X}}
      \hat{f}(\mathbf{x})

2. Evaluating the true objective at
   :math:`\hat{\mathbf{x}}^*`.

This final evaluation provides an estimate of the achieved optimum
under the evaluation budget.

Multi-Objective Extension
-------------------------

For vector-valued objectives

.. math::

   \mathbf{f}(\mathbf{x})
   =
   (f_1(\mathbf{x}), \dots, f_M(\mathbf{x})),

Bayesian optimization seeks to approximate the Pareto frontier.

In this case:

- The surrogate models each objective.
- Acquisition functions are based on hypervolume improvement.
- The result is a set of non-dominated solutions.

Further details are provided in the
:doc:`acquisition` and :doc:`optimizers` sections.

Key Properties of Bayesian Optimization
----------------------------------------

Bayesian optimization is particularly suitable when:

- Evaluations are expensive
- The objective is non-convex or noisy
- Gradient information is unavailable
- The number of design variables is moderate

However, BO performance may degrade in very high-dimensional spaces
without additional structure or dimensionality reduction techniques.

Summary
-------

Bayesian optimization in mfbo consists of:

- A probabilistic surrogate model
- An acquisition function
- An inner optimizer for acquisition maximization
- A stopping rule based on budget or convergence

Together, these components form a modular framework for
single-objective, multi-objective, single-fidelity,
and multi-fidelity optimization problems.