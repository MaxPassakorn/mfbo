Practical Considerations
========================

While Bayesian optimization provides a principled framework for
sample-efficient optimization, practical performance depends heavily
on modeling choices, scaling, numerical stability, and optimization
settings.

This section summarizes implementation-level guidance for using mfbo
effectively in real-world problems.

Input Scaling
-------------

Always scale design variables to comparable ranges.

For bounded box constraints

.. math::

   \ell_j \le x_j \le u_j,

it is recommended to normalize inputs to

.. math::

   x_j^{(scaled)} \in [0,1].

Reasons:

- Improves conditioning of surrogate models
- Stabilizes gradient-based acquisition optimization
- Prevents anisotropic kernel distortion in Gaussian processes
- Improves neural network training dynamics

mfbo assumes properly scaled inputs unless otherwise specified.

Output Normalization
--------------------

Objective values may vary across several orders of magnitude.
Standardizing outputs improves both surrogate stability and acquisition behavior.

A typical normalization is:

.. math::

   y^{(scaled)} =
   \frac{y - \mu_y}{\sigma_y}.

Benefits:

- Stabilizes kernel hyperparameter estimation
- Prevents domination of large-magnitude objectives
- Improves acquisition optimization

For multi-objective problems, each objective should be normalized independently.

Handling Noise
--------------

In noisy settings:

.. math::

   y_i = f(\mathbf{x}_i) + \varepsilon_i,

where :math:`\varepsilon_i` represents observation noise.

Recommendations:

- Use surrogate models that explicitly model noise variance.
- Prefer noise-aware acquisition functions (e.g., qNEHVI).
- Avoid overfitting when using neural surrogates with small datasets.

Ignoring noise may lead to:

- Overconfident uncertainty estimates
- Premature convergence
- Poor exploration

Acquisition Optimization Strategy
----------------------------------

Acquisition functions are typically non-convex.

Recommended settings:

- Use multi-start L-BFGS-B.
- Randomly initialize starting points.
- Increase restarts in higher-dimensional problems.
- Use gradient-based optimization whenever gradients are available.

Because acquisition evaluation is inexpensive relative to objective evaluation,
aggressive multi-start strategies are generally beneficial.

Batch Optimization
------------------

For parallel evaluations with batch size :math:`q`:

- Use q-batch acquisition functions.
- Increase Monte Carlo sample count for stable estimates.
- Be aware that larger :math:`q` increases computational cost.

Parallelization is especially beneficial when:

- High-fidelity evaluations are extremely expensive.
- Hardware resources allow concurrent simulations.

Reference Point Selection (Multi-Objective)
-------------------------------------------

Hypervolume-based acquisition requires a reference point
:math:`\mathbf{r}` dominated by all feasible objective values.

Guidelines:

- Choose a point slightly worse than known objective ranges.
- Avoid selecting reference points too close to the Pareto frontier.
- Re-evaluate reference point if objective scales change.

An improper reference point may distort hypervolume improvement
and bias exploration.

Multi-Fidelity Considerations
-----------------------------

When using multi-fidelity models:

- Ensure correlation between fidelity levels exists.
- Avoid extreme mismatch between low- and high-fidelity outputs.
- Normalize both fidelity outputs consistently.
- Monitor for negative transfer effects.

Multi-fidelity modeling is most beneficial when:

- Low-fidelity data is inexpensive and abundant.
- High-fidelity evaluations are costly.
- The discrepancy is structured rather than random.

Dimensionality Considerations
-----------------------------

Bayesian optimization performance degrades as dimensionality increases.

Practical limits depend on surrogate choice:

- Gaussian processes: typically effective below ~20 dimensions.
- Neural surrogates: may handle higher dimensions but require more data.

For higher-dimensional problems:

- Consider dimensionality reduction.
- Use structured kernels.
- Increase initial sample size.

Stopping Criteria
-----------------

Common stopping rules include:

- Fixed evaluation budget
- Acquisition value stagnation
- Surrogate convergence
- Validation error threshold (for benchmark problems)

Avoid relying solely on surrogate predicted optimum
without final evaluation of the true objective.

Computational Cost Trade-offs
-----------------------------

The overall computational cost in Bayesian optimization includes:

- Surrogate training
- Acquisition evaluation
- Acquisition optimization
- True objective evaluation

In most engineering applications:

- True objective evaluation dominates cost.
- Surrogate training cost is secondary.
- Acquisition optimization cost is negligible.

Tuning should prioritize reducing expensive evaluations
rather than minimizing acquisition runtime.

Common Failure Modes
--------------------

Typical pitfalls include:

- Poor input scaling
- Insufficient initial design size
- Overconfident surrogate uncertainty
- Inadequate acquisition optimization
- Reference point misconfiguration
- Extremely small Monte Carlo sample counts for qNEHVI

Careful monitoring of surrogate predictions and acquisition behavior
can prevent most failure modes.

Summary
-------

For reliable performance in mfbo:

- Normalize inputs and outputs.
- Use multi-start acquisition optimization.
- Choose appropriate acquisition functions.
- Handle noise explicitly when present.
- Validate final recommended solutions.

Bayesian optimization is powerful but sensitive to modeling choices.
Thoughtful configuration significantly improves robustness and efficiency.