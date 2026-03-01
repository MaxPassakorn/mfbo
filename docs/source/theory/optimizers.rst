Optimization Algorithms
=======================

Bayesian optimization involves two distinct optimization tasks:

1. **Acquisition maximization**, performed at every iteration to select the next evaluation point.
2. **Surrogate minimization**, optionally performed at termination to recommend a final design.

Because these problems may differ in structure (single-objective vs multi-objective,
gradient availability, dimensionality), mfbo provides complementary optimization
algorithms suited to each setting.

This section describes the optimization strategies used internally.

Overview
--------

The optimization problem considered in mfbo takes the general form

.. math::

   \min_{\mathbf{x} \in \mathcal{X}} g(\mathbf{x}),

where:

- :math:`g(\mathbf{x})` may represent a surrogate model,
- or the negative acquisition function,
- and :math:`\mathcal{X}` is a bounded domain defined by box constraints.

For multi-objective problems, the task becomes

.. math::

   \min_{\mathbf{x} \in \mathcal{X}} \mathbf{g}(\mathbf{x}),
   \qquad
   \mathbf{g} \in \mathbb{R}^M,

where the goal is to approximate the Pareto frontier.

Bound-Constrained Optimization: L-BFGS-B
-----------------------------------------

For single-objective problems with available gradients,
mfbo uses the Limited-memory BFGS algorithm with bound constraints
(L-BFGS-B).

Problem Formulation
^^^^^^^^^^^^^^^^^^^

L-BFGS-B solves

.. math::

   \min_{\mathbf{x} \in \mathbb{R}^d}
   g(\mathbf{x})
   \quad
   \text{subject to}
   \quad
   \ell_j \le x_j \le u_j,
   \quad j = 1,\dots,d.

The lower and upper bounds define a hyper-rectangular feasible region.

In Bayesian optimization, this formulation applies to:

- Acquisition maximization (minimizing negative acquisition)
- Final surrogate minimization

Algorithm Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^

L-BFGS-B is a quasi-Newton method that:

- Approximates the inverse Hessian using a limited memory buffer
- Avoids storing full :math:`d \times d` matrices
- Uses projected gradients to enforce bound constraints

Instead of forming a full Hessian approximation,
the algorithm stores only the most recent :math:`m`
curvature correction pairs. This results in memory
complexity proportional to :math:`\mathcal{O}(md)`.

Iteration Structure
^^^^^^^^^^^^^^^^^^^

Each iteration conceptually involves:

1. **Projected gradient step**

   Variables at their bounds are identified, forming an active set.

2. **Subspace minimization**

   A quadratic approximation of the objective is minimized
   over free variables.

3. **Line search**

   A line search satisfying Wolfe conditions is performed
   to ensure sufficient decrease and curvature conditions.

Termination typically occurs when:

- The projected gradient norm becomes small
- The relative reduction in objective falls below tolerance

Why L-BFGS-B in mfbo?
^^^^^^^^^^^^^^^^^^^^^

L-BFGS-B is particularly suitable for acquisition optimization because:

- Acquisition functions are differentiable
- Design variables are box-constrained
- Dimensionality is moderate
- Fast local convergence is desirable

Since acquisition maximization is inexpensive compared
to objective evaluations, mfbo often performs
multi-start L-BFGS-B to mitigate local optima.

Multi-Objective Optimization: U-NSGA-III
----------------------------------------

For multi-objective surrogate optimization,
mfbo employs an evolutionary algorithm based on
the Unified NSGA-III framework (U-NSGA-III).

Problem Formulation
^^^^^^^^^^^^^^^^^^^

The multi-objective optimization problem is

.. math::

   \min_{\mathbf{x} \in \mathcal{X}}
   \mathbf{g}(\mathbf{x})
   =
   (g_1(\mathbf{x}), \dots, g_M(\mathbf{x})).

Instead of a single optimum, the solution is a set of
non-dominated points approximating the Pareto frontier.

Algorithm Overview
^^^^^^^^^^^^^^^^^^

U-NSGA-III is a population-based evolutionary algorithm that:

- Uses non-dominated sorting
- Maintains diversity via reference directions
- Applies niching-based selection

The algorithm operates without requiring gradient information.

Main steps include:

1. Initialize a population within bounds.
2. Evaluate objectives.
3. Perform non-dominated sorting.
4. Associate individuals with reference directions.
5. Apply niching selection to preserve diversity.
6. Generate offspring via crossover and mutation.
7. Repeat until convergence or maximum generations.

Reference Direction Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To maintain diversity across objectives,
reference directions are distributed on the unit simplex:

.. math::

   \sum_{m=1}^{M} z_m = 1,
   \qquad
   z_m \ge 0.

mfbo may generate these directions using energy-based
distribution methods that produce quasi-uniform spacing
without combinatorial constructions.

Why Evolutionary Methods?
^^^^^^^^^^^^^^^^^^^^^^^^^

Gradient-based methods are generally unsuitable for:

- Non-convex Pareto fronts
- Discontinuous dominance structures
- High-dimensional objective spaces

Evolutionary strategies provide:

- Global search capability
- Robustness to non-smoothness
- Simultaneous Pareto front approximation

Optimization in the BO Workflow
--------------------------------

Within Bayesian optimization:

- L-BFGS-B is typically used to maximize acquisition functions.
- U-NSGA-III may be used for surrogate-based multi-objective
  recommendation.
- Multi-start strategies improve robustness.
- Optimization cost is negligible compared to true objective evaluations.

Practical Recommendations
-------------------------

When using optimization routines in mfbo:

- Always normalize design variables to comparable ranges.
- Use multiple random restarts for acquisition optimization.
- Increase population size for high-dimensional multi-objective tasks.
- Monitor convergence tolerances to avoid premature termination.

Summary
-------

mfbo provides complementary optimization tools:

- **L-BFGS-B** for efficient bound-constrained local optimization
  in single-objective settings.
- **U-NSGA-III** for Pareto-front approximation in multi-objective
  problems.

Together, these algorithms enable robust optimization across
single- and multi-objective Bayesian optimization workflows.