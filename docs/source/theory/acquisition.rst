Acquisition Functions
=====================

Acquisition functions guide the search in Bayesian optimization by
quantifying the utility of evaluating a candidate design point.
While the surrogate model approximates the objective function,
the acquisition function determines **where to evaluate next**.

In mfbo, acquisition functions are designed to:

- Balance exploration and exploitation
- Remain numerically stable across extreme regimes
- Support gradient-based optimization
- Handle both single-objective and multi-objective settings

Exploration vs Exploitation
---------------------------

An acquisition function :math:`\alpha(\mathbf{x})` is a deterministic
function constructed from the surrogate posterior. At iteration :math:`n`,
the next candidate point is selected as

.. math::

   \mathbf{x}_{\mathrm{cand}}
   =
   \arg\max_{\mathbf{x} \in \mathcal{X}}
   \alpha(\mathbf{x}).

Acquisition functions typically increase when:

- The predicted mean is promising (exploitation)
- The predictive uncertainty is large (exploration)

This trade-off enables Bayesian optimization to efficiently search
non-convex design spaces using a limited evaluation budget.

Single-Objective Acquisition
----------------------------

For single-objective minimization problems, mfbo employs the
**logarithmic Expected Improvement (logEI)** acquisition.

Classical Expected Improvement (EI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let

- :math:`\mu(\mathbf{x})` denote the posterior mean,
- :math:`\sigma(\mathbf{x})` denote the posterior standard deviation,
- :math:`y^*` denote the best observed objective value.

Define

.. math::

   z(\mathbf{x})
   =
   \frac{\mu(\mathbf{x}) - y^*}{\sigma(\mathbf{x})}.

The classical Expected Improvement is

.. math::

   \mathrm{EI}(\mathbf{x})
   =
   \sigma(\mathbf{x})
   \left[
   \phi(z) + z \Phi(z)
   \right],

where :math:`\phi` and :math:`\Phi` denote the standard normal
density and distribution functions.

Although EI is analytically tractable, direct evaluation can become
numerically unstable when :math:`z \ll 0`.

Logarithmic Expected Improvement (logEI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To improve numerical robustness, mfbo uses the logarithmic form:

.. math::

   \log \mathrm{EI}(\mathbf{x})
   =
   \log\!\left(
   \phi(z) + z \Phi(z)
   \right)
   +
   \log \sigma(\mathbf{x}).

This transformation prevents underflow when the improvement is
extremely small and stabilizes gradient-based acquisition optimization.

Numerical Stability
^^^^^^^^^^^^^^^^^^^

Direct computation of

.. math::

   \log\!\left(\phi(z) + z\Phi(z)\right)

may suffer from:

- Underflow in the extreme left tail
- Catastrophic cancellation when subtracting small numbers

To mitigate this, mfbo evaluates the expression piecewise using:

- Asymptotic approximations in the far tail
- Log-space evaluation
- Stable special-function representations

These techniques ensure well-behaved acquisition values and gradients
across all :math:`z`.

Multi-Objective Acquisition
---------------------------

For multi-objective problems, Bayesian optimization aims to expand
the Pareto frontier rather than optimize a single scalar objective.

Let

.. math::

   \mathbf{f}(\mathbf{x})
   =
   (f_1(\mathbf{x}), \dots, f_M(\mathbf{x})).

A common scalar performance indicator is the **hypervolume**,
which measures the dominated region relative to a reference point
:math:`\mathbf{r}`.

Hypervolume Improvement
^^^^^^^^^^^^^^^^^^^^^^^

Given a current Pareto set :math:`\mathcal{P}`, the hypervolume
improvement contributed by a candidate outcome :math:`\mathbf{y}`
is

.. math::

   \mathrm{HVI}(\mathbf{y})
   =
   \mathrm{HV}(\mathcal{P} \cup \{\mathbf{y}\}, \mathbf{r})
   -
   \mathrm{HV}(\mathcal{P}, \mathbf{r}).

Expected Hypervolume Improvement (EHVI) computes the expectation of
this quantity under the surrogate posterior.

q-Batch and Noisy Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

In practical settings:

- Evaluations may be performed in parallel (batch size :math:`q`)
- Observations may contain noise

To address these cases, mfbo supports:

- **qEHVI** (parallel batch acquisition)
- **qNEHVI** (noise-aware hypervolume improvement)

These methods evaluate the acquisition using Monte Carlo sampling
from the joint posterior distribution.

Logarithmic qNEHVI (qLogNEHVI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hypervolume improvements can become extremely small when candidate
points lie near the Pareto frontier. Direct subtraction may therefore
suffer from catastrophic cancellation.

To improve stability, mfbo applies a logarithmic transformation to
the hypervolume increment:

.. math::

   \Delta \log \mathrm{HV}
   =
   \log
   \left(
   \mathrm{HV}(\mathcal{P} \cup \tilde{\mathbf{Y}}, \mathbf{r})
   -
   \mathrm{HV}(\mathcal{P}, \mathbf{r})
   \right).

The qLogNEHVI acquisition is then estimated via Monte Carlo:

.. math::

   \alpha_{\mathrm{qLogNEHVI}}(\mathbf{X})
   =
   \mathbb{E}
   \left[
   \Delta \log \mathrm{HV}
   \right].

Benefits of the logarithmic formulation include:

- Improved numerical stability
- Reduced underflow in high-dimensional objective spaces
- Better gradient behavior for inner optimization

Acquisition Optimization
------------------------

Acquisition functions are typically non-convex.
mfbo therefore employs:

- Multi-start gradient-based optimization (e.g., L-BFGS-B)
- Randomized restarts
- Optional evolutionary methods for multi-objective settings

Because acquisition maximization is much cheaper than evaluating
the true objective, aggressive multi-start strategies are often
beneficial.

Practical Considerations
------------------------

When using acquisition functions:

- Always scale inputs to comparable ranges.
- Ensure objective values are properly normalized.
- Choose a meaningful reference point for hypervolume-based methods.
- Increase Monte Carlo sample size for more stable qNEHVI estimates
  at the cost of computation time.

Summary
-------

Acquisition functions in mfbo:

- Convert surrogate uncertainty into actionable decisions
- Support single-objective and multi-objective optimization
- Employ logarithmic transformations for numerical stability
- Are optimized via gradient-based multi-start methods

They form the decision-making core of Bayesian optimization.