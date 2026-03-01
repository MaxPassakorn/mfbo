Forrester Function (1D, Single Objective)
=========================================

This tutorial demonstrates **single-objective Bayesian optimization** on the
classical **Forrester 1D** benchmark using both single-fidelity and multi-fidelity
surrogate models implemented in :mod:`mfbo`. We compare predictive accuracy,
uncertainty calibration, and acquisition behavior under a small evaluation budget.

We build and compare the following surrogates:

- **Kriging (single-fidelity GP)** using BoTorch :class:`botorch.models.SingleTaskGP`
  with a custom kernel (:class:`mfbo.gp.kernels.kriging.Kriging`) wrapped in
  :class:`gpytorch.kernels.ScaleKernel`.
- **AR(1) co-Kriging** using :class:`mfbo.gp.cokriging.CoKrigingAR1` with an explicit
  low-fidelity dataset.
- **Neural ensembles**:
  - :class:`mfbo.nn.ensembles.MLPEnsemble` (single-fidelity NN ensemble)
  - :class:`mfbo.nn.ensembles.MFNNEnsemble`
  - :class:`mfbo.nn.ensembles.Ada2MFEnsemble`
  - :class:`mfbo.nn.ensembles.AGMFNetEnsemble`

All methods use the same acquisition function:

- :class:`botorch.acquisition.analytic.LogExpectedImprovement`
  (Log-EI, minimized objective)

At each iteration, each method proposes one new point by maximizing Log-EI, and we
track:

- **RMSE** against the true function over a dense test grid
- **:math:`r^2`** score over the same grid
- **MARE at the known global minimizer** :math:`x^*` (local, high-value diagnostic)

Problem definition
------------------

High-fidelity function
^^^^^^^^^^^^^^^^^^^^^^

The Forrester function on the unit interval is:

.. math::

   f(x) = (6x - 2)^2 \sin(12x - 4), \qquad x \in [0,1].

In code:

.. code-block:: python

   def f(x):
       return (6*x-2)**2*torch.sin(12*x-4)

Low-fidelity approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the standard biased low-fidelity proxy:

.. math::

   f_L(x) = 0.5 f(x) + 10(x - 0.5) - 5.

This approximation is correlated with :math:`f` but systematically biased,
which makes it useful for multi-fidelity learning.

.. code-block:: python

   def low_f(x):
       return 0.5*f(x)+10*(x-0.5)-5

Setup
-----

Imports and plotting defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   import math
   import torch
   import matplotlib as mpl
   import matplotlib.pyplot as plt

   from torcheval.metrics import MeanSquaredError
   from ignite.metrics.regression import MeanAbsoluteRelativeError
   from ignite.engine import Engine

   from botorch.models import SingleTaskGP
   from botorch.fit import fit_gpytorch_mll
   from gpytorch.mlls import ExactMarginalLogLikelihood
   from botorch.optim.fit import fit_gpytorch_mll_torch
   from gpytorch.kernels import ScaleKernel
   from botorch.models.transforms.outcome import Standardize
   from botorch.models.transforms.input import Normalize

   from botorch.acquisition.analytic import LogExpectedImprovement
   from botorch.optim.optimize import optimize_acqf

   from mfbo.gp.kernels.kriging import Kriging
   from mfbo.gp.cokriging import CoKrigingAR1
   from mfbo.nn.ensembles import (
       MLPEnsemble, MFNNEnsemble, Ada2MFEnsemble, AGMFNetEnsemble
   )

   mpl.rcParams.update({
       "font.size": 8,
       "axes.titlesize": 10,
       "axes.labelsize": 8,
       "xtick.labelsize": 7,
       "ytick.labelsize": 7,
       "legend.fontsize": 8,
       "figure.titlesize": 10,
   })

Reproducibility and device
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   torch.manual_seed(42)
   torch.set_default_dtype(torch.float64)
   torch.set_default_device("cuda")

Data generation
---------------

Initial design and test grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We start with only three high-fidelity observations (a very low-data regime):

.. math::

   X_e = \{0, 0.5, 1\}.

We evaluate performance on a dense grid of 200 points:

.. math::

   X_{\text{test}} = \{x_i\}_{i=1}^{200} \subset [0,1].

.. code-block:: python

   xe = torch.linspace(0, 1, 3)     # initial HF points
   x  = torch.linspace(0, 1, 200)   # evaluation grid

We evaluate:

- low-fidelity on the dense grid
- high-fidelity on the dense grid
- high-fidelity on the initial points

.. code-block:: python

   yc = low_f(x)    # LF on grid
   ye = f(x)        # HF on grid
   y  = f(xe)       # HF at initial points

Output scaling for neural and co-Kriging models
-----------------------------------------------

Why scale outputs?
^^^^^^^^^^^^^^^^^^

The neural surrogates (and the co-Kriging in your setup) are trained in a
standardized output space to:

- stabilize optimization
- keep losses well-scaled
- make uncertainty comparable across methods

We compute mean and std from the *initial high-fidelity observations* and reuse
that scaling consistently.

.. code-block:: python

   class YScaler:
       def __init__(self, Y: torch.Tensor):
           self.mean = Y.mean(dim=0, keepdim=True)
           self.std  = Y.std(dim=0, keepdim=True).clamp_min(1e-12)

       def encode(self, Y: torch.Tensor) -> torch.Tensor:
           return (Y - self.mean) / self.std

       def decode(self, Ys: torch.Tensor) -> torch.Tensor:
           return Ys * self.std + self.mean

       def stddecode(self, Ys: torch.Tensor) -> torch.Tensor:
           return Ys * self.std

   y_scaler = YScaler(y)
   yc_s = y_scaler.encode(yc)

Scaled wrappers
^^^^^^^^^^^^^^^

For convenience, we define:

- scaled high-fidelity targets for training
- scaled low-fidelity function handle for MFNN/Ada2MF/AGMFNet

.. code-block:: python

   def low_f_scaled(X: torch.Tensor) -> torch.Tensor:
       return y_scaler.encode(low_f(X))

   def f_scaled_X(X: torch.Tensor) -> torch.Tensor:
       return y_scaler.encode(f(X))

Acquisition optimization helper
-------------------------------

We maximize Log-EI to propose the next evaluation point.

.. code-block:: python

   def optLogEI(logEI):
       bounds = torch.stack([torch.zeros(1), torch.ones(1)])
       new_x, acq_val = optimize_acqf(
           logEI,
           bounds=bounds,
           q=1,
           num_restarts=500,
           raw_samples=10000,
       )
       return new_x, acq_val

Important notes:

- ``maximize=False`` is used when constructing ``LogExpectedImprovement``
  because we minimize :math:`f(x)`.
- We use many restarts and raw samples because 1D problems are cheap and this
  makes acquisition maximization robust.

Evaluation metrics
------------------

RMSE on the grid
^^^^^^^^^^^^^^^^

We compute RMSE between predicted mean and the true function on the dense grid:

.. math::

   \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2}.

We use :class:`torcheval.metrics.MeanSquaredError` and take its square root
implicitly by applying it to the already-flattened tensors (your script stores
the MSE tensor; if you want true RMSE, apply ``sqrt`` when reporting).

:math:`r^2` on the grid
^^^^^^^^^^^^^^^^^^^^^^^

Your custom :math:`r^2` implementation computes squared Pearson correlation:

.. code-block:: python

   def r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
       y_pred = y_pred.flatten().double()
       y_true = y_true.flatten().double()
       yp = y_pred - y_pred.mean()
       yt = y_true - y_true.mean()
       num = (yp * yt).sum()
       den = torch.sqrt((yp**2).sum() * (yt**2).sum())
       r = num / (den + 1e-12)
       return r**2

MARE at the global minimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also evaluate MARE at a known minimizer:

.. math::

   x^* \approx 0.7572487578\ldots

Your code computes MARE for a single point by wrapping Ignite's metric
(:class:`ignite.metrics.regression.MeanAbsoluteRelativeError`) and calling an
``Engine`` with a one-item batch.

.. code-block:: python

   xmin = torch.tensor([0.75724875784185587], dtype=torch.float64)

Models compared
---------------

1) Single-fidelity Kriging (GP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We fit a BoTorch :class:`botorch.models.SingleTaskGP` using:

- input normalization (:class:`botorch.models.transforms.input.Normalize`)
- output standardization (:class:`botorch.models.transforms.outcome.Standardize`)
- a custom kernel :class:`mfbo.gp.kernels.kriging.Kriging` inside a
  :class:`gpytorch.kernels.ScaleKernel`

.. code-block:: python

   x_train = X_krig.unsqueeze(-1)
   y_train = f(X_krig).unsqueeze(-1)

   covar_module = ScaleKernel(Kriging(power_init=1.5, ard_num_dims=x_train.size(-1)))

   krig_model = SingleTaskGP(
       train_X=x_train,
       train_Y=y_train,
       train_Yvar=torch.full_like(y_train, 1e-6),
       covar_module=covar_module,
       input_transform=Normalize(d=x_train.shape[-1]),
       outcome_transform=Standardize(m=1),
   )

   mll = ExactMarginalLogLikelihood(krig_model.likelihood, krig_model)

Training is performed by optimizing the exact marginal log-likelihood:

.. code-block:: python

   fit_gpytorch_mll(
       mll,
       optimizer=fit_gpytorch_mll_torch,
       optimizer_kwargs={"step_limit": total_iter},
   )

After fitting, the posterior mean and standard deviation are computed on the test
grid:

.. code-block:: python

   with torch.no_grad():
       post = krig_model.posterior(X_test)
   mu  = post.mean.squeeze(-1)
   std = post.variance.squeeze(-1).sqrt()

2) Multi-fidelity co-Kriging (AR1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We train :class:`mfbo.gp.cokriging.CoKrigingAR1` using:

- low-fidelity dataset: dense grid ``x`` with low-fidelity outputs ``yc_s``
- high-fidelity dataset: current design set ``X_cokrig`` with outputs ``y_cokrig``

Important: high-fidelity targets are passed in **scaled** space via ``f_scaled_X``.

.. code-block:: python

   y_cokrig = f_scaled_X(X_cokrig)
   cokrig_model = CoKrigingAR1(x, yc_s, X_cokrig, y_cokrig)

Noise initialization (helps optimization stability):

.. code-block:: python

   for lik in cokrig_model.lik_l_list:
       lik.noise_covar.initialize(noise=1e-4)
   for lik in cokrig_model.lik_h_list:
       lik.noise_covar.initialize(noise=1e-4)

Fit with staged optimization:

.. code-block:: python

   cokrig_model.fit(iters_per_stage=120, stages=3, lr_low=0.01, lr_delta=0.01, verbose=True)

Prediction returns mean and std in scaled space; we decode them back:

.. code-block:: python

   mu_s, std_s = cokrig_model.predict(X_test)
   mu  = y_scaler.decode(mu_s.squeeze(-1))
   std = y_scaler.stddecode(std_s.squeeze(-1))

3) Neural ensembles
^^^^^^^^^^^^^^^^^^^

All neural models are trained in the **scaled output space**:

- ``y_* = f_scaled_X(X_*)``

For each model, the ensemble forward pass returns multiple samples (ensemble members).
We compute:

- predictive mean: ensemble mean
- predictive std: ensemble standard deviation

and then decode both back to physical units.

MLPEnsemble (single-fidelity)
""""""""""""""""""""""""""""""

.. code-block:: python

   y_MLP = f_scaled_X(X_MLP)
   MLP_model = MLPEnsemble(
       X_train=X_MLP,
       y_train=y_MLP,
       ensemble_size=50,
       hid_features=5,
       n_hid_layers=2,
   )
   MLP_model.fit(epochs=1000, optimizer="AdamW", lr=1e-3)

MFNNEnsemble / Ada2MFEnsemble / AGMFNetEnsemble (multi-fidelity)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

These models incorporate the low-fidelity function handle ``low_f_scaled``:

.. code-block:: python

   MFNN_model = MFNNEnsemble(
       X_train=X_MFNN,
       low_fn=low_f_scaled,
       y_train=f_scaled_X(X_MFNN),
       ensemble_size=50,
       hid_features=5,
       n_hid_layers=2,
   )

   Ada2MF_model = Ada2MFEnsemble(
       X_train=X_Ada2MF,
       low_fn=low_f_scaled,
       y_train=f_scaled_X(X_Ada2MF),
       ensemble_size=50,
       hid_features=5,
       n_layers=2,
   )

   AGMF_model = AGMFNetEnsemble(
       X_train=X_AGMFNet,
       low_fn=low_f_scaled,
       y_train=f_scaled_X(X_AGMFNet),
       ensemble_size=50,
       hid_features=5,
       n_layers=2,
   )

Posterior mean/std for ensembles
"""""""""""""""""""""""""""""""""

Given outputs ``Y_out`` returned by an ensemble forward pass,
we compute:

.. math::

   \mu(x) = \mathbb{E}_e[\hat{y}_e(x)],\qquad
   \sigma(x) = \sqrt{\mathbb{V}_e[\hat{y}_e(x)]}.

In code (pattern repeated across models):

.. code-block:: python

   out = model(x.reshape(-1, 1))  # ensemble predictions in scaled space

   mean_s = out.mean(dim=1).squeeze(0).squeeze(-1)
   std_s  = out.std(dim=1).squeeze(0).squeeze(-1)

   mean = y_scaler.decode(mean_s)
   std  = y_scaler.stddecode(std_s)

Bayesian optimization loop
--------------------------

We maintain a separate design history for each method:

.. code-block:: python

   X_krig    = xe
   X_cokrig  = xe
   X_MLP     = xe
   X_MFNN    = xe
   X_Ada2MF  = xe
   X_AGMFNet = xe

At each iteration:

1. Fit each surrogate on its current dataset.
2. Predict mean and uncertainty over the grid ``x``.
3. Compute metrics (RMSE, :math:`r^2`, MARE at :math:`x^*`).
4. Build Log-EI for each model and optimize it to propose a new candidate.
5. Append the candidate to that method's design set.

Log-EI construction
^^^^^^^^^^^^^^^^^^^

For minimization, we set ``maximize=False`` and provide the current best observed
training value.

- For Kriging: best value from physical outputs ``y_krig.min()``
- For scaled models (co-Kriging and neural): best value in scaled space
  ``y_model.min()`` where ``y_model = f_scaled_X(X_model)``

.. code-block:: python

   krig_logEI = LogExpectedImprovement(model=krig_model, best_f=y_krig.min(), maximize=False)

   cokrig_logEI = LogExpectedImprovement(model=cokrig_model, best_f=y_cokrig.min(), maximize=False)

   MLP_logEI = LogExpectedImprovement(model=MLP_model, best_f=y_MLP.min(), maximize=False)

   MFNN_logEI = LogExpectedImprovement(model=MFNN_model, best_f=y_MFNN.min(), maximize=False)

   Ada2MF_logEI = LogExpectedImprovement(model=Ada2MF_model, best_f=y_Ada2MF.min(), maximize=False)

   AGMF_logEI = LogExpectedImprovement(model=AGMF_model, best_f=y_AGMFNet.min(), maximize=False)

Candidate selection and caching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make the tutorial deterministic and avoid retraining differences due to new
candidates, your script caches candidates to disk:

.. code-block:: python

   cand_path = f"Trained_model/candidates1_{i}.pt"
   if os.path.exists(cand_path):
       cands = torch.load(cand_path, weights_only=False)
       can_krig   = cands["krig"]
       can_cokrig = cands["cokrig"]
       ...
   else:
       can_krig   = optLogEI(krig_logEI)[0]
       can_cokrig = optLogEI(cokrig_logEI)[0]
       ...
       torch.save({...}, cand_path)

After selecting candidates, we update each dataset:

.. code-block:: python

   X_krig    = torch.cat([X_krig,    can_krig.view(-1)], dim=0)
   X_cokrig  = torch.cat([X_cokrig,  can_cokrig.view(-1)], dim=0)
   X_MLP     = torch.cat([X_MLP,     can_MLP.view(-1)], dim=0)
   X_MFNN    = torch.cat([X_MFNN,    can_MFNN.view(-1)], dim=0)
   X_Ada2MF  = torch.cat([X_Ada2MF,  can_Ada2MF.view(-1)], dim=0)
   X_AGMFNet = torch.cat([X_AGMFNet, can_AGMF.view(-1)], dim=0)

Visualization (Posterior and Acquisition)
-----------------------------------------

At each Bayesian optimization iteration, we generate a two-panel figure:

Visualization (Posterior and Acquisition)
-----------------------------------------

At each Bayesian optimization iteration, we generate a two-panel figure:

#. **Posterior plot (top/left panel)**

   - True function :math:`f(x)` on a dense grid
   - Low-fidelity approximation :math:`f_L(x)` on the same grid
   - Initial observations :math:`X_e`
   - For each surrogate: posterior mean :math:`\mu(x)` and a 95% uncertainty band
     :math:`\mu(x)\pm 2\sigma(x)`
   - The newly selected candidates from the previous iteration (one per method)

#. **Acquisition plot (top/right panel)**

   - Log Expected Improvement (Log-EI) values across the same grid, for each method.

This visualization makes it easy to see (i) where the surrogate is uncertain,
(ii) whether the mean matches the true function, and (iii) where Log-EI drives
sampling.

Test grid and cached numpy arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We evaluate all models on the same dense grid:

.. code-block:: python

   X_test = x.unsqueeze(-1)            # [200, 1]
   x_cpu  = x.detach().cpu().numpy()
   xe_cpu = xe.detach().cpu().numpy()
   ye_cpu = ye.detach().cpu().numpy()
   yc_cpu = yc.detach().cpu().numpy()

Kriging posterior mean and uncertainty (SingleTaskGP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the single-fidelity Kriging model (BoTorch/GPyTorch), the posterior is
computed via ``model.posterior(X_test)``. The mean and variance come directly
from the GP posterior:

.. code-block:: python

   krig_model.eval()
   krig_model.likelihood.eval()
   with torch.no_grad():
       post = krig_model.posterior(X_test)

   y_kriging = post.mean.squeeze(-1)                       # [200]
   std_krig  = post.variance.squeeze(-1).sqrt()            # [200]

We plot a 95% band using:

.. math::

   \mu(x)\pm 2\sigma(x).

.. code-block:: python

   krig_lower = y_kriging - 2.0 * std_krig
   krig_upper = y_kriging + 2.0 * std_krig

co-Kriging posterior mean and uncertainty (CoKrigingAR1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The AR(1) co-Kriging model in :mod:`mfbo` returns predictions in the *scaled*
output space. We decode both the mean and the standard deviation back to the
physical function units:

.. code-block:: python

   mu_s, std_s = cokrig_model.predict(X_test)              # scaled space
   y_cokriging = y_scaler.decode(mu_s.squeeze(-1))         # physical mean
   std_cokrig  = y_scaler.stddecode(std_s.squeeze(-1))     # physical std

   cokrig_lower = y_cokriging - 2.0 * std_cokrig
   cokrig_upper = y_cokriging + 2.0 * std_cokrig

Neural ensemble posterior mean and uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For neural ensembles, the forward pass returns multiple ensemble predictions.
We interpret the ensemble mean as the predictive mean, and the ensemble standard
deviation as epistemic uncertainty:

.. math::

   \mu(x)=\mathbb{E}_e[\hat{y}_e(x)],\qquad
   \sigma(x)=\sqrt{\mathbb{V}_e[\hat{y}_e(x)]}.

In your implementation, the returned tensor has an ensemble dimension, so we compute:

.. code-block:: python

   out = model(x.reshape(-1, 1))                 # scaled space
   mean_s = out.mean(dim=1).squeeze(0).squeeze(-1)
   std_s  = out.std(dim=1).squeeze(0).squeeze(-1)

   mean = y_scaler.decode(mean_s)                # physical mean
   std  = y_scaler.stddecode(std_s)              # physical std

   lower = mean - 2.0 * std
   upper = mean + 2.0 * std

This same pattern is used for MLP, MFNN, Ada2MF, and AGMF-Net.

Acquisition values on a grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BoTorch analytic acquisition functions expect inputs of shape
``[..., q, d]``. In 1D with ``q=1``, we pass ``x.unsqueeze(-2)``:

.. code-block:: python

   def acqpreplot(logEI, x):
       acq = logEI(x.unsqueeze(-2))   # shape [N, 1]
       return y_scaler.decode(acq).detach().cpu().numpy()

Then we compute Log-EI curves for all methods:

.. code-block:: python

   krig_acq  = acqpreplot(krig_logEI,  X_test)
   cokrig_acq = acqpreplot(cokrig_logEI, X_test)
   MLP_acq   = acqpreplot(MLP_logEI,   X_test)
   MFNN_acq  = acqpreplot(MFNN_logEI,  X_test)
   Ada2MF_acq = acqpreplot(Ada2MF_logEI, X_test)
   AGMF_acq  = acqpreplot(AGMF_logEI,  X_test)

Two-panel plot layout
^^^^^^^^^^^^^^^^^^^^^

We create a two-panel figure:

- left: function + posterior mean + uncertainty bands
- right: acquisition curves

.. code-block:: python

   fig, (ax1, ax2) = plt.subplots(
       1, 2, figsize=(7.48, 2.7), sharex=True,
       gridspec_kw={"hspace": 0.1}
   )

Posterior panel: what is plotted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first plot the baseline curves and initial points:

.. code-block:: python

   ax1.plot(xe_cpu, y_cpu, "o", label="Initial Observations")
   ax1.plot(x_cpu, yc_cpu, "--", label="Low-fidelity Function")
   ax1.plot(x_cpu, ye_cpu, "-",  label="True Function")

Then for each method, we add:

- uncertainty band with ``fill_between``
- posterior mean as a line

Example for Kriging and co-Kriging:

.. code-block:: python

   ax1.fill_between(x_cpu, krig_lower.cpu(), krig_upper.cpu(), alpha=0.15)
   ax1.plot(x_cpu, y_kriging.cpu(), "-", label="Kriging")

   ax1.fill_between(x_cpu, cokrig_lower.cpu(), cokrig_upper.cpu(), alpha=0.15)
   ax1.plot(x_cpu, y_cokriging.cpu(), "-", label="co-Kriging")

The same pattern is repeated for MLP, MFNN, Ada2MF, and AGMF-Net using their
decoded ``mean/lower/upper`` arrays.

Candidate markers
^^^^^^^^^^^^^^^^^

From the second iteration onward, we also mark the candidate selected by each
method (one point per method). Your script uses cached candidates from disk,
so the plot is deterministic:

.. code-block:: python

   if i != 0:
       ax1.plot(can_krig_cpu,   ycan_krig_cpu,   "o")
       ax1.plot(can_cokrig_cpu, ycan_cokrig_cpu, "o")
       ax1.plot(can_MLP_cpu,    ycan_MLP_cpu,    "o")
       ax1.plot(can_MFNN_cpu,   ycan_MFNN_cpu,   "o")
       ax1.plot(can_Ada2MF_cpu, ycan_Ada2MF_cpu, "o")
       ax1.plot(can_AGMF_cpu,   ycan_AGMF_cpu,   "o")

Acquisition panel
^^^^^^^^^^^^^^^^^

The right panel shows Log-EI curves for each method:

.. code-block:: python

   ax2.plot(x_cpu, krig_acq,   label="Kriging")
   ax2.plot(x_cpu, cokrig_acq, label="co-Kriging")
   ax2.plot(x_cpu, MLP_acq,    label="MLP")
   ax2.plot(x_cpu, MFNN_acq,   label="MFNN")
   ax2.plot(x_cpu, Ada2MF_acq, label="Ada2MF")
   ax2.plot(x_cpu, AGMF_acq,   label="AGMF-Net")

Axes formatting and legend
^^^^^^^^^^^^^^^^^^^^^^^^^^

We format axes to match your script:

.. code-block:: python

   ax1.set_xlabel("x")
   ax1.set_ylabel("y")
   ax1.set_ylim(-10, 21)
   ax1.grid(True)

   ax2.set_xlabel("x")
   ax2.set_ylabel("Log-EI")
   ax2.set_ylim(-50, 15)
   ax2.grid(True)

Finally, you collect legend handles from ``ax1`` and place a shared legend
at the top of the figure in multiple columns:

.. code-block:: python

   handles, labels = ax1.get_legend_handles_labels()
   ncols = math.ceil(len(labels) / 2)

   fig.legend(
       handles, labels,
       loc="upper center",
       bbox_transform=fig.transFigure,
       ncol=ncols,
       frameon=False,
       columnspacing=1.0,
   )
   fig.subplots_adjust(top=0.85)
   plt.show()

Summary
-------

In this tutorial, we:

- Defined the Forrester 1D benchmark and a biased low-fidelity proxy
- Constructed initial designs with only three high-fidelity evaluations
- Trained six surrogates: Kriging GP, co-Kriging AR1, and four NN ensemble methods
- Performed Bayesian optimization using **Log Expected Improvement**
- Compared methods using grid-based RMSE/:math:`r^2` and local MARE at the known optimum
- Visualized posterior mean/uncertainty and acquisition landscapes over iterations

The next tutorial extends this workflow to a 2D single-objective benchmark
(Modified Branin) with richer visualization and larger budgets.