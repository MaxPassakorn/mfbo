Modified Branin Function (2D, Single Objective)
===============================================

This tutorial demonstrates multi-fidelity Bayesian optimization on a
two-dimensional, single-objective benchmark based on the Branin function.
We evaluate surrogate accuracy and infill behavior under a fixed budget,
comparing multi-fidelity surrogates available in :mod:`mfbo`.

Compared models
---------------

We consider the following multi-fidelity surrogates:

- :class:`mfbo.gp.cokriging.CoKrigingAR1` (Gaussian-process AR(1) co-Kriging)
- :class:`mfbo.nn.ensembles.MFNNEnsemble`
- :class:`mfbo.nn.ensembles.Ada2MFEnsemble`
- :class:`mfbo.nn.ensembles.AGMFNetEnsemble`

All models are coupled to the same low-fidelity function :math:`f_L(x)` and
use :class:`botorch.acquisition.analytic.LogExpectedImprovement` to select
new infill points.

Problem definition
------------------

The classical (unnormalized) Branin function is:

.. math::

   \mathrm{Branin}(x_1, x_2) =
   \left(x_2 - \frac{5.1}{4\pi^2}x_1^2 + \frac{5}{\pi}x_1 - 6\right)^2
   + \left(10 - \frac{10}{8\pi}\right)\cos(x_1) - 44.81.

In this tutorial we optimize a normalized form on the unit box:

.. math::

   u = (u_1, u_2) \in [0,1]^2,

with an affine mapping:

.. math::

   x_1 = 15 u_1 - 5,\qquad x_2 = 15 u_2.

The high-fidelity objective is:

.. math::

   f(u) = \frac{\mathrm{Branin}(x_1, x_2)}{51.95},

and the low-fidelity approximation is:

.. math::

   f_L(u) = \frac{0.5\,\mathrm{Branin}(x_1, x_2) + 10(x_1 - 2.5)}{51.95}.

The normalization scale :math:`51.95` matches the provided experiment script
and keeps values in a convenient numerical range.

Setup
-----

.. code-block:: python

   import math
   import os

   import numpy as np
   import torch
   import matplotlib.pyplot as plt
   import matplotlib.ticker as mtick
   from matplotlib.patches import Patch

   from skopt.sampler import Lhs
   from skopt.space import Space

   from botorch.acquisition.analytic import LogExpectedImprovement
   from botorch.optim.optimize import optimize_acqf

   from mfbo.gp.cokriging import CoKrigingAR1
   from mfbo.nn.ensembles import MFNNEnsemble, Ada2MFEnsemble, AGMFNetEnsemble


   torch.manual_seed(42)
   torch.set_default_dtype(torch.float64)
   torch.set_default_device("cuda")   # or "cpu"


Objective functions
-------------------

.. code-block:: python

   def branin(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
       term1 = x2 - (5.1 * x1**2) / (4 * math.pi**2) + (5.0 * x1) / math.pi - 6.0
       term2 = 10.0 - 10.0 / (8 * math.pi)
       return term1**2 + term2 * torch.cos(x1) - 44.81


   def f(u: torch.Tensor) -> torch.Tensor:
       # High-fidelity: u in [0,1]^2 -> (x1,x2) in [-5,10]x[0,15]
       x1 = 15.0 * u[:, 0] - 5.0
       x2 = 15.0 * u[:, 1]
       y = branin(x1, x2) / 51.95
       return y.unsqueeze(-1)


   def low_f(u: torch.Tensor) -> torch.Tensor:
       x1 = 15.0 * u[..., 0] - 5.0
       x2 = 15.0 * u[..., 1]
       bias = 10.0 * (x1 - 2.5)
       y = (0.5 * branin(x1, x2) + bias) / 51.95
       return y.unsqueeze(-1)


Initial designs
---------------

We generate an initial high-fidelity design set :math:`X_e` using Latin
hypercube sampling (LHS) with ``n0 = 16`` points, matching your script.

We also generate a large low-fidelity support set :math:`X_c` with
``2000`` points, used by the co-Kriging model.

.. code-block:: python

   l_bounds = np.zeros(2, np.float64)
   u_bounds = np.ones(2, np.float64)
   space = Space(np.column_stack((l_bounds, u_bounds)))

   lhs = Lhs(lhs_type="centered", iterations=10_000_000)

   Xe = torch.tensor(lhs.generate(space.dimensions, n_samples=16))
   Xc = torch.tensor(lhs.generate(space.dimensions, n_samples=2000))

   n0 = Xe.shape[0]


Scaling
-------

Neural multi-fidelity models typically benefit from standardized outputs.
We standardize high-fidelity targets using mean / std computed from the
initial high-fidelity observations. The same scaler is applied to the
low-fidelity evaluations so that both are in a consistent normalized space.

.. code-block:: python

   class YScaler:
       def __init__(self, Y: torch.Tensor):
           self.mean = Y.mean(dim=0, keepdim=True)
           self.std = Y.std(dim=0, keepdim=True).clamp_min(1e-12)

       def encode(self, Y: torch.Tensor) -> torch.Tensor:
           return (Y - self.mean) / self.std

       def decode(self, Ys: torch.Tensor) -> torch.Tensor:
           return Ys * self.std + self.mean


   y0 = f(Xe)                  # [n0, 1]
   y_scaler = YScaler(y0)

   yc = low_f(Xc)              # [2000, 1]
   yc_s = y_scaler.encode(yc)  # scaled low-fidelity outputs


   def f_scaled(X: torch.Tensor) -> torch.Tensor:
       return y_scaler.encode(f(X))

   def low_f_scaled(X: torch.Tensor) -> torch.Tensor:
       return y_scaler.encode(low_f(X))


Evaluation grid for visualization
---------------------------------

For contour visualization we evaluate the surrogates on a dense
``200 x 200`` grid over :math:`[0,1]^2`.

.. code-block:: python

   x1_val = torch.linspace(0, 1, 200)
   x2_val = torch.linspace(0, 1, 200)
   x1, x2 = torch.meshgrid(x1_val, x2_val, indexing="xy")
   X = torch.column_stack([x1.ravel(), x2.ravel()])   # [40000, 2]

   y_true = f(X)                                      # [40000, 1]
   y_true_2d = y_true.squeeze(-1).detach().cpu().numpy().reshape(x1.shape)


Acquisition optimization helper
-------------------------------

We optimize Log Expected Improvement over the unit box.

.. code-block:: python

   def opt_log_ei(acq, dim: int):
       bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
       candidate, _ = optimize_acqf(
           acq,
           bounds=bounds,
           q=1,
           num_restarts=50,
           raw_samples=1000,
           return_best_only=True,
       )
       return candidate


Co-Kriging prediction helper
----------------------------

The co-Kriging model exposes ``predict(X) -> (mean, std)``. For large grids,
we evaluate in batches to avoid GPU/CPU memory spikes.

.. code-block:: python

   def batched_predict(model, X: torch.Tensor, batch_size: int = 1000):
       model.eval()
       means, stds = [], []
       with torch.no_grad():
           for i in range(0, X.shape[0], batch_size):
               Xi = X[i : i + batch_size]
               m, s = model.predict(Xi)
               means.append(m)
               stds.append(s)
       return torch.cat(means, dim=0), torch.cat(stds, dim=0)


Bayesian optimization loop
--------------------------

We run multi-fidelity Bayesian optimization for ``infill_num = 2`` iterations.
Each method maintains its own training set and selects its own candidate.

.. code-block:: python

   infill_num = 2

   X_cokrig = Xe.clone()
   X_mfnn   = Xe.clone()
   X_ada2mf = Xe.clone()
   X_agmf   = Xe.clone()

   for i in range(infill_num + 1):

       # ----- co-Kriging (AR1) -----
       y_h = f_scaled(X_cokrig)

       cokrig = CoKrigingAR1(Xc, yc_s, X_cokrig, y_h)

       # optional noise init (matches your experiment script)
       for lik in cokrig.lik_l_list:
           lik.noise_covar.initialize(noise=1e-4)
       for lik in cokrig.lik_h_list:
           lik.noise_covar.initialize(noise=1e-4)

       cokrig.fit(iters_per_stage=120, stages=3, lr_low=0.01, lr_delta=0.01, verbose=False)

       mu_ck_s, _ = batched_predict(cokrig, X)
       mu_ck = y_scaler.decode(mu_ck_s)
       mu_ck_2d = mu_ck.squeeze(-1).detach().cpu().numpy().reshape(x1.shape)

       # ----- MFNN ensemble -----
       y_mfnn = f_scaled(X_mfnn)

       mfnn = MFNNEnsemble(
           X_train=X_mfnn,
           y_train=y_mfnn,
           low_fn=low_f_scaled,
           ensemble_size=50,
           hid_features=5,
           n_hid_layers=2,
       )
       mfnn.fit(epochs=6000, optimizer="AdamW", lr=1e-3)

       mu_mfnn_s = mfnn(X).mean(dim=1).squeeze(0)      # [N,1]
       mu_mfnn = y_scaler.decode(mu_mfnn_s)
       mu_mfnn_2d = mu_mfnn.squeeze(-1).detach().cpu().numpy().reshape(x1.shape)

       # ----- Ada2MF ensemble -----
       y_ada = f_scaled(X_ada2mf)

       ada = Ada2MFEnsemble(
           X_train=X_ada2mf,
           y_train=y_ada,
           low_fn=low_f_scaled,
           ensemble_size=50,
           hid_features=5,
           n_layers=2,
       )
       ada.fit(epochs=3000, optimizer="AdamW", lr=1e-3)

       mu_ada_s = ada(X).mean(dim=1).squeeze(0)
       mu_ada = y_scaler.decode(mu_ada_s)
       mu_ada_2d = mu_ada.squeeze(-1).detach().cpu().numpy().reshape(x1.shape)

       # ----- AGMF-Net ensemble -----
       y_ag = f_scaled(X_agmf)

       agmf = AGMFNetEnsemble(
           X_train=X_agmf,
           y_train=y_ag,
           low_fn=low_f_scaled,
           ensemble_size=50,
           hid_features=5,
           n_layers=2,
       )
       agmf.fit(epochs=15000, optimizer="AdamW", lr=1e-3)

       mu_ag_s = agmf(X).mean(dim=1).squeeze(0)
       mu_ag = y_scaler.decode(mu_ag_s)
       mu_ag_2d = mu_ag.squeeze(-1).detach().cpu().numpy().reshape(x1.shape)

       # ----- acquisition + candidate selection -----
       # best_f is computed in the *scaled* space for neural models
       acq_ck = LogExpectedImprovement(model=cokrig, best_f=y_h.min(), maximize=False)
       acq_mf = LogExpectedImprovement(model=mfnn,   best_f=y_mfnn.min(), maximize=False)
       acq_ad = LogExpectedImprovement(model=ada,    best_f=y_ada.min(), maximize=False)
       acq_ag = LogExpectedImprovement(model=agmf,   best_f=y_ag.min(), maximize=False)

       cand_ck = opt_log_ei(acq_ck, dim=2)
       cand_mf = opt_log_ei(acq_mf, dim=2)
       cand_ad = opt_log_ei(acq_ad, dim=2)
       cand_ag = opt_log_ei(acq_ag, dim=2)

       # ----- update designs -----
       X_cokrig = torch.cat([X_cokrig, cand_ck], dim=0)
       X_mfnn   = torch.cat([X_mfnn,   cand_mf], dim=0)
       X_ada2mf = torch.cat([X_ada2mf, cand_ad], dim=0)
       X_agmf   = torch.cat([X_agmf,   cand_ag], dim=0)


Visualization
-------------

Your script visualizes surrogate means as filled contour plots, and overlays:

- initial observations :math:`X_e` (black edge)
- infill points selected by each method

Below is the plotting pattern, using a shared color scale for fair comparison.

.. code-block:: python

   import numpy as np

   fig, axes = plt.subplots(1, 5, figsize=(10, 2.4), gridspec_kw={"wspace": 0.15})
   ax1, ax2, ax3, ax4, ax5 = axes

   global_min = min(mu_ck_2d.min(), mu_mfnn_2d.min(), mu_ada_2d.min(), mu_ag_2d.min(), y_true_2d.min())
   global_max = max(mu_ck_2d.max(), mu_mfnn_2d.max(), mu_ada_2d.max(), mu_ag_2d.max(), y_true_2d.max())
   levels = np.linspace(global_min, global_max, 30)

   cs0 = ax1.contourf(x1.cpu(), x2.cpu(), mu_ck_2d, levels=levels, vmin=global_min, vmax=global_max, cmap="viridis_r")
   ax1.set_title("co-Kriging")

   ax2.contourf(x1.cpu(), x2.cpu(), mu_mfnn_2d, levels=levels, vmin=global_min, vmax=global_max, cmap="viridis_r")
   ax2.set_title("MFNN")

   ax3.contourf(x1.cpu(), x2.cpu(), mu_ada_2d, levels=levels, vmin=global_min, vmax=global_max, cmap="viridis_r")
   ax3.set_title("Ada2MF")

   ax4.contourf(x1.cpu(), x2.cpu(), mu_ag_2d, levels=levels, vmin=global_min, vmax=global_max, cmap="viridis_r")
   ax4.set_title("AGMF-Net")

   ax5.contourf(x1.cpu(), x2.cpu(), y_true_2d, levels=levels, vmin=global_min, vmax=global_max, cmap="viridis_r")
   ax5.set_title("True Branin")

   for ax in axes:
       ax.set_xlabel("$x_1$")
       ax.set_xlim(0, 1)
       ax.set_ylim(0, 1)
       ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

   # shared colorbar
   fig.colorbar(cs0, ax=axes, fraction=0.07, pad=0.22, format="%.2f", orientation="horizontal")
   plt.show()


Summary
-------

In this tutorial, we:

- Defined a normalized 2D Branin benchmark on :math:`[0,1]^2`
- Constructed a biased low-fidelity approximation
- Trained and compared co-Kriging, MFNN, Ada2MF, and AGMF-Net surrogates
- Ran a Bayesian optimization loop using Log Expected Improvement
- Visualized the surrogate landscapes and infill sampling behavior

The next tutorial extends the workflow to three dimensions using
the Hartmann-3D function.