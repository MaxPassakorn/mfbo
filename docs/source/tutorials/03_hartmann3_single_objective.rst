Hartmann-3D Function (3D, Single Objective)
===========================================

This tutorial demonstrates **single-objective multi-fidelity Bayesian optimization**
on the **Hartmann-3D** benchmark defined on the unit hypercube :math:`[0,1]^3`.

We compare four surrogate models implemented in :mod:`mfbo`:

- :class:`mfbo.gp.cokriging.CoKrigingAR1` (AR(1) multi-fidelity co-Kriging)
- :class:`mfbo.nn.ensembles.MFNNEnsemble`
- :class:`mfbo.nn.ensembles.Ada2MFEnsemble`
- :class:`mfbo.nn.ensembles.AGMFNetEnsemble`

All methods use the same low-fidelity function :math:`f_L(x)` and select new points
by optimizing :class:`botorch.acquisition.analytic.LogExpectedImprovement` (LogEI).

Compared to the 2D Branin tutorial, this example highlights two practical issues
in 3D:

1. **Visualization** is no longer a simple contour plot, so we use **isosurfaces**
   (via marching cubes) to visualize surrogate landscapes.
2. **Forward evaluation** on dense 3D grids can be heavy, so we use **batched**
   prediction (for co-Kriging) and a memory-safe forward pass (for ensembles).

Problem definition
------------------

Hartmann-3D is a standard nonconvex test function with multiple local minima and
a known global minimum. On :math:`x \in [0,1]^3`, it is defined as

.. math::

   f(x) = - \sum_{i=1}^{4} \alpha_i
   \exp \left( - \sum_{j=1}^{3} A_{ij} (x_j - P_{ij})^2 \right).

We use the classical parameterization (same as your script):

.. math::

   A =
   \begin{bmatrix}
   3 & 10 & 30 \\
   0.1 & 10 & 35 \\
   3 & 10 & 30 \\
   0.1 & 10 & 35
   \end{bmatrix},
   \quad
   P = 10^{-4}
   \begin{bmatrix}
   3689 & 1170 & 2673 \\
   4699 & 4387 & 7470 \\
   1091 & 8732 & 5547 \\
   381  & 5743 & 8828
   \end{bmatrix},
   \quad
   \alpha = [1.0, 1.2, 3.0, 3.2].

In this tutorial:

- The **high-fidelity** function is :math:`f(x)` (Hartmann-3D itself).
- The **low-fidelity** function :math:`f_L(x)` is a *biased* variant that keeps
  strong correlation but is not identical to the high-fidelity model.

Setup
-----

.. code-block:: python

   import os
   import numpy as np
   import torch

   import matplotlib as mpl
   import matplotlib.pyplot as plt
   from matplotlib.colors import Normalize
   from matplotlib.cm import ScalarMappable
   from mpl_toolkits.mplot3d.art3d import Poly3DCollection

   from skopt.sampler import Lhs
   from skopt.space import Space
   from skimage.measure import marching_cubes

   from botorch.acquisition.analytic import LogExpectedImprovement
   from botorch.optim.optimize import optimize_acqf

   from torcheval.metrics import MeanSquaredError
   from ignite.metrics.regression import MeanAbsoluteRelativeError
   from ignite.engine import Engine

   from mfbo.gp.cokriging import CoKrigingAR1
   from mfbo.nn.ensembles import MFNNEnsemble, Ada2MFEnsemble, AGMFNetEnsemble


   torch.manual_seed(42)
   torch.set_default_dtype(torch.float64)
   torch.set_default_device("cuda")  # or "cpu"

Hartmann-3D implementation
--------------------------

We implement Hartmann-3D using tensors for :math:`A, P, \alpha`, and evaluate
the function in batch form.

.. code-block:: python

   _H3_A = torch.tensor([
       [3.0, 10.0, 30.0],
       [0.1, 10.0, 35.0],
       [3.0, 10.0, 30.0],
       [0.1, 10.0, 35.0],
   ], dtype=torch.float64)

   _H3_P = 1e-4 * torch.tensor([
       [3689.0, 1170.0, 2673.0],
       [4699.0, 4387.0, 7470.0],
       [1091.0, 8732.0, 5547.0],
       [381.0,  5743.0, 8828.0],
   ], dtype=torch.float64)

   _H3_alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float64)


   def hartmann3(u: torch.Tensor) -> torch.Tensor:
       x = u.view(-1, 3).to(dtype=torch.float64)
       diff = x.unsqueeze(1) - _H3_P              # [N, 4, 3]
       quad = (_H3_A * (diff ** 2)).sum(dim=-1)   # [N, 4]
       val  = -(_H3_alpha * torch.exp(-quad)).sum(dim=-1)  # [N]
       return val.view(u.shape[:-1] + (1,))       # [..., 1]


   def f(u: torch.Tensor) -> torch.Tensor:
       # High-fidelity
       return hartmann3(u)

Low-fidelity function
---------------------

To construct a smooth, biased low-fidelity approximation, we modify the location
parameter :math:`P` (shrink it toward the origin). This changes the effective
peaks/valleys while preserving a similar global structure.

.. code-block:: python

   def low_f(u: torch.Tensor) -> torch.Tensor:
       x = u.view(-1, 3).to(dtype=torch.float64)

       # Bias: move P closer to origin
       P_adj = 0.375 * _H3_P

       diff = x.unsqueeze(1) - P_adj
       quad = (_H3_A * diff.pow(2)).sum(-1)
       y = -(_H3_alpha * torch.exp(-quad)).sum(-1)
       return y.view(u.shape[:-1] + (1,))

Initial designs
---------------

We use **Latin hypercube sampling (LHS)** to generate:

- an initial high-fidelity set :math:`X_e` with **30** points
- a large low-fidelity support set :math:`X_c` with **2000** points

The low-fidelity support is used by the co-Kriging model as a fixed background
dataset (as in your experiments).

We also cache sampled designs to disk, so rerunning the tutorial yields the same
design points.

.. code-block:: python

   l_bounds = np.zeros(3, np.float64)
   u_bounds = np.ones(3, np.float64)
   space = Space(np.column_stack((l_bounds, u_bounds)))

   lhs_e = Lhs(lhs_type="centered", iterations=100_000_000)
   lhs_c = Lhs(lhs_type="centered", iterations=1_000_000)

   Xe_path = "Trained_Model/Bench3-30_Xe"
   Xc_path = "Trained_Model/Bench3-Xc"

   if os.path.exists(Xe_path):
       Xe = torch.load(Xe_path, weights_only=False)
   else:
       Xe = torch.tensor(lhs_e.generate(space.dimensions, n_samples=30))
       torch.save(Xe, Xe_path)

   if os.path.exists(Xc_path):
       Xc = torch.load(Xc_path, weights_only=False)
   else:
       Xc = torch.tensor(lhs_c.generate(space.dimensions, n_samples=2000))
       torch.save(Xc, Xc_path)

   n0 = Xe.shape[0]

Scaling (standardization)
-------------------------

Neural-network surrogates are typically more stable if targets are standardized.
We compute mean and standard deviation from the **initial high-fidelity outputs**
and apply the same transformation to both:

- high-fidelity targets :math:`f(X)`
- low-fidelity targets :math:`f_L(X)`

This ensures both fidelities live in the same standardized output space.

.. code-block:: python

   class YScaler:
       def __init__(self, Y: torch.Tensor):
           self.mean = Y.mean(dim=0, keepdim=True)
           self.std  = Y.std(dim=0, keepdim=True).clamp_min(1e-12)

       def encode(self, Y: torch.Tensor) -> torch.Tensor:
           return (Y - self.mean) / self.std

       def decode(self, Ys: torch.Tensor) -> torch.Tensor:
           return Ys * self.std + self.mean


   y0 = f(Xe)              # [n0, 1]
   y_scaler = YScaler(y0)

   yc   = low_f(Xc)        # [2000, 1]
   yc_s = y_scaler.encode(yc)


   def f_scaled_X(X: torch.Tensor) -> torch.Tensor:
       return y_scaler.encode(f(X))

   def low_f_scaled(X: torch.Tensor) -> torch.Tensor:
       return y_scaler.encode(low_f(X))

Evaluation grid for visualization
---------------------------------

To visualize a 3D function, we evaluate it on a structured grid

- :math:`N_1 = N_2 = N_3 = 30`, so :math:`30^3 = 27{,}000` points
- the grid spans :math:`[0,1]^3`

.. code-block:: python

   N1 = N2 = N3 = 30
   x1_val = torch.linspace(0, 1, N1, dtype=torch.float64)
   x2_val = torch.linspace(0, 1, N2, dtype=torch.float64)
   x3_val = torch.linspace(0, 1, N3, dtype=torch.float64)

   x1, x2, x3 = torch.meshgrid(x1_val, x2_val, x3_val, indexing="ij")
   X = torch.stack([x1, x2, x3], dim=-1).view(-1, 3)

   y_true = f(X)   # [N1*N2*N3, 1]

Isosurface visualization (marching cubes)
-----------------------------------------

In 2D we could use contour plots, but in 3D we typically visualize the function
with **isosurfaces**. We use marching cubes to compute polygon meshes for several
iso-levels chosen as percentiles of the true function values. Percentiles help
avoid empty surfaces.

.. code-block:: python

   from skimage.measure import marching_cubes
   from matplotlib.colors import Normalize
   from mpl_toolkits.mplot3d.art3d import Poly3DCollection

   y_vol = y_true.detach().cpu().numpy().reshape(N1, N2, N3).squeeze()

   X1 = x1_val.detach().cpu().numpy()
   X2 = x2_val.detach().cpu().numpy()
   X3 = x3_val.detach().cpu().numpy()

   d1 = float(X1[1] - X1[0])
   d2 = float(X2[1] - X2[0])
   d3 = float(X3[1] - X3[0])

   levels = np.percentile(y_vol, [10, 25, 40, 60, 80])

   fig = plt.figure(figsize=(3.74, 3.74))
   ax  = fig.add_subplot(111, projection="3d")

   norm = Normalize(vmin=y_vol.min(), vmax=y_vol.max())
   cmap = plt.cm.viridis

   for lv in levels:
       verts, faces, _, _ = marching_cubes(y_vol, level=lv, spacing=(d1, d2, d3))
       origin = np.array([X1[0], X2[0], X3[0]])
       verts = verts + origin

       mesh = Poly3DCollection(
           verts[faces], facecolors=cmap(norm(lv)), edgecolor="none", alpha=0.5
       )
       ax.add_collection3d(mesh)

   Xe_cpu = Xe.detach().cpu().numpy()
   ax.scatter(Xe_cpu[:,0], Xe_cpu[:,1], Xe_cpu[:,2],
              s=30, c="k", depthshade=False, label="Initial Observations")

   ax.set_xlim(X1.min(), X1.max())
   ax.set_ylim(X2.min(), X2.max())
   ax.set_zlim(X3.min(), X3.max())
   ax.set_xlabel("$x_1$")
   ax.set_ylabel("$x_2$")
   ax.set_zlabel("$x_3$")

   try:
       ax.set_box_aspect((1, 1, 1))
   except Exception:
       pass

   m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
   m.set_array(y_vol)
   fig.colorbar(m, ax=ax, shrink=0.6, pad=0.15, label="f(x)")

   plt.tight_layout()
   plt.savefig("Max_thesis/figures/hartmann3d.pdf", bbox_inches="tight")
   plt.show()

Surrogate prediction helpers
----------------------------

Co-Kriging provides a ``predict`` method and is typically used on large
evaluation grids. We evaluate it in batches to avoid GPU/CPU memory spikes.

.. code-block:: python

   def batched_predict(model, X, batch_size=1000):
       model.eval()
       means, stds = [], []
       with torch.no_grad():
           for i in range(0, X.shape[0], batch_size):
               Xi = X[i : i + batch_size]
               m, s = model.predict(Xi)
               means.append(m)
               stds.append(s)
       return torch.cat(means, dim=0), torch.cat(stds, dim=0)

For neural ensembles, the raw forward output may have a shape like
``[B, E, 1]`` (batch, ensemble members, output). The function below performs a
memory-safe batched forward pass and returns a mean prediction on CPU.

.. code-block:: python

   def _module_device_dtype(m, fallback_device=None, fallback_dtype=None):
       for p in m.parameters(recurse=True):
           return p.device, p.dtype
       for b in m.buffers(recurse=True):
           return b.device, b.dtype
       return (
           fallback_device if fallback_device is not None else torch.device("cpu"),
           fallback_dtype  if fallback_dtype  is not None else torch.get_default_dtype()
       )

   def ensemble_mean_batched(model, X, batch_size=4096):
       run_device, run_dtype = _module_device_dtype(
           model, fallback_device=X.device, fallback_dtype=X.dtype
       )
       model.eval()
       outs = []
       for i in range(0, X.shape[0], batch_size):
           Xi = X[i:i+batch_size].to(run_device, dtype=run_dtype, non_blocking=True)
           Yi = model(Xi)  # e.g. [B, E, 1]
           m  = Yi.mean(dim=tuple(range(1, Yi.dim())), keepdim=True)  # -> [B,1]
           outs.append(m.cpu())
           del Xi, Yi, m
           if run_device.type == "cuda":
               torch.cuda.empty_cache()
       return torch.cat(outs, dim=0)

Acquisition optimization (LogEI)
--------------------------------

We optimize :class:`botorch.acquisition.analytic.LogExpectedImprovement` over the
unit box :math:`[0,1]^3`:

.. code-block:: python

   def optLogEI(acq, dim: int, return_best_only: bool = True):
       bounds = torch.stack([
           torch.zeros(dim, dtype=torch.float64),
           torch.ones(dim, dtype=torch.float64),
       ])
       new_x, _ = optimize_acqf(
           acq,
           bounds=bounds,
           q=1,
           num_restarts=500,
           raw_samples=10000,
           return_best_only=return_best_only,
       )
       return new_x

Metrics
-------

We track surrogate fidelity over the full evaluation grid:

- **RMSE** (MeanSquaredError, square-rooted RMSE in your printouts)
- **:math:`r^2`** computed as squared Pearson correlation
- **MARE at the known minimizer** (single-point relative error)

The global minimizer is set to a known Hartmann-3D optimum:

.. code-block:: python

   xmin = torch.tensor([[0.114614, 0.555649, 0.852547]], dtype=torch.float64)

The :math:`r^2` function used in your experiments:

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

Bayesian optimization loop
--------------------------

We run **15 infill iterations** (plus the initial state at :math:`i=0`), and each
surrogate maintains its own growing training set:

- :math:`X_{\text{coKrig}}`
- :math:`X_{\text{MFNN}}`
- :math:`X_{\text{Ada2MF}}`
- :math:`X_{\text{AGMF}}`

At each iteration, we:

1. Fit each surrogate using current data.
2. Predict the mean response on the 3D grid.
3. Compute metrics vs. the true function.
4. Optimize LogEI to obtain one new infill point per method.
5. Append the candidate to that method’s design set.

.. code-block:: python

   methods = ["cokriging", "mfnn", "ada2mf", "agmf"]
   rmse_scores = {m: [] for m in methods}
   r2_scores   = {m: [] for m in methods}
   mare_y      = {m: [] for m in methods}

   infill_num = 15

   X_cokrig   = Xe.clone()
   X_MFNN     = Xe.clone()
   X_Ada2MF   = Xe.clone()
   X_AGMFNet  = Xe.clone()

   for i in range(infill_num + 1):

       # ----- co-Kriging -----
       y_cokrig = f_scaled_X(X_cokrig)
       cokrig_model = CoKrigingAR1(Xc, yc, X_cokrig, y_cokrig)

       for lik in cokrig_model.lik_l_list:
           lik.noise_covar.initialize(noise=1e-4)
       for lik in cokrig_model.lik_h_list:
           lik.noise_covar.initialize(noise=1e-4)

       cokrig_model.fit(iters_per_stage=120, stages=3, lr_low=0.01, lr_delta=0.01, verbose=True)

       mu_ck_s, _ = batched_predict(cokrig_model, X, batch_size=1000)
       mu_ck = y_scaler.decode(mu_ck_s).view(-1)

       # ----- MFNN -----
       y_MFNN = f_scaled_X(X_MFNN)
       MFNN_model = MFNNEnsemble(
           X_train=X_MFNN,
           low_fn=low_f_scaled,
           y_train=y_MFNN,
           ensemble_size=50,
           hid_features=5,
           n_hid_layers=2,
       )
       MFNN_model.fit(epochs=8000, optimizer="AdamW", lr=1e-3)
       mu_mfnn = y_scaler.decode(MFNN_model(X).mean(dim=1).squeeze(0)).view(-1)

       # ----- Ada2MF -----
       y_Ada2MF = f_scaled_X(X_Ada2MF)
       Ada2MF_model = Ada2MFEnsemble(
           X_train=X_Ada2MF,
           low_fn=low_f_scaled,
           y_train=y_Ada2MF,
           ensemble_size=50,
           hid_features=5,
           n_layers=2,
       )
       Ada2MF_model.fit(epochs=4000, optimizer="AdamW", lr=1e-3)
       mu_ada = y_scaler.decode(Ada2MF_model(X).mean(dim=1).squeeze(0)).view(-1)

       # ----- AGMF-Net -----
       y_AGMFNet = f_scaled_X(X_AGMFNet)
       AGMF_model = AGMFNetEnsemble(
           X_train=X_AGMFNet,
           low_fn=low_f_scaled,
           y_train=y_AGMFNet,
           ensemble_size=50,
           hid_features=5,
           n_layers=2,
       )
       AGMF_model.fit(epochs=12000, optimizer="AdamW", lr=1e-3)
       mu_ag = y_scaler.decode(AGMF_model(X).mean(dim=1).squeeze(0)).view(-1)

       # ----- metrics on the full grid -----
       target_1d = f(X).view(-1)

       preds = {"cokriging": mu_ck, "mfnn": mu_mfnn, "ada2mf": mu_ada, "agmf": mu_ag}

       for name, y_pred in preds.items():
           rmse_metric = MeanSquaredError(device="cuda")
           rmse_metric.update(y_pred, target_1d)
           rmse_scores[name].append(rmse_metric.compute())

           r2_scores[name].append(r2(y_pred, target_1d))

       # ----- LogEI candidates -----
       cokrig_logEI = LogExpectedImprovement(model=cokrig_model, best_f=y_cokrig.min(), maximize=False)
       MFNN_logEI   = LogExpectedImprovement(model=MFNN_model,   best_f=y_MFNN.min(),   maximize=False)
       Ada2MF_logEI = LogExpectedImprovement(model=Ada2MF_model, best_f=y_Ada2MF.min(), maximize=False)
       AGMF_logEI   = LogExpectedImprovement(model=AGMF_model,   best_f=y_AGMFNet.min(), maximize=False)

       can_cokrig = optLogEI(cokrig_logEI, dim=3)
       can_MFNN   = optLogEI(MFNN_logEI,   dim=3)
       can_Ada2MF = optLogEI(Ada2MF_logEI, dim=3)
       can_AGMF   = optLogEI(AGMF_logEI,   dim=3)

       # update each method's design
       X_cokrig  = torch.cat([X_cokrig,  can_cokrig], dim=0)
       X_MFNN    = torch.cat([X_MFNN,    can_MFNN],   dim=0)
       X_Ada2MF  = torch.cat([X_Ada2MF,  can_Ada2MF], dim=0)
       X_AGMFNet = torch.cat([X_AGMFNet, can_AGMF],   dim=0)

Isosurface comparison across surrogates
---------------------------------------

To compare surrogate landscapes, we plot five panels at selected iterations
``plot_iters = {0, 5, 10, 15}``:

- co-Kriging mean
- MFNN mean
- Ada2MF mean
- AGMF-Net mean
- True function

We use a shared colormap normalization based on the true function range for
consistent visual comparison.

The helper below draws isosurfaces for a volume with shared color mapping.

.. code-block:: python

   def plot_h3_isosurfaces(ax, y_vol, X1, X2, X3, levels, norm, cmap, alpha=0.5):
       d1 = float(X1[1] - X1[0])
       d2 = float(X2[1] - X2[0])
       d3 = float(X3[1] - X3[0])

       origin = np.array([X1[0], X2[0], X3[0]])

       for lv in levels:
           verts, faces, _, _ = marching_cubes(y_vol, level=lv, spacing=(d1, d2, d3))
           verts = verts + origin

           mesh = Poly3DCollection(
               verts[faces],
               facecolors=cmap(norm(lv)),
               edgecolor="none",
               alpha=alpha,
           )
           ax.add_collection3d(mesh)

At each plotting iteration, we reshape the predicted grid means into volumes
``[N1,N2,N3]``, then plot all five panels. We also overlay:

- initial points :math:`X_e` (black)
- infill points chosen so far by each method (colored)

.. code-block:: python

   plot_iters = {0, 5, 10, 15}

   global_min = y_vol.min()
   global_max = y_vol.max()
   norm = Normalize(vmin=global_min, vmax=global_max)
   cmap = plt.cm.viridis

   sm = ScalarMappable(norm=norm, cmap=cmap)
   sm.set_array([])

   cmap_pts = plt.get_cmap("Dark2")
   s_cand = 10
   alpha_cand = 0.65

   if i in plot_iters:

       cokrig_vol = y_cokriging_cpu.reshape(N1, N2, N3)
       mfnn_vol   = MFNN_mean_cpu.reshape(N1, N2, N3)
       ada2mf_vol = Ada2MF_mean_cpu.reshape(N1, N2, N3)
       agmf_vol   = AGMF_mean_cpu.reshape(N1, N2, N3)

       fig = plt.figure(figsize=(14.96, 6))
       axes = [fig.add_subplot(1, 5, j+1, projection="3d") for j in range(5)]
       titles = ["co-Kriging", "MFNN", "Ada2MF", "AGMF-Net", "True Function"]
       vols   = [cokrig_vol, mfnn_vol, ada2mf_vol, agmf_vol, y_vol]

       for ax, title, vol in zip(axes, titles, vols):
           plot_h3_isosurfaces(ax, vol, X1, X2, X3, levels=levels, norm=norm, cmap=cmap, alpha=0.45)
           ax.set_title(title, pad=2)
           ax.set_xlabel("$x_1$")
           ax.set_ylabel("$x_2$")
           ax.set_zlabel("$x_3$")
           try:
               ax.set_box_aspect((1, 1, 1))
           except Exception:
               pass
           ax.view_init(elev=20, azim=45)

       # initial observations on all subplots
       for ax in axes:
           ax.scatter(Xe_cpu[:,0], Xe_cpu[:,1], Xe_cpu[:,2], s=12, c="k", depthshade=False)

       # candidates up to iteration i (per method)
       axes[0].scatter(cokrig_cand[:,0], cokrig_cand[:,1], cokrig_cand[:,2],
                       s=s_cand, color=cmap_pts(1), alpha=alpha_cand, depthshade=False)

       axes[1].scatter(mfnn_cand[:,0], mfnn_cand[:,1], mfnn_cand[:,2],
                       s=s_cand, color=cmap_pts(1), alpha=alpha_cand, depthshade=False)

       axes[2].scatter(ada2mf_cand[:,0], ada2mf_cand[:,1], ada2mf_cand[:,2],
                       s=s_cand, color=cmap_pts(1), alpha=alpha_cand, depthshade=False)

       axes[3].scatter(agmf_cand[:,0], agmf_cand[:,1], agmf_cand[:,2],
                       s=s_cand, color=cmap_pts(1), alpha=alpha_cand, depthshade=False)

       cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.1, orientation="horizontal")
       cbar.set_label("Hartmann-3D function value $f(x)$")

       plt.savefig(f"Max_thesis/figures/hartmann3d_{i}.pdf", bbox_inches="tight")
       plt.show()
       plt.close(fig)

Caching trained models and candidates
-------------------------------------

Your experiment script caches trained model checkpoints and the selected
candidates at each iteration. This has two benefits:

- you can rerun the notebook/script without retraining everything
- you can reproduce plots and metrics exactly

Typical caching pattern:

.. code-block:: python

   model_path = f"Trained_model/AGMF_Bench3_{i}.pth"
   if os.path.exists(model_path):
       model = torch.load(model_path, weights_only=False)
   else:
       model.fit(...)
       torch.save(model, model_path)


   cand_path = f"Trained_model/candidates3_{i}.pt"
   if os.path.exists(cand_path):
       cands = torch.load(cand_path, weights_only=False)
       can = cands["AGMF"]
   else:
       can = optLogEI(...)
       torch.save({"AGMF": can, ...}, cand_path)

Summary
-------

In this tutorial, we:

- Defined a 3D single-objective multi-fidelity benchmark using **Hartmann-3D**
- Constructed a biased low-fidelity approximation by modifying :math:`P`
- Standardized outputs using a shared high-fidelity-based :class:`YScaler`
- Compared co-Kriging, MFNN, Ada2MF, and AGMF-Net surrogates
- Ran a Bayesian optimization loop using **LogEI** for infill sampling
- Visualized surrogate landscapes in 3D using **isosurfaces** with marching cubes
- Tracked prediction quality using RMSE and :math:`r^2` over a 3D grid

The next tutorial extends this workflow to **multi-objective** optimization
(using batch acquisition functions such as qLogNEHVI).