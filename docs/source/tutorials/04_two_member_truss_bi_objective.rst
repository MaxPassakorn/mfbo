Two-Member Truss (3D Design, Bi-Objective)
==========================================

This tutorial demonstrates **bi-objective multi-fidelity Bayesian optimization**
on the classical **two-member truss** benchmark. The design space is 3D
(two cross-sectional areas and one geometry parameter), and we optimize two
competing objectives:

- :math:`f_1(x)` = **volume** (proxy for weight / material usage)
- :math:`f_2(x)` = **maximum stress** (worst-case member stress)

We compare four multi-fidelity surrogate models from :mod:`mfbo`:

- :class:`mfbo.gp.cokriging.CoKrigingAR1` (multi-output AR(1) co-Kriging)
- :class:`mfbo.nn.ensembles.MFNNEnsemble`
- :class:`mfbo.nn.ensembles.Ada2MFEnsemble`
- :class:`mfbo.nn.ensembles.AGMFNetEnsemble`

For multi-objective infill, we use **batch** Monte-Carlo acquisition:

- :class:`botorch.acquisition.multi_objective.logei.qLogNoisyExpectedHypervolumeImprovement`
  (qLogNEHVI)

At each iteration, each surrogate selects a batch of :math:`q=3` new candidates.
We then update each method's design set and track surrogate quality using:

- RMSE / NRMSE / :math:`r^2` on a 3D evaluation grid
- MARE in the **Pareto-front (PF) region**, comparing surrogate PF vs. true PF

Problem definition
------------------

Design variables
^^^^^^^^^^^^^^^^

We optimize over :math:`U = (x_1, x_2, y)` where:

- :math:`x_1` = cross-sectional area of member AC :math:`[m^2]`
- :math:`x_2` = cross-sectional area of member BC :math:`[m^2]`
- :math:`y`   = geometry parameter (height) :math:`[m]`

Domain bounds (as used in your script):

.. math::

   x_1 \in [10^{-6}, 10^{-2}],\quad
   x_2 \in [10^{-6}, 10^{-2}],\quad
   y \in [1, 3].

Objectives
^^^^^^^^^^

Your high-fidelity model returns two objectives:

1. **Volume**:

.. math::

   f_1(x) =
   x_1 \sqrt{16 + y^2} + x_2 \sqrt{1 + y^2}.

2. **Maximum stress** (worst member stress):

.. math::

   \sigma_{AC} = \frac{20\sqrt{16+y^2}}{y\,x_1},\qquad
   \sigma_{BC} = \frac{80\sqrt{1+y^2}}{y\,x_2},\qquad
   f_2(x) = \max(\sigma_{AC},\sigma_{BC}).

Hence the high-fidelity objective vector is:

.. math::

   f(x) = [f_1(x), f_2(x)].

Low-fidelity model
^^^^^^^^^^^^^^^^^^

The low-fidelity model introduces controlled bias in both objectives:

- It uses a **first-order linearization** of the square-root terms around
  :math:`y_0 = 2`
- It replaces :math:`\max(\sigma_{AC},\sigma_{BC})` with a smooth approximation:

.. math::

   \mathrm{softmax}_p(a,b) = (a^p + b^p)^{1/p}

with :math:`p=4` by default.

This yields a correlated but imperfect approximation, which is ideal for
multi-fidelity experiments.

Setup
-----

.. code-block:: python

   import os
   import numpy as np
   import torch

   import matplotlib as mpl
   import matplotlib.pyplot as plt
   from matplotlib.colors import Normalize, LogNorm
   from mpl_toolkits.mplot3d.art3d import Poly3DCollection
   from skimage.measure import marching_cubes

   from skopt.sampler import Lhs
   from skopt.space import Space

   from botorch.sampling.normal import SobolQMCNormalSampler
   from botorch.optim.optimize import optimize_acqf
   from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
   from botorch.acquisition.multi_objective.objective import (
       GenericMCMultiOutputObjective,
       IdentityMCMultiOutputObjective,
   )

   from pymoo.core.problem import Problem
   from pymoo.algorithms.moo.unsga3 import UNSGA3
   from pymoo.optimize import minimize
   from pymoo.util.ref_dirs import get_reference_directions
   from pymoo.problems import get_problem

   from mfbo.gp.cokriging import CoKrigingAR1
   from mfbo.nn.ensembles import MFNNEnsemble, Ada2MFEnsemble, AGMFNetEnsemble


   torch.manual_seed(42)
   torch.set_default_dtype(torch.float64)
   torch.set_default_device("cuda")

High-fidelity and low-fidelity functions
----------------------------------------

Domain limits
^^^^^^^^^^^^^

.. code-block:: python

   X1_MIN, X1_MAX = 1e-6, 1.0e-2
   X2_MIN, X2_MAX = 1e-6, 1.0e-2
   Y_MIN,  Y_MAX  = 1.0,  3.0

Linearization around :math:`y_0=2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We linearize :math:`\sqrt{16+y^2}` and :math:`\sqrt{1+y^2}` at :math:`y_0 = 2`:

.. code-block:: python

   _y0 = torch.tensor(2.0, dtype=torch.float64)

   _a1 = torch.sqrt(16.0 + _y0*_y0)  # sqrt(20)
   _b1 = _y0 / _a1                   # 2/sqrt(20)

   _a2 = torch.sqrt(1.0 + _y0*_y0)   # sqrt(5)
   _b2 = _y0 / _a2                   # 2/sqrt(5)

   def _sqrt16y2_lin(y):
       return _a1 + _b1 * (y - _y0)

   def _sqrt1y2_lin(y):
       return _a2 + _b2 * (y - _y0)

High-fidelity :math:`f(x)`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def f(U: torch.Tensor) -> torch.Tensor:
       U = U.view(-1, 3).double()
       x1, x2, y = U[:,0], U[:,1], U[:,2]

       vol = x1 * torch.sqrt(16.0 + y*y) + x2 * torch.sqrt(1.0 + y*y)

       sigAC = 20.0 * torch.sqrt(16.0 + y*y) / (y * x1)
       sigBC = 80.0 * torch.sqrt(1.0 + y*y) / (y * x2)

       stress = torch.maximum(sigAC, sigBC)
       return torch.cat([vol.unsqueeze(-1), stress.unsqueeze(-1)], dim=-1)

Low-fidelity :math:`f_L(x)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def low_f(U: torch.Tensor, p: int = 4) -> torch.Tensor:
       U = U.view(-1, 3).double()
       x1, x2, y = U[:,0], U[:,1], U[:,2]

       vol_lin = x1 * _sqrt16y2_lin(y) + x2 * _sqrt1y2_lin(y)

       sigAC = 20.0 * _sqrt16y2_lin(y) / (y * x1)
       sigBC = 80.0 * _sqrt1y2_lin(y) / (y * x2)

       stress_smooth = (sigAC.pow(p) + sigBC.pow(p)).pow(1.0 / p)

       return torch.cat([vol_lin.unsqueeze(-1), stress_smooth.unsqueeze(-1)], dim=-1)

Input / output scaling
----------------------

Why scaling is important
^^^^^^^^^^^^^^^^^^^^^^^^

In this benchmark:

- Inputs live in **very different physical scales**
  (:math:`x_1, x_2` are :math:`10^{-6}` to :math:`10^{-2}` while :math:`y` is :math:`1` to :math:`3`)
- The second objective (stress) can span **orders of magnitude**

To make neural surrogates train reliably and to make acquisition optimization
well-conditioned, we:

1. map physical inputs to :math:`[0,1]^3` using :class:`XScaler`
2. standardize outputs per objective using :class:`YScaler`

XScaler
^^^^^^^

.. code-block:: python

   class XScaler:
       def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
           self.lower = lower.view(1, -1)
           self.scale = (upper.view(1, -1) - self.lower).clamp_min(1e-12)

       def encode(self, X: torch.Tensor) -> torch.Tensor:
           return (X - self.lower) / self.scale

       def decode(self, Xs: torch.Tensor) -> torch.Tensor:
           return Xs * self.scale + self.lower

YScaler
^^^^^^^

.. code-block:: python

   class YScaler:
       def __init__(self, Y: torch.Tensor):
           self.mean = Y.mean(dim=0, keepdim=True)
           self.std  = Y.std(dim=0, keepdim=True).clamp_min(1e-12)

       def encode(self, Y: torch.Tensor) -> torch.Tensor:
           return (Y - self.mean) / self.std

       def decode(self, Ys: torch.Tensor) -> torch.Tensor:
           return Ys * self.std + self.mean

Bounds for acquisition optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We optimize acquisition functions in scaled coordinates :math:`[0,1]^3`:

.. code-block:: python

   lower = torch.tensor([X1_MIN, X2_MIN, Y_MIN])
   upper = torch.tensor([X1_MAX, X2_MAX, Y_MAX])

   bounds = torch.stack([torch.zeros_like(lower), torch.ones_like(lower)], dim=0)

Initial designs and evaluation grid
-----------------------------------

We create:

- initial high-fidelity design :math:`X_e` with **30** LHS samples
- large low-fidelity design :math:`X_c` with **2000** samples
- a coarse 3D grid :math:`10^3 = 1000` points for visualization + metrics

.. code-block:: python

   space = Space([(X1_MIN, X1_MAX), (X2_MIN, X2_MAX), (Y_MIN, Y_MAX)])

   lhs_e = Lhs(lhs_type="centered", iterations=100_000_000)
   lhs_c = Lhs(lhs_type="centered", iterations=1_000_000)

   Xe = torch.tensor(lhs_e.generate(space.dimensions, n_samples=30))    # [30,3]
   Xc = torch.tensor(lhs_c.generate(space.dimensions, n_samples=2000))  # [2000,3]

   N1 = N2 = N3 = 10
   x1_val = torch.linspace(X1_MIN, X1_MAX, N1)
   x2_val = torch.linspace(X2_MIN, X2_MAX, N2)
   x3_val = torch.linspace(Y_MIN,  Y_MAX,  N3)

   x1, x2, x3 = torch.meshgrid(x1_val, x2_val, x3_val, indexing="ij")
   X = torch.stack([x1, x2, x3], dim=-1).view(-1, 3)  # [1000,3]

We then scale inputs and outputs consistently:

.. code-block:: python

   x_scaler = XScaler(lower, upper)
   Xe_s = x_scaler.encode(Xe)
   Xc_s = x_scaler.encode(Xc)
   X_s  = x_scaler.encode(X)

   yc      = low_f(Xc)     # LF at support points
   yc_plot = low_f(X)      # LF on grid
   ye      = f(X)          # HF on grid
   y       = f(Xe)         # HF at initial points

   y_scaler = YScaler(y)   # compute mean/std from initial HF only

   yc_s      = y_scaler.encode(yc)
   yc_s_plot = y_scaler.encode(yc_plot)
   ye_s      = y_scaler.encode(ye)
   y_s       = y_scaler.encode(y)

For the neural models, we define scaled wrappers that accept scaled inputs:

.. code-block:: python

   def low_f_scaled(Xs: torch.Tensor, p: int = 4) -> torch.Tensor:
       X_phys = x_scaler.decode(Xs)
       return y_scaler.encode(low_f(X_phys, p=p))

   def f_scaled_X(Xs: torch.Tensor) -> torch.Tensor:
       X_phys = x_scaler.decode(Xs)
       return y_scaler.encode(f(X_phys))

Multi-objective acquisition: qLogNEHVI
--------------------------------------

Why qLogNEHVI
^^^^^^^^^^^^^

In multi-objective optimization we do not optimize a single scalar objective.
Instead, we aim to improve the **Pareto front**. A standard approach is to
maximize the **hypervolume improvement** (HVI). In noisy / MC settings,
BoTorch provides qNEHVI and qLogNEHVI.

We use:

- ``qLogNoisyExpectedHypervolumeImprovement`` (qLogNEHVI)
- batch size :math:`q=3`
- Sobol QMC sampling with ``mc_samples=256``

Handling minimization
^^^^^^^^^^^^^^^^^^^^^

BoTorch's NEHVI is typically formulated for **maximization** of objectives.
Your objectives are **minimized**. The tutorial follows your implementation:

- Convert minimization to maximization by negating outputs in the objective:
  ``objective = lambda Y: -Y``
- Build a consistent reference point in the negated space.

.. code-block:: python

   def build_qLogNEHVI(
       model,
       X_baseline: torch.Tensor,
       Y_baseline: torch.Tensor,
       *,
       minimize: bool = True,
       ref_point=None,
       constraints=None,
       mc_samples: int = 256,
   ):
       sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

       if minimize:
           objective = GenericMCMultiOutputObjective(lambda Y, X=None: -Y)
           rp = (-Y_baseline).min(dim=0).values - 1e-8 if ref_point is None \
                else -torch.as_tensor(ref_point, dtype=Y_baseline.dtype, device=Y_baseline.device)
       else:
           objective = IdentityMCMultiOutputObjective()
           rp = (Y_baseline.min(dim=0).values - 1e-8) if ref_point is None \
                else torch.as_tensor(ref_point, dtype=Y_baseline.dtype, device=Y_baseline.device)

       return qLogNoisyExpectedHypervolumeImprovement(
           model=model,
           X_baseline=X_baseline,
           ref_point=rp.tolist(),
           sampler=sampler,
           objective=objective,
           constraints=constraints,
           prune_baseline=False,
           cache_root=False,
       )

Candidate generation with optimize_acqf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We optimize qLogNEHVI over :math:`[0,1]^3` and return :math:`q=3` candidates:

.. code-block:: python

   def qLogNEHVI_cands(acq, q=3, num_restarts=10, raw_samples=128, sequential=True):
       Xq, _ = optimize_acqf(
           acq_function=acq,
           bounds=bounds,
           q=q,
           num_restarts=num_restarts,
           raw_samples=raw_samples,
           sequential=sequential,
       )
       return Xq

3D visualization of objectives
------------------------------

To inspect how each objective varies over the 3D design space, we visualize the
functions as **isosurfaces** on the evaluation grid using marching cubes.

Your helper ``plot_3d``:

- reshapes a 1D grid evaluation into a 3D volume
- chooses iso-levels from percentiles for robustness
- optionally uses log normalization for stress

.. code-block:: python

   def plot_3d(y, filename, use_log=False, p_lo=1.0, p_hi=99.0, n_levels=5):
       N1, N2, N3 = len(x1_val), len(x2_val), len(x3_val)
       A = y.detach().cpu().numpy().reshape(N1, N2, N3).squeeze()

       X1 = x1_val.detach().cpu().numpy()
       X2 = x2_val.detach().cpu().numpy()
       X3 = x3_val.detach().cpu().numpy()
       d1 = float(X1[1] - X1[0])
       d2 = float(X2[1] - X2[0])
       d3 = float(X3[1] - X3[0])

       finite = np.isfinite(A)
       lo = np.percentile(A[finite], p_lo)
       hi = np.percentile(A[finite], p_hi)

       if use_log:
           eps = 1e-12
           lo = max(lo, eps)
           levels = np.geomspace(lo, hi, n_levels)
           norm = LogNorm(vmin=lo, vmax=hi)
       else:
           levels = np.linspace(lo, hi, n_levels)
           norm = Normalize(vmin=lo, vmax=hi)

       fig = plt.figure(figsize=(3.8, 3.8))
       ax  = fig.add_subplot(111, projection="3d")
       cmap = plt.cm.viridis

       for lv in levels:
           verts, faces, _, _ = marching_cubes(A, level=lv, spacing=(d1, d2, d3))
           origin = np.array([float(X1[0]), float(X2[0]), float(X3[0])])
           verts = verts + origin
           ax.add_collection3d(
               Poly3DCollection(verts[faces], facecolors=cmap(norm(lv)),
                                edgecolor="none", alpha=0.5)
           )

       ax.scatter(Xe_cpu[:,0], Xe_cpu[:,1], Xe_cpu[:,2], s=30, c="k", depthshade=False)
       ax.set_xlabel("$x_1$")
       ax.set_ylabel("$x_2$")
       ax.set_zlabel("$y$")
       fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.1)

       plt.tight_layout()
       plt.savefig(filename)
       plt.show()

Example calls (as in your script):

.. code-block:: python

   plot_3d(ye[:, 0],     "Max_thesis/figures/truss2d_vol.pdf",       use_log=False, p_lo=5, p_hi=95)
   plot_3d(yc_plot[:,0], "Max_thesis/figures/truss2d_lf_vol.pdf",    use_log=False, p_lo=5, p_hi=95)

   plot_3d(ye[:, 1],     "Max_thesis/figures/truss2d_stress.pdf",    use_log=True,  p_lo=5, p_hi=95)
   plot_3d(yc_plot[:,1], "Max_thesis/figures/truss2d_lf_stress.pdf", use_log=False, p_lo=5, p_hi=95)

True Pareto front and surrogate Pareto fronts
---------------------------------------------

True Pareto front
^^^^^^^^^^^^^^^^^

We use Pymoo's built-in test problem ``truss2d`` to obtain a reference Pareto
front:

.. code-block:: python

   problem = get_problem("truss2d")
   F_true = problem.pareto_front()  # [N_pf, 2]

Surrogate Pareto front via UNSGA3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

qLogNEHVI improves designs, but to **compare Pareto fronts** at each iteration
we also compute an approximate surrogate PF by solving:

.. math::

   \min_x \; \hat{f}(x)

subject to the same constraints as the original problem.

We do this using :class:`pymoo.algorithms.moo.unsga3.UNSGA3` with fixed reference
directions.

Reference directions
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ref_dirs = get_reference_directions("energy", 2, 64, seed=1)

   fig, ax = plt.subplots(figsize=(3.74, 3.74))
   ax.scatter(ref_dirs[:, 0], ref_dirs[:, 1], s=5)
   ax.set_xlabel(r"$f_1$")
   ax.set_ylabel(r"$f_2$")
   ax.set_aspect("equal", adjustable="box")
   ax.grid(True, alpha=0.3)
   plt.savefig("Max_thesis/figures/trussrefdir.pdf", bbox_inches="tight", backend="pgf")

Wrapping a surrogate as a Pymoo problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pymoo expects a problem class with ``_evaluate`` returning:

- ``out["F"]`` : objective values
- ``out["G"]`` : inequality constraints

Your wrapper:

1. encodes inputs using :class:`XScaler`
2. evaluates the model in scaled space
3. converts ensemble outputs to a mean if needed
4. decodes outputs back to physical objective space
5. retrieves constraints from the true Pymoo problem (so constraints are exact)

.. code-block:: python

   class Truss2D_Surrogate(Problem):
       def __init__(self, model, xl=None, xu=None):
           xl = problem.xl if xl is None else xl
           xu = problem.xu if xu is None else xu
           super().__init__(n_var=problem.n_var, n_obj=2, n_ieq_constr=problem.n_ieq_constr, xl=xl, xu=xu)
           self.model = model

       def _evaluate(self, X, out, *args, **kwargs):
           N = X.shape[0]

           with torch.no_grad():
               Xt = torch.tensor(X, dtype=torch.float64, device="cuda")
               Xs = x_scaler.encode(Xt)

               Y = self.model(Xs)
               if isinstance(Y, (tuple, list)):
                   Y = Y[0]

               # unify possible output shapes
               if Y.dim() == 2:
                   pass
               elif Y.dim() == 3:
                   if Y.shape[0] == N:
                       Y = Y.mean(dim=1)
                   elif Y.shape[1] == N:
                       Y = Y.mean(dim=0)
                   else:
                       raise RuntimeError(f"Unexpected 3D shape {tuple(Y.shape)} for N={N}")
               elif Y.dim() == 4:
                   Y = Y.mean(dim=1).squeeze(0)
               else:
                   raise RuntimeError(f"Unhandled output shape {tuple(Y.shape)}")

               Y = y_scaler.decode(Y)
               F = Y.detach().cpu().numpy()

           G = problem.evaluate(X, return_values_of=["G"], return_as_dictionary=True)["G"]

           out["F"] = F
           out["G"] = G

Running UNSGA3 on a surrogate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def run_unsga3(model):
       prob = Truss2D_Surrogate(model=model)
       algo = UNSGA3(ref_dirs=ref_dirs, pop_size=256)
       res  = minimize(prob, algo, verbose=True, copy_algorithm=False, copy_termination=False)
       return res.F

Accuracy metrics
----------------

Grid-based predictive metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the evaluation grid, we compute for each objective:

- RMSE
- NRMSE (normalized by std or by range)
- :math:`r^2`

.. code-block:: python

   def _rmse(a, b):
       a = a.double().flatten()
       b = b.double().flatten()
       return torch.sqrt(torch.mean((a - b) ** 2))

   def _nrmse(a, b, method="std"):
       a = a.double().flatten()
       b = b.double().flatten()
       rmse = torch.sqrt(torch.mean((a - b) ** 2))
       if method == "std":
           denom = b.std().clamp_min(1e-12)
       elif method == "range":
           denom = (b.max() - b.min()).clamp_min(1e-12)
       else:
           raise ValueError("method ∈ {'std','range'}")
       return rmse / denom

   def r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
       y_pred = y_pred.flatten().double()
       y_true = y_true.flatten().double()
       yp = y_pred - y_pred.mean()
       yt = y_true - y_true.mean()
       num = (yp * yt).sum()
       den = torch.sqrt((yp**2).sum() * (yt**2).sum())
       r = num / (den + 1e-12)
       return r**2

MARE on Pareto front region
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A surrogate can be accurate in global RMSE yet still miss the Pareto region.
To evaluate quality where it matters, we compute MARE between:

- true PF points :math:`F_{\mathrm{true}}`
- surrogate PF points :math:`F_{\mathrm{hat}}`

For each true PF point, we find the nearest surrogate PF point in objective space,
then compute per-objective MARE.

.. code-block:: python

   def _mare_vec(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
       rel = (y_pred - y_true).abs() / (y_true.abs() + eps)
       return rel.mean(dim=0)

   def mare_pareto(F_hat_np: np.ndarray, F_true_np: np.ndarray, eps: float = 1e-8) -> torch.Tensor:
       F_hat  = torch.from_numpy(F_hat_np).double()
       F_true = torch.from_numpy(F_true_np).double()

       diff  = F_true.unsqueeze(1) - F_hat.unsqueeze(0)  # [N_true, N_hat, 2]
       dist2 = (diff ** 2).sum(dim=-1)
       idx   = dist2.argmin(dim=1)

       y_pred = F_hat[idx]
       y_true = F_true
       return _mare_vec(y_pred, y_true, eps=eps)

Bi-objective BO loop (qLogNEHVI)
--------------------------------

We run for ``infill_num = 10`` rounds. Each round:

1. Fit each surrogate on its own dataset.
2. Compute surrogate mean on the grid for metrics.
3. Compute a surrogate Pareto front using UNSGA3.
4. Compare surrogate PF to true PF.
5. Build qLogNEHVI and select a batch of :math:`q=3` candidates.
6. Append candidates to each method’s dataset.

Important implementation details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Co-Kriging** is trained on scaled inputs and scaled outputs (multi-output GP).
- **Ensembles** return an ensemble dimension; we average across ensemble members
  to form the predicted mean.
- We keep each method’s training set in **scaled input space** (``Xe_s`` etc.),
  because qLogNEHVI optimization runs in :math:`[0,1]^3`.
- Before evaluating the true HF function :math:`f`, we decode candidates back to
  physical space via :meth:`XScaler.decode`.

Core loop structure
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   infill_num = 10

   X_cokrig  = Xe_s.clone()
   X_MFNN    = Xe_s.clone()
   X_Ada2MF  = Xe_s.clone()
   X_AGMFNet = Xe_s.clone()

   titles = ["co-Kriging", "MFNN", "Ada2MF", "AGMF-Net"]

   mare_pf_scores = {m: [] for m in ["cokriging", "mfnn", "ada2mf", "agmf"]}

   for i in range(infill_num + 1):

       fig, axes = plt.subplots(1, 4, figsize=(7.48, 2.3), gridspec_kw={"wspace": 0.15})

       # ===== co-Kriging =====
       y_cokrig = f_scaled_X(X_cokrig)
       cokrig_model = CoKrigingAR1(Xc_s, yc_s, X_cokrig, y_cokrig)

       for lik in cokrig_model.lik_l_list:
           lik.noise_covar.initialize(noise=1e-4)
       for lik in cokrig_model.lik_h_list:
           lik.noise_covar.initialize(noise=1e-4)

       cokrig_model.fit(iters_per_stage=120, stages=3, lr_low=0.01, lr_delta=0.01, verbose=True)

       y_ck_s, _ = batched_predict(cokrig_model, X_s, batch_size=1000)
       y_ck = y_scaler.decode(y_ck_s)                 # [N,2]

       acq_cokrig = build_qLogNEHVI(cokrig_model, X_cokrig, y_cokrig, minimize=True)
       F_cokrig = run_unsga3(cokrig_model)

       # ===== MFNN =====
       y_MFNN = f_scaled_X(X_MFNN)
       MFNN_model = MFNNEnsemble(
           X_train=X_MFNN,
           low_fn=low_f_scaled,
           y_train=y_MFNN,
           ensemble_size=50,
           hid_features=5,
           n_hid_layers=2,
       )
       MFNN_model.fit(epochs=500, optimizer="AdamW", lr=1e-3)

       MFNN_mean_s = MFNN_model(X_s).mean(dim=1).squeeze(0)
       y_mfnn = y_scaler.decode(MFNN_mean_s)

       acq_MFNN = build_qLogNEHVI(MFNN_model, X_MFNN, y_MFNN, minimize=True)
       F_MFNN = run_unsga3(MFNN_model)

       # ===== Ada2MF =====
       y_Ada2MF = f_scaled_X(X_Ada2MF)
       Ada2MF_model = Ada2MFEnsemble(
           X_train=X_Ada2MF,
           low_fn=low_f_scaled,
           y_train=y_Ada2MF,
           ensemble_size=50,
           hid_features=5,
           n_layers=2,
       )
       Ada2MF_model.fit(epochs=500, optimizer="AdamW", lr=1e-3)

       Ada2MF_mean_s = Ada2MF_model(X_s).mean(dim=1).squeeze(0)
       y_ada = y_scaler.decode(Ada2MF_mean_s)

       acq_Ada2MF = build_qLogNEHVI(Ada2MF_model, X_Ada2MF, y_Ada2MF, minimize=True)
       F_Ada2MF = run_unsga3(Ada2MF_model)

       # ===== AGMF-Net =====
       y_AGMFNet = f_scaled_X(X_AGMFNet)
       AGMF_model = AGMFNetEnsemble(
           X_train=X_AGMFNet,
           low_fn=low_f_scaled,
           y_train=y_AGMFNet,
           ensemble_size=50,
           hid_features=5,
           n_layers=2,
       )
       AGMF_model.fit(epochs=15000, optimizer="AdamW", lr=1e-3)

       AGMF_mean_s = AGMF_model(X_s).mean(dim=1).squeeze(0)
       y_ag = y_scaler.decode(AGMF_mean_s)

       acq_AGMF = build_qLogNEHVI(AGMF_model, X_AGMFNet, y_AGMFNet, minimize=True)
       F_AGMF = run_unsga3(AGMF_model)

       # ===== metrics on grid =====
       pred_mat = {"cokriging": y_ck, "mfnn": y_mfnn, "ada2mf": y_ada, "agmf": y_ag}

       # per objective (k=0 volume, k=1 stress)
       for k in range(2):
           y_true_k = ye[:, k]
           for name, Yhat in pred_mat.items():
               _ = _rmse(Yhat[:, k], y_true_k)
               _ = _nrmse(Yhat[:, k], y_true_k, method="std")
               _ = r2(Yhat[:, k], y_true_k)

       # ===== plot PFs =====
       fronts = [F_cokrig, F_MFNN, F_Ada2MF, F_AGMF]
       for ax, F_hat, title in zip(axes, fronts, titles):
           ax.scatter(F_true[:,0], F_true[:,1], s=8, alpha=0.25, label="True PF")
           ax.scatter(F_hat[:,0],  F_hat[:,1],  s=10, alpha=0.9,  label="Surrogate PF")
           ax.set_title(title)
           ax.grid(True, alpha=0.3)

       # ===== PF-region MARE =====
       pf_names = ["cokriging", "mfnn", "ada2mf", "agmf"]
       for name, F_hat in zip(pf_names, fronts):
           mare_vec = mare_pareto(F_hat, F_true)
           mare_pf_scores[name].append(mare_vec.cpu())

       # ===== select q=3 candidates =====
       can_cokrig = qLogNEHVI_cands(acq_cokrig, q=3)
       can_MFNN   = qLogNEHVI_cands(acq_MFNN,   q=3)
       can_Ada2MF = qLogNEHVI_cands(acq_Ada2MF, q=3)
       can_AGMF   = qLogNEHVI_cands(acq_AGMF,   q=3)

       # update datasets (still scaled)
       X_cokrig  = torch.cat([X_cokrig,  can_cokrig], dim=0)
       X_MFNN    = torch.cat([X_MFNN,    can_MFNN],   dim=0)
       X_Ada2MF  = torch.cat([X_Ada2MF,  can_Ada2MF], dim=0)
       X_AGMFNet = torch.cat([X_AGMFNet, can_AGMF],   dim=0)

       # for printing / evaluation in physical space
       can_cokrig_phys = x_scaler.decode(can_cokrig)
       can_MFNN_phys   = x_scaler.decode(can_MFNN)
       can_Ada2MF_phys = x_scaler.decode(can_Ada2MF)
       can_AGMF_phys   = x_scaler.decode(can_AGMF)

       axes[0].set_xlabel("$f_1$ (weight)")
       axes[0].set_ylabel("$f_2$ (deflection)")
       for ax in axes[1:]:
           ax.set_xlabel("$f_1$ (weight)")
       axes[0].legend(frameon=False, loc="best")

       plt.savefig(f"Max_thesis/figures/truss{i}.pdf", bbox_inches="tight", backend="pgf")
       plt.show()

Notes on candidate batching
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- With qLogNEHVI, we propose **three points per iteration** (:math:`q=3`).
- Your code uses ``optimize_acqf`` with ``sequential=True`` which picks the batch
  sequentially (conditioning on previously selected points in the same batch).
- You also implemented an alternative batch selection method using UNSGA3
  on the acquisition function itself (``pick_batch_with_unsga3``), which can be
  useful when ``optimize_acqf`` struggles. In this tutorial we keep the default
  ``optimize_acqf`` approach.

Summary
-------

In this tutorial, we:

- Defined a **bi-objective** two-member truss problem with (volume, stress)
- Built a correlated **low-fidelity** model by linearizing geometry terms and
  smoothing max-stress
- Normalized inputs to :math:`[0,1]^3` and standardized outputs per objective
- Trained and compared co-Kriging, MFNN, Ada2MF, and AGMF-Net surrogates
- Used **qLogNEHVI** (batch :math:`q=3`) for multi-objective infill sampling
- Computed surrogate Pareto fronts via **UNSGA3** and compared them to the true PF
- Reported both global grid metrics and **Pareto-region** accuracy (MARE on PF)

This workflow is a practical template for multi-objective multi-fidelity BO
in :mod:`mfbo`, combining BoTorch (infill) and Pymoo (Pareto-front extraction).