# Quickstart

This quickstart gives **minimal working recipes** for common mfbo workflows:

- 1D **single-objective** (LogEI)
- multi-D **single-objective** (LogEI)
- multi-D **bi-objective** (qLogNEHVI)

It focuses on *the essential setup differences* (tensor shapes, bounds, acquisitions).  
Full benchmark scripts, plots, and analysis live in **Tutorials**.

---

## Conventions (important)

Throughout the docs, we use these shapes:

- Inputs `X`: `torch.Tensor` of shape **`[n, d]`** in **`[0, 1]^d`** (scaled domain)
- Single-objective outputs `Y`: shape **`[n, 1]`**
- Multi-objective outputs `Y`: shape **`[n, m]`**, e.g. `m=2` for bi-objective
- Bounds for BoTorch optimization: `bounds = torch.stack([lower, upper])` with shape **`[2, d]`**.

Recommended default dtype and device:

```python
import torch
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")  # or "cpu"
```

## Recipe A - 1 design variable, single objective (LogEI)

This is the smallest "end-to-end" loop.

### 1) Define objective and initial points

```python
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim.optimize import optimize_acqf

def f(x: torch.Tensor) -> torch.Tensor:
    # x: [n,1] in [0,1]
    return ((6*x - 2)**2 * torch.sin(12*x - 4))  # [n,1]

# initial design (n0 points)
X = torch.linspace(0.0, 1.0, 6).unsqueeze(-1)   # [n0,1]
Y = f(X)                                        # [n0,1]

bounds = torch.stack([torch.zeros(1), torch.ones(1)])  # [2,1]
```

### 2) Pick a surrogate model

Single-fidelity GP (Kriging kernel):

```python
from botorch.models import SingleTaskGP
from botorch.models.transforms.input   import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_torch

from mfbo.gp.kernels.kriging import Kriging

covar = ScaleKernel(Kriging(power_init=1.5, ard_num_dims=1))
model = SingleTaskGP(
    train_X=X,
    train_Y=Y,
    train_Yvar=torch.full_like(Y, 1e-6),
    covar_module=covar,
    input_transform=Normalize(d=1),
    outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={"step_limit": 2000})
model.eval()
```

Or single-fidelity neural ensemble:

```python
from mfbo.nn.ensembles import MLPEnsemble

mlp = MLPEnsemble(
    X_train=X.squeeze(-1),      # mfbo ensembles often expect [n] or [n,d] depending on your implementation
    y_train=Y.squeeze(-1),      # keep consistent with your model API
    ensemble_size=20,
    hid_features=16,
    n_hid_layers=2,
)
mlp.fit(epochs=500, optimizer="AdamW", lr=1e-3)
```

### 3) Acquisition + next point

For the GP case:

```python
logEI = LogExpectedImprovement(model=model, best_f=Y.min(), maximize=False)

X_next, _ = optimize_acqf(
    logEI,
    bounds=bounds,
    q=1,
    num_restarts=50,
    raw_samples=512,
)

Y_next = f(X_next)
X = torch.cat([X, X_next], dim=0)
Y = torch.cat([Y, Y_next], dim=0)
```

## Recipe B - multiple design variables, single objective (LogEI)

The only key differences from Recipe A:
- X is [n, d] instead of [n, 1]
- bounds is [2, d]
- Use LHS / Sobol for initial DOE

```python
import numpy as np
import torch
from skopt.sampler import Lhs
from skopt.space import Space

d = 2
space = Space(np.column_stack((np.zeros(d), np.ones(d))))
lhs = Lhs(lhs_type="centered", iterations=100000)

X = torch.tensor(lhs.generate(space.dimensions, n_samples=16), dtype=torch.float64)  # [n0, d]

def f(u: torch.Tensor) -> torch.Tensor:
    # u: [n,d] in [0,1]^d
    # return [n,1]
    x1 = 15.0 * u[:, 0] - 5.0
    x2 = 15.0 * u[:, 1]
    y  = (x2 - (5.1 * x1**2) / (4 * np.pi**2) + (5.0 * x1) / np.pi - 6.0)**2
    return (y / 50.0).unsqueeze(-1)

Y = f(X)  # [n0,1]
bounds = torch.stack([torch.zeros(d), torch.ones(d)])  # [2,d]
```

Then everything else is the same: fit surrogate --> LogExpectedImprovement --> optimize_acqf.

Tip: in multi-D, reduce num_restarts/raw_samples first to keep quickstart fast.

## Recipe C - multi-objective (bi-objective), batch candidates (qLogNEHVI)

Multi-objective BO differs in three places:
1. Output is [n, m] (here m=2)
2. You use qLogNEHVI (Monte Carlo) instead of analytic LogEI
3. You typically propose a batch q>1 points per iteration

Minimal setup:
```python
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf

# Example: returns [n,2]
def f(X: torch.Tensor) -> torch.Tensor:
    # X: [n,3] in [0,1]^3
    # Replace with your real objectives
    obj1 = X[:, 0] + X[:, 1]
    obj2 = (X[:, 2] - 0.5).abs()
    return torch.stack([obj1, obj2], dim=-1)

# initial DOE
n0, d, m = 30, 3, 2
X = torch.rand(n0, d)
Y = f(X)  # [n0,2]
bounds = torch.stack([torch.zeros(d), torch.ones(d)])  # [2,d]
```

### Acquisition (minimization via negation)

If you minimize objectives, a common pattern is to negate inside the objective:
```python
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
objective = GenericMCMultiOutputObjective(lambda Y, X=None: -Y)  # minimize

# choose a reference point in the transformed space (here: -Y)
ref_point = (-Y).min(dim=0).values - 1e-8

acq = qLogNoisyExpectedHypervolumeImprovement(
    model=your_model,          # must be a BoTorch Model returning posteriors over m outputs
    X_baseline=X,              # [n,d]
    ref_point=ref_point.tolist(),
    sampler=sampler,
    objective=objective,
    prune_baseline=False,
    cache_root=False,
)
```

### Optimize for a batch of candidates
```python
Xq, _ = optimize_acqf(
    acq_function=acq,
    bounds=bounds,
    q=3,                  # propose 3 points per iteration
    num_restarts=10,
    raw_samples=128,
    sequential=True,
)

Yq = f(Xq.view(-1, d)).view(3, m)   # evaluate and reshape if needed
```

For full multi-objective workflows (constraints, reference directions, PF metrics, plotting), see Tutorials.

## Multi-fidelity note (where mfbo shines)

If you have low-fidelity and high-fidelity data, mfbo provides models like:

```python
from mfbo.gp.cokriging import CoKrigingAR1
from mfbo.nn.ensembles import MFNNEnsemble, Ada2MFEnsemble, AGMFNetEnsemble
```

The BO loop remains the same; only the model training data changes (LF/HF) and how your model returns the posterior / predictive distribution.
