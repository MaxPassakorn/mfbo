API Reference
=============

This section documents the public API of **mfbo**.

The library is organized into the following main modules:

- Gaussian-process surrogates (``mfbo.gp``)
- Neural surrogate models (``mfbo.nn``)
- Neural ensemble surrogates (``mfbo.nn.ensembles``)
- Posterior distributions (``mfbo.posteriors``)
- Utility functions (``mfbo.utils``)

All models are compatible with **BoTorch** and can be used
directly in Bayesian optimization workflows.

.. toctree::
   :maxdepth: 1

   gp
   kernels
   nn
   ensembles
   posteriors
   utils