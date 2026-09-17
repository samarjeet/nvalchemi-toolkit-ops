:mod:`nvalchemiops.jax.lbfgs`: Geometry Optimization
====================================================

JAX bindings for the batched L-BFGS geometry optimizer.

.. automodule:: nvalchemiops.jax.lbfgs
    :no-members:
    :no-inherited-members:

.. tip::
   State is explicit and JAX arrays are immutable: each step returns updated
   positions and a reconstructed state object. Donate those values when using
   ``jax.jit`` to permit storage reuse. Set ``jax_enable_x64=True`` before
   preparing state; L-BFGS retains float64 reductions and recursion
   coefficients even when coordinates use float32.

Coordinate Relaxation
---------------------

.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord

Variable-Cell Relaxation
------------------------

Coordinates and cell are mapped into one packed coordinate vector, so the
two-loop recursion couples them automatically.
The cell-state allocator derives and stores the extended topology from the sorted ``batch_idx`` input.
The current cell and complete state may be donated together because the
allocator retains an independent reference-cell buffer. The caller remains
responsible for convergence, batch/state alignment, and proposal validation.

.. autofunction:: nvalchemiops.jax.lbfgs.prepare_lbfgs_state
.. autofunction:: nvalchemiops.jax.lbfgs.prepare_lbfgs_cell_state
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord_cell
