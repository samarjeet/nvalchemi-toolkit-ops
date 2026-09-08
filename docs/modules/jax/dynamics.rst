:mod:`nvalchemiops.jax.lbfgs`: Geometry Optimization
====================================================

JAX bindings for the batched L-BFGS geometry optimizer.

.. automodule:: nvalchemiops.jax.lbfgs
    :no-members:
    :no-inherited-members:

.. tip::
   JAX arrays are immutable, so these entry points return a new state rather
   than mutating one. Donate the state with ``jax.jit(donate_argnums=...)`` so
   XLA can reuse the buffers; see the module documentation above for the
   donation and CUDA-graph contract.

Coordinate Relaxation
---------------------

.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_allocate_state
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_converged

Variable-Cell Relaxation
------------------------

Coordinates and cell are mapped into one packed coordinate vector, so the
two-loop recursion couples them automatically.

.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_allocate_cell_state
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_set_reference_cell
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord_cell
