Using and Validating Development Builds
=======================================

A successful compiler invocation proves only that artifacts were produced.
Before testing a change, verify that the current Python process imports the
checkout, loads the newly built native libraries, and can execute an operation
through the selected framework.

Confirm the selected installation
---------------------------------

Inspect the imported package and installed distribution:

.. code-block:: python

   from importlib.metadata import distribution
   from pathlib import Path

   import transformer_engine

   package_path = Path(transformer_engine.__file__).resolve()
   dist = distribution("transformer-engine")

   print(f"Imported package: {package_path}")
   print(f"Installed distribution: {dist.locate_file('')}")

For the NGC workflow, the imported package path should be under
``/workspace/TransformerEngine``. For another editable installation, it
should be under that checkout rather than only under ``site-packages``.

The runtime loader can also report the native artifacts it selected:

.. code-block:: python

   from transformer_engine.common import _get_shared_object_file

   print(f"Common library: {_get_shared_object_file('core')}")

``_get_shared_object_file`` is a private diagnostic helper, not a public
application API. It searches the editable checkout, a regular source
installation, and the split-wheel library directory. Use it to diagnose a
development environment, but do not depend on it in user code.

Run common functionality
------------------------

A common-only build can be checked without importing PyTorch or JAX:

.. code-block:: python

   import ctypes

   from transformer_engine.common import _get_shared_object_file

   common_library = _get_shared_object_file("core")
   ctypes.CDLL(str(common_library))
   print(common_library)

This confirms that the framework-independent shared library exists and that
the dynamic loader can resolve its dependencies. It does not validate an
operation's numerical behavior. Changes to the C API or common kernels should
also run the focused C++ tests under ``tests/cpp/``; the testing section
describes that workflow.

Run a framework integration
---------------------------

Use a small forward and backward computation to exercise Python code, the
private framework binding, and the common library together.

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         import torch
         import transformer_engine.pytorch as te

         x = torch.randn(8, 16, device="cuda", requires_grad=True)
         layer = te.Linear(16, 16).cuda()

         y = layer(x)
         y.sum().backward()
         torch.cuda.synchronize()

         print(y.shape)
         print(x.grad.shape)

   .. tab:: JAX

      .. code-block:: python

         import jax
         import jax.numpy as jnp
         import transformer_engine.jax.flax as te_flax

         x = jnp.ones((8, 16), dtype=jnp.float32)
         layer = te_flax.DenseGeneral(features=16)
         variables = layer.init(jax.random.PRNGKey(0), x)

         y = jax.jit(layer.apply)(variables, x)
         y.block_until_ready()

         print(y.shape)
         print(jax.devices())

The snippets should complete on a GPU and report an output shape of
``(8, 16)``. They are smoke checks, not substitutes for comparison with a
native-framework reference.

Choose validation after a change
--------------------------------

Start with the smallest check that can fail for the modified boundary, then
expand to the relevant unit and integration tests.

.. list-table::
   :header-rows: 1
   :widths: 31 32 37

   * - Change
     - First validation
     - Follow-up
   * - Framework-independent Python
     - Import the affected module and execute its focused behavior.
     - Test through both frameworks when the code is shared.
   * - PyTorch Python
     - PyTorch smoke computation above.
     - Focused test under ``tests/pytorch/``.
   * - JAX Python
     - JAX smoke computation above.
     - Focused test under ``tests/jax/``.
   * - Framework binding C++
     - Print the selected framework library path and execute the affected
       operation.
     - Framework-specific numerical or integration test.
   * - Common C++ or CUDA
     - Load the common library and execute the affected operation through one
       frontend.
     - Focused ``tests/cpp/`` test plus every affected frontend test.
   * - Build or packaging logic
     - Build in a clean directory or environment and inspect artifact paths.
     - Matching wheel test under ``qa/L0_pytorch_wheel/`` or
       ``qa/L0_jax_wheel/``.
   * - Optional distributed component
     - Import with the component enabled and inspect the build log.
     - Focused multi-GPU launcher from the appropriate ``qa/`` class.

For numerical changes, compare outputs and gradients with the native framework
reference rather than checking only that execution completed. For example,
``tests/pytorch/test_numerics.py``
(``test_linear_accuracy``) is the reference pattern for a PyTorch linear
layer.

Use runnable examples
---------------------

Programs under ``examples/`` are useful after the focused smoke and unit
tests pass. They exercise realistic compositions such as MNIST models,
Transformer encoders, FSDP integration, expert parallelism, and communication
overlap.

Examples are not the primary correctness oracle:

* they may require extra packages, datasets, or multiple GPUs;
* they cover an end-to-end configuration rather than every boundary case;
* many are intended to demonstrate execution rather than compare against a
  high-precision or native-framework reference.

Use ``examples/pytorch/`` or ``examples/jax/`` for runnable standalone
programs. The material under ``docs/examples/`` is tutorial-oriented and
may be embedded in documentation or notebooks. Test commands and CI class
selection belong in :doc:`../testing_and_engineering_quality/index`.
