..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Setup
=====

Precision debug tools for the Transformer Engine use `Nvidia-DL-Framework-Inspect <https://github.com/NVIDIA/nvidia-dlfw-inspect>`_ package from NVIDIA. 
Please refer to the Nvidia-DL-Framework-Inspect `documentation <https://github.com/NVIDIA/nvidia-dlfw-inspect/tree/main/docs>`_ for more details.
Below, we outline the steps for debug initialization.

initialize()
-----------

Must be called once on every rank in the global context to initialize Nvidia-DL-Framework-Inspect.

**Parameters**

- **config_file** (*str*, default=""): Path to the configuration YAML file containing features to enable and layer names. If one wants to run without the configuration file, pass ``""``.
- **feature_dirs** (*List[str] | str*): List of directories containing features to load and register. One needs to pass ``[/path/to/transformerengine/transformer_engine/debug/features]`` to use TE features.
- **logger** (*Union[BaseLogger, None]*, default=None): Logger for logging tensor statistics. Should adhere to ``BaseLogger`` from the `Nvidia-DL-Framework-Inspect <https://github.com/NVIDIA/nvidia-dlfw-inspect>`_ package.
- **log_dir** (*str*, default= "."): Directory path to hold ``debug_logs`` and ``debug_statistics_logs``.
- **tb_writer** (*TensorBoardWriter*, default=None): TensorBoard writer for logging.
- **default_logging_enabled** (*bool*, default=False): Enable default logging to the file.

.. code-block:: python

    import nvdlfw_inspect.api as debug_api

    debug_api.initialize(
        config_file="./config.yaml",
        feature_dirs=["/path/to/transformer_engine/debug/features"],
        log_dir="./log_dir")

set_tensor_reduction_group()
--------------------------

Needed only for logging tensor stats. In multi-GPU training, activation and gradient tensors are distributed across multiple nodes. This method lets you specify the group for the reduction of stats; see the `reduction group section <./4_distributed.rst#reduction-groups>`_ for more details.

If the tensor reduction group is not specified, then statistics are reduced across all nodes in the run.

**Parameters**

- **group** (torch.distributed.ProcessGroup): The process group across which tensors will be reduced to get stats.


.. code-block:: python

    import nvdlfw_inspect.api as debug_api

    # initialization
    # (...)

    pipeline_parallel_group = initialize_pipeline_parallel_group() 

    debug_api.set_tensor_reduction_group(pipeline_parallel_group)

    # training
    # (...)
    # activation/gradient tensor statistics are reduced along pipeline_parallel_group

set_weight_tensor_tp_group_reduce()
---------------------------------

By default, weight tensor statistics are reduced within the tensor parallel group. This function allows you to disable that behavior; for more details, see `reduction group section <./4_distributed.rst#reduction-groups>`_.

This method is not provided by the ``debug_api``, but by the ``transformer_engine.debug``.

**Parameters**

- **enabled** (*bool*, default=True): A boolean flag to enable or disable the reduction of weight tensor statistics within the tensor parallel group.


.. code-block:: python

    import nvdlfw_inspect.api as debug_api
    from transformer_engine.debug import set_weight_tensor_tp_group_reduce

    # initialization
    # (...)

    set_weight_tensor_tp_group_reduce(False)

    # training
    # (...)
    # weight tensor statistics are not reduced
