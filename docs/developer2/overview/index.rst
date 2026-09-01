Developer Guide Overview
========================

Transformer Engine spans several implementation layers, framework
integrations, and hardware-specific paths. This guide helps contributors
understand how those pieces fit together, make changes safely, and determine
how those changes should be validated.

It complements the user guide and API reference. Those documents explain how
to use Transformer Engine and describe its public interfaces; this guide
focuses on the implementation, its design decisions, and the workflows used to
develop and maintain it.

Where to start
--------------

You do not need to read the guide from beginning to end. Start with the section
that best matches what you are trying to do.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Goal
     - Start with
   * - Prepare a first contribution
     - :doc:`Setup, Build, and Run <../setup_build_and_run/index>`, followed by
       :doc:`Development Workflow <../development_workflow/index>`
   * - Understand the codebase
     - :doc:`Project and Architecture <../project_and_architecture/index>`
   * - Modify an existing subsystem
     - :doc:`Implementation Architecture <../implementation_architecture/index>`
   * - Add a new capability
     - :doc:`Extending Transformer Engine <../extending_transformer_engine/index>`
   * - Test a change or investigate a failure
     - :doc:`Testing and Engineering Quality
       <../testing_and_engineering_quality/index>`
   * - Understand why a durable design choice was made
     - :doc:`Design Decisions <../design_decisions/index>`
   * - Perform packaging, documentation, or release work
     - :doc:`Maintainer Operations <../maintainer_operations/index>`
   * - Find a command, term, support boundary, or external reference
     - :doc:`Reference <../reference/index>`

How the documentation fits together
-----------------------------------

The `nightly documentation <https://nvidia.github.io/TransformerEngine/>`_
follows the in-development version of Transformer Engine. The `released
documentation
<https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_
covers supported releases. Both contain the user guide and API reference.

Contribution requirements are maintained in `CONTRIBUTING.rst
<https://github.com/NVIDIA/TransformerEngine/blob/main/CONTRIBUTING.rst>`_ at
the repository root. Changes between releases are recorded on `GitHub Releases
<https://github.com/NVIDIA/TransformerEngine/releases>`_ and in the published
`release notes
<https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html>`_.

Keeping the guide useful
------------------------

Reference material should have one maintained home. Developer pages should
link to public APIs, configuration options, compatibility tables, and release
information instead of maintaining separate copies. A small amount of context
may still be repeated when it is needed to explain an internal design.

The implementation and its tests are the primary references for internal
behavior. Build, test, and automation behavior is defined by the corresponding
project configuration and scripts. If these sources disagree with the
documentation, verify the intended behavior and treat the disagreement as a
defect rather than documenting both versions.

When pointing into the source tree, use a repository-relative file path and a
stable symbol such as a function, class, module, or build target. Avoid exact
line numbers, which drift as the implementation changes. For example, prefer
``path/to/file.py:ClassName.method`` over a bare filename or line reference.

Documentation status
--------------------

This replacement developer guide is being assembled incrementally. Skeleton
pages and explicit TODO comments describe planned coverage; they are not
normative project guidance.

The guide is versioned with the rest of the documentation and should describe
the corresponding repository revision. The documentation landing page displays
a warning when the nightly, in-development version is shown and links to the
released documentation. Stale guidance should be corrected or reported like
any other project defect.
