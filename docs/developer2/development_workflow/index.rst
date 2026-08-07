Development Workflow
====================

The expected path from identifying a change through implementation, validation,
review, and merge.

Planned coverage
----------------

* Preparing and scoping a change.
* Coding conventions for each implementation language.
* Formatting, linting, commits, and pull requests.
* Contribution, ownership, review, and security expectations.
* Assessing change impact across APIs, frontends, tests, packaging, and
  documentation.

Documentation conventions
-------------------------

Technical chapters should use the following structure where it is useful:

#. Purpose and responsibilities.
#. Relevant files and symbols.
#. Concepts and terminology.
#. Execution or data flow.
#. Invariants and constraints.
#. Common change points.
#. Validation procedures.
#. Common pitfalls.
#. Related documentation.

Not every chapter needs every part. Conceptual, procedural, and reference
material should remain distinct enough that readers can find an explanation,
perform a task, or look up a fact without reading unrelated content.

Command examples should state:

* the directory in which they are run;
* prerequisites and significant environment assumptions;
* placeholders that the reader must replace; and
* the expected result or success criterion.

Pages should distinguish mandatory requirements from recommendations and
examples. Platform, framework, hardware, and version restrictions should be
stated next to the guidance they qualify.

.. TODO: Describe the shortest reliable path from a clean checkout to a
   reviewable change, then identify any workflows that need dedicated pages.
