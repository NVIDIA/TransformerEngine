# Continuous Integration for Transformer Engine

## Transformer Engine

Transformer Engine (TE) is a library for accelerating Transformer
models on NVIDIA GPUs. Most of its software development takes place in
a publicly-accessible repository at
https://github.com/NVIDIA/transformerengine. A secondary
NVIDIA-internal repository at
https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine
is used for continuous integration and sensitive tasks.

## `te_ci` branch

The
[`te_ci` branch](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/tree/te_ci)
contains a GitLab CI configuration file (`.gitlab-ci.yml`) and various
helper scripts and Dockerfiles. Be advised it is somewhat hacky and
has been adapted ad hoc from other CI workflows at NVIDIA (e.g.
[MXNet](https://gitlab-master.nvidia.com/dl/dgx/mxnet) and
[PyTorch](https://gitlab-master.nvidia.com/dl/dgx/pytorch)).

When a CI pipeline is launched with `te_ci`, it first runs a single
job called `create_branch`. This job merges `te_ci` with some
user-specified TE branch from GitHub or GitLab, pushes the merged
branch to GitLab, and launches the main CI pipeline on that branch.
The main CI pipeline involves several stages:

- `build base`: Build TE Docker container for each DL framework. TE is
  installed and a copy of the source code is available at
  `/opt/transformerengine`.
- `build devel`: Build TE development containers. They are similar to
  the base containers, but the TE source code at
  `/opt/transformerengine` has Git metadata and objects from the
  `te_ci` branch.
- `build stage`: Build TE QA containers. They are similar to the
  development containers, but with special handling for QA workflows.
  This stage is only enabled with TE releases, i.e. GitHub branches
  with the name `release_v*`.
- `test`: Run tests. Each test is a shell script within the
  [`qa` directory](https://github.com/NVIDIA/TransformerEngine/tree/main/qa).
  The tests are run on a variety of GPU systems.
- `finalize`: Upload logs to Blossom if needed.

The build stages will push Docker containers to the container registry
at `gitlab-master.nvidia.com/dl/transformerengine/transformerengine`
with a tag in the form `<TAG_ROOT>-<DLFW>-py3-<build stage>` (e.g.
`test_main_12345678_CI-pytorch-py3-base`).

## Launching a pipeline manually

The easiest way to launch a pipeline is by navigating to the GitLab
repository and accessing "CI/CD"/"Pipelines"/"Run pipeline".
Alternatively, navigate to
[this address](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/new).
Make sure to set the branch name to `te_ci`. The following variables
may also be set:

| Key                   | Default value | Description                                               |
|-----------------------|---------------|-----------------------------------------------------------|
| `GH_BRANCH`           | `main`        | GitHub branch                                             |
| `GH_PR`               |               | GitHub PR number (overrides `GH_BRANCH`)                  |
| `GL_MR`               |               | GitLab MR number (overrides `GH_BRANCH`)                  |
| `CORE_IMAGE`          |               | Base Docker container for core build                      |
| `PYTORCH_IMAGE`       | Nightly build | Base Docker container for PyTorch build                   |
| `JAX_IMAGE`           |               | Base Docker container for JAX build                       |
| `BUILD_CORE`          | `1`           | Enable core build and tests                               |
| `BUILD_PYTORCH`       | `1`           | Enable PyTorch build and tests                            |
| `BUILD_JAX`           | `1`           | Enable JAX build and tests                                |
| `RUN_L0_TESTS`        | `1`           | Run L0 tests automatically (otherwise run tests manually) |
| `RUN_L1_TESTS`        | `0`           | Run L1 tests automatically (otherwise run tests manually) |
| `RUN_L2_TESTS`        | `0`           | Run L2 tests automatically (otherwise run tests manually) |
| `RUN_L3_TESTS`        | `0`           | Run L3 tests automatically (otherwise run tests manually) |
| `TAG_ROOT`            |               | Base string for Docker container tags                     |
| `SEND_SLACK_MESSAGE`  | `0`           | Send Slack message on CI completion                       |

## Launching a pipeline from GitHub

One of the main use-cases for the CI pipeline is to validate GitHub
PRs. If an approved TE developer comments "/te-ci" on a PR and
approves a Duo push notification, it will launch a CI pipeline and
include the results in the PR status. Including DL frameworks in the
comment, e.g. "/te-ci pytorch" will restrict the pipeline to those
frameworks.

This workflow involves a stack with three components: GitHub Actions,
Blossom, and GitLab CI/CD pipelines.

### GitHub Actions

The TE GitHub repository is configured to run GitHub actions on PRs
(see
[`.github/workflows`](https://github.com/NVIDIA/TransformerEngine/tree/main/.github/workflows)).
The
["TE-CI Trigger" workflow](https://github.com/NVIDIA/TransformerEngine/blob/main/.github/workflows/trigger-ci.yml)
checks for PR comments that contain "/te-ci", makes sure it is from an
approved GitHub user, and triggers a Blossom pipeline. The
["TE-CI Logs" workflow](https://github.com/NVIDIA/TransformerEngine/blob/main/.github/workflows/upload-ci-logs.yml)
is triggered by Blossom and updates the PR status with the CI results.

### Blossom

Blossom is an NVIDIA-internal CI/CD service (see the
[Confluence page](https://confluence.nvidia.com/display/BLOS/Getting+Started))
that provides each project a Jenkins instance within a centralized
Kubernetes cluster. The TE instance can be accessed at
https://prod.blsm.nvidia.com/dl-fw-transformerengine-ci/.
Note that Jenkins pipelines are implemented in Groovy and that TE uses
helper functions from
[this repository](https://gitlab-master.nvidia.com/ptredak/blossom-github-jenkins-lib/-/tree/te_ci).
The `te-ci` pipeline, triggered from GitHub, launches a GitLab
pipeline. The `te-ci-logs` pipeline, launched from GitLab, forwards
the CI results to GitHub.

### GitLab CI/CD pipelines

We launch GitLab CI pipelines on the `te_ci` branch. See the above
discussion for more information.

### Adding a developer

- Give their GitHub account write access to the TE GitHub repository.
- Add their GitHub username to
  [`.github/workflows/trigger-ci.yml`](https://github.com/NVIDIA/TransformerEngine/blob/main/.github/workflows/trigger-ci.yml)
  in the TE GitHub repository.
- File an NVBug ticket to "IPP - Blossom Support" with the TE GitHub
  repository, their GitHub username, and their NVIDIA email. See an
  [example](https://nvbugspro.nvidia.com/bug/4476510) and the
  [documentation for Blossom GitHub support](https://confluence.nvidia.com/display/BLOS/Github+Support+User+Documentation).
- For optional Slack notification support
  - Add their GitHub to NVIDIA username mapping in `.gitlab-ci.yml` in the `get_nvidia_username_from_github_username` function in the `send_result_to_slack` job
  - Add their NVIDIA username to Slack user ID mapping in `.gitlab-ci.yml` in the `get_slack_user_id_from_nvidia_username` function in the `send_result_to_slack` job

## QA workflow

QA expects the TE team to provide test containers that are compatible
with their scripts. The QA containers should have the form
`gitlab-master.nvidia.com/dl/transformerengine/transformerengine:<monthly release version>-<DL framework>-py3-qa`.
To build these containers, launch a GitLab CI pipeline on the `te_ci`
branch with the following variables:

| Key             | Value                                                                           |
|-----------------|---------------------------------------------------------------------------------|
| `GH_BRANCH`     | `release_v<latest TE version>`                                                  |
| `PYTORCH_IMAGE` | `gitlab-master.nvidia.com:5005/dl/dgx/pytorch:<release version>-py3-base-amd64` |
| `TAG_ROOT`      | TE release version                                                              |
| `RUN_L0_TESTS`  | `1` (optional)                                                                  |
| `RUN_L1_TESTS`  | `1` (optional)                                                                  |
| `RUN_L2_TESTS`  | `1` (optional)                                                                  |
| `RUN_L3_TESTS`  | `1` (optional)                                                                  |

This is somewhat of a messy process since other framework teams may
delay building their containers or may need to rebuild. When
containers have been built, we should notify QA on the
`#swdl-fw-builds` Slack channel.
