#!/bin/bash
################################################################################
# Build Docker Containers
################################################################################

usage() {
cat <<EOF
$0

Builds containers based on present state of repository. Can be called either
locally or from within gitlab CI. By default, base and devel images are built
and pushed to the registry. Alternatively, if the --stage option is specified,
then the qa and stage images are built from an existing base image.

All options are optional and if not specified some will be looked up from CI
environment variables (see description below for details).

Usage:
  $0 [--OPTION=[VAL] ...]

  OPTION         DESCRIPTION
  --pyver        Python version (3.6 or 2.7) included in built container.
                 Default is 3.6.
  --tag-root     Name used in docker tags. If not specified, default is taken
                 from CI_COMMIT_REF_NAME or, if that doesn't exist, from the
                 current git branch. Any '-devel' suffix is dropped.
  --docker-img   Name for docker images not including the tag (cf. --tag-root).
                 If not specified, defaults to CI_REGISTRY_IMAGE or, if that is
                 not found, to 'mxlocal'.
  --pipeline     Optional versioning ID to include in image tags. Default is
                 taken from CI_PIPELINE_ID if that exists. Otherwise, a git
                 hash is used.
  --from-image   Base image to use for build.
  --devops-image build-scripts image to use for build
  --[no]pull     Whether to update cached docker images from registries during
                 build or prefer local caches. Default is PULL.
  --[no]push     Whether or not to push built containers and 'stage' git tag to
                 remote registry and repository. Default is to NOT push.
  --virtual      Do not push non-versioned image tags and overwrite CUDA libs
                 by versions specified in the environment variables
                 CUDA_VERSION, NCCL_VERSION, CUBLAS_VERSION, CUDNN_VERSION, and
                 TRT_VERSION.
  --mode         Can be combination of 'b' to build base image, 'd' to build
                 devel image, 's' to build qa and stage images, and c to contamer
                 scan base image. Default is 'bd'. Either 'c', 'd' or 's' assume
                 that a base image exists in the local or remote registry.
  --framework    Framework extensions to build with Transformer Engine. Options
                 include 'pytorch', 'jax', 'core', 'all'. Framework should
                 already be installed in base image.
  --arch         CPU architecture. Options inlcude 'amd64', 'arm64'.

EOF
}

set -o pipefail

valcheck() {
  if [[ -z "$2" ]]; then
    usage
    echo "ABORT: $1 expects a value."
    exit 1
  fi
}

# Defaults
MODE=bd
PYVER=3.8
PUSH=0
PULL=1
VIRTUAL=0
DOCKER_IMG="${CI_REGISTRY_IMAGE:-mxlocal}"
ARCH=amd64

# Parse options
while [[ $# -gt 0 ]]; do
  if [[ "$1" =~ ^-.*= ]]; then
    key="${1%%=*}"
    val="${1#*=}"
    val_separate=0
  else
    key="$1"
    val="$2"
    val_separate=1
  fi
  key="$(echo "$key" | tr '[:upper:]' '[:lower:]')"

  case "$key" in
    --pyver)
      PYVER="$val"
      if [[ "$PYVER" != "3.6" && "$PYVER" != "3.8" ]]; then
        usage
        echo "ABORT: Illegal python version $PYVER"
        exit 1
      fi
      shift
      ;;
    --tag-root)
      valcheck "$key" "$val"
      TAG_ROOT="$val"
      shift $((val_separate+1))
      ;;
    --docker-img)
      valcheck "$key" "$val"
      DOCKER_IMG="$val"
      shift $((val_separate+1))
      ;;
    --pipeline)
      valcheck "$key" "$val"
      PIPELINE="$val"
      shift $((val_separate+1))
      ;;
    --from-image)
      valcheck "$key" "$val"
      FROM_IMAGE="$val"
      shift $((val_separate+1))
      ;;
    --framework)
      valcheck "$key" "$val"
      FRAMEWORK="$val"
      shift $((val_separate+1))
      ;;
    --devops-image)
      valcheck "$key" "$val"
      FROM_SCRIPTS_IMAGE="$val"
      shift $((val_separate+1))
      ;;
    --pull)
      PULL=1
      shift
      ;;
    --nopull)
      PULL=0
      shift
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --nopush)
      PUSH=0
      shift
      ;;
    --virtual)
      VIRTUAL=1
      shift
      ;;
    --novirtual)
      VIRTUAL=0
      shift
      ;;
    --mode)
      valcheck "$key" "$val"
      MODE="$val"
      shift $((val_separate+1))
      ;;
    --arch)
      valcheck "$key" "$val"
      ARCH="$val"
      shift $((val_separate+1))
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      echo "ABORT: Unrecognize option $key."
      exit 1
      ;;
  esac
done

# Determine which images we are building
BUILD_BASE=0
BUILD_DEVEL=0
BUILD_STAGE=0
CONTAMER_SCAN=0
for (( i=0; i<${#MODE}; i++ )); do
  case "${MODE:$i:1}" in
    b|B)
      BUILD_BASE=1
      ;;
    d|D)
      BUILD_DEVEL=1
      ;;
    s|S)
      BUILD_STAGE=1
      ;;
    c|C)
      CONTAMER_SCAN=1
      ;;
    *)
      usage
      echo ABORT: Illegal mode flag, $MODE
      exit 1
      ;;
  esac
done

# Run script from repo root
cd "$(dirname "${BASH_SOURCE[0]}")"

# Looking up the git branch to set a default value may cause errors,
# so we do it after option parsing.
if [[ -z "$TAG_ROOT" ]]; then
  if [[ -n "${CI_COMMIT_REF_NAME}" ]]; then
    TAG_ROOT="${CI_COMMIT_REF_NAME}"
  else
    GIT_LINE="$(git branch | grep \*)"
    if [[ $? -ne 0 ]]; then
      echo "ABORT: Failed to obtain branch name from git."
      echo "Please specify --tag-root option manually."
      exit 1
    fi
    if [[ "$GIT_LINE" =~ HEAD[[:space:]]detached[[:space:]]at ]]; then
      TEMP="$(echo "$GIT_LINE" | cut -d ' '  -f5)"
      TAG_ROOT="${TEMP%)}"
    else
      TAG_ROOT="$(echo "$GIT_LINE" | cut -d ' ' -f2)"
    fi
  fi
fi
TAG_ROOT="${TAG_ROOT%-devel}"

COMMIT_SHA="${CI_COMMIT_SHA:-$(git rev-parse HEAD)}"
COMMIT_SHA=${COMMIT_SHA:-NONE}
if [[ -z "${COMMIT_SHA}" ]]; then
  echo "Failed to obtain commit hash."
  exit 1
fi

TRANSFORMERENGINE_HASH=$(git rev-parse HEAD)

# The pipeline ID will fall back to the git hash.
if [[ -z "$PIPELINE" ]]; then
  if [[ -n "$CI_PIPELINE_ID" ]]; then
    PIPELINE="$CI_PIPELINE_ID"
  else
    PIPELINE="$(echo "$COMMIT_SHA" | cut -c1-8)"

    # Add TE hash to PIPELINE for one-off builds
    if [[ "$ONE_OFF_BUILD" -eq 1 ]]; then
      PIPELINE="${PIPELINE}.$(echo "$TRANSFORMERENGINE_HASH" | cut -c1-8)"
    fi
  fi
fi

# Other vars

JOB_ID=${CI_JOB_ID:-NONE}

FROM_IMAGE_ARG=$(docker pull "${FROM_IMAGE}" >& /dev/null && echo "--build-arg FROM_IMAGE_NAME=${FROM_IMAGE}")
[[ -n "${FRAMEWORK}" ]] \
    && FRAMEWORK_ARG="--build-arg FRAMEWORK=${FRAMEWORK}" \
    || FRAMEWORK_ARG=""

BRANCH_NAME_SLUG=$(echo "${TAG_ROOT}" | sed 's/[^A-Za-z0-9._\-]/../g')
TAG_ATTRIB="${FRAMEWORK}-py${PYVER%.*}"
IMAGE_NAME_ROOT="${DOCKER_IMG}:${BRANCH_NAME_SLUG}-${TAG_ATTRIB}"
VER_IMAGE_NAME_ROOT="${IMAGE_NAME_ROOT}.${PIPELINE}"

[[ "$BUILD_BASE" -eq 1 ]] && echo "BUILDING BASE IMAGE"
[[ "$BUILD_DEVEL" -eq 1 ]] && echo "BUILDING DEVEL IMAGE"
[[ "$BUILD_STAGE" -eq 1 ]] && echo "STAGING RELEASE"
echo "(versioned stem: $VER_IMAGE_NAME_ROOT)"
echo "Building at rev $COMMIT_SHA"
echo "TransformerEngine sources at rev $TRANSFORMERENGINE_HASH"
if [[ "$VIRTUAL" -eq 1 ]]; then
  echo "Build is VIRTUAL"
else
  echo "Build is not VIRTUAL"
fi
[[ "$PUSH" -eq 1 ]] && echo "Push ENABLED" || echo "Push DISABLED"
[[ "$PULL" -eq 1 ]] && echo "Force PULL ENABLED" || echo "Force PULL DISABLED"


# TODO Maybe hide build warnings
# TODO Better Error checking everywhere

if [[ "$BUILD_BASE" -eq 1 ]]; then

  REGISTRY="${CI_REGISTRY:-gitlab-master.nvidia.com:5005}"
  ########
  ## TEMPORARY WAR https://jirasw.nvidia.com/browse/DLR-316 - do not merge to master
  FROM_SCRIPTS_IMAGE="${FROM_SCRIPTS_IMAGE:-${REGISTRY}/dl/devops/build-scripts:bringup}"
  ########
  PULL_FLAG=""
  CACHE_FROM="--cache-from type=local,src=/tmp/docker-cache"
  CACHE_TO="--cache-to type=local,dest=/tmp/docker-cache,mode=max"
  if [[ $PULL -eq 1 ]]; then
    PULL_FLAG="--pull"
    MASTER_BASE_IMAGE_NAME="${DOCKER_IMG}:main-${TAG_ATTRIB}-base-${ARCH}"
    docker pull "${MASTER_BASE_IMAGE_NAME}"
    docker pull "${IMAGE_NAME_ROOT}-base-${ARCH}"
    docker pull "${FROM_SCRIPTS_IMAGE}"
    CACHE_FROM="--cache-from type=registry,ref=${DOCKER_IMG}"
    CACHE_TO="--cache-to type=registry,ref=${DOCKER_IMG},mode=max"
  fi

  # Docker arguments for image tags
  TAGS="-t ${VER_IMAGE_NAME_ROOT}-base-${ARCH}"
  if [[ "${VIRTUAL}" -ne 1 && "${ONE_OFF_BUILD}" -eq "0" ]]; then
      TAGS="-t ${IMAGE_NAME_ROOT}-base-${ARCH} ${TAGS}"
  fi

  export DOCKER_CLI_EXPERIMENTAL=enabled
  docker buildx create --name buildkit --node node_${CI_RUNNER_ID} --config dlfw-ci/buildkitd.toml \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=10485760 --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=10485760 --use
  docker buildx inspect --bootstrap
  if [[ "${PUSH}" -eq 1 ]]; then PUSH_ARG="--push";  else PUSH_ARG="--load"; fi
  docker buildx build $PULL_FLAG $FROM_FLAG \
      ${CACHE_FROM} \
      ${CACHE_TO} \
      ${TAGS} \
      --platform "linux/${ARCH}" \
      --provenance=false \
      $FROM_IMAGE_ARG \
      $FRAMEWORK_ARG \
      -f Dockerfile.base ${PUSH_ARG} .
  RV=$?
  echo "exit code from the previous command -> $RV"
  docker stop buildx_buildkit_node_${CI_RUNNER_ID}
  docker rm buildx_buildkit_node_${CI_RUNNER_ID}
  docker buildx rm buildkit
  exit $RV
fi

if [[ "$BUILD_DEVEL" -eq 1 ]]; then

  REGISTRY="${CI_REGISTRY:-gitlab-master.nvidia.com:5005}"
  ########
  ## TEMPORARY WAR https://jirasw.nvidia.com/browse/DLR-316 - do not merge to master
  FROM_SCRIPTS_IMAGE="${FROM_SCRIPTS_IMAGE:-${REGISTRY}/dl/devops/build-scripts:bringup}"
  ########
  BASE_IMAGE="${VER_IMAGE_NAME_ROOT}-base-${ARCH}"

  if [[ "$PULL" -eq 1 ]]; then
    docker pull "${FROM_SCRIPTS_IMAGE}"
    docker pull "${BASE_IMAGE}"
  fi

  # Docker arguments for image tags
  TAGS="-t ${VER_IMAGE_NAME_ROOT}-devel-${ARCH}"
  if [[ "${VIRTUAL}" -ne 1 && "${ONE_OFF_BUILD}" -eq "0" ]]; then
      TAGS="-t ${IMAGE_NAME_ROOT}-devel-${ARCH} ${TAGS}"
  fi

  if [[ "$VIRTUAL" -eq 1 ]]; then
      OVERRIDES="--build-arg VIRTUAL=${VIRTUAL}"
      if [[ -n "${CUDA_DRIVER_VERSION}" ]]; then
        OVERRIDES="$OVERRIDES --build-arg CUDA_DRIVER_VERSION_OVERRIDE=$CUDA_DRIVER_VERSION"
      fi
      if [[ -n "${CUDA_VERSION}" ]]; then
        OVERRIDES="$OVERRIDES --build-arg CUDA_VERSION_OVERRIDE=$CUDA_VERSION"
      fi
      if [[ -n "${NCCL_VERSION}" ]]; then
        OVERRIDES="$OVERRIDES --build-arg NCCL_VERSION_OVERRIDE=$NCCL_VERSION"
      fi
      if [[ -n "${CUBLAS_VERSION}" ]]; then
        OVERRIDES="$OVERRIDES --build-arg CUBLAS_VERSION_OVERRIDE=$CUBLAS_VERSION"
      fi
      if [[ -n "${CUDNN_VERSION}" ]]; then
        OVERRIDES="$OVERRIDES --build-arg CUDNN_VERSION_OVERRIDE=$CUDNN_VERSION"
      fi
      if [[ -n "${TRT_VERSION}" ]]; then
        OVERRIDES="$OVERRIDES --build-arg TRT_VERSION_OVERRIDE=$TRT_VERSION"
      fi
  fi

  docker build --progress=plain --network=host \
      ${TAGS} \
      --platform "linux/${ARCH}" \
      $FRAMEWORK_ARG \
      --build-arg "FROM_SCRIPTS_IMAGE=${FROM_SCRIPTS_IMAGE}" \
      --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
      --build-arg "NVIDIA_PIPELINE_ID=${PIPELINE}" \
      --build-arg "NVIDIA_BUILD_ID=${JOB_ID}" \
      $OVERRIDES -f Dockerfile.devel .
  if [[ $? -ne 0 ]]; then
    echo Failed to build devel container
    exit 1
  fi
  if [[ "${PUSH}" -eq 1 ]]; then
    if [[ "${VIRTUAL}" -ne 1 ]]; then
      docker push "${IMAGE_NAME_ROOT}-devel-${ARCH}"
      if [[ $? -ne 0 ]]; then
        echo "Failed to push ${IMAGE_NAME_ROOT}-devel-${ARCH}"
        exit 1
      fi
    fi
    docker push "${VER_IMAGE_NAME_ROOT}-devel-${ARCH}"
    if [[ $? -ne 0 ]]; then
      echo "Failed to push ${VER_IMAGE_NAME_ROOT}-devel-${ARCH}"
      exit 1
    fi
  fi

fi

if [[ "$BUILD_STAGE" -eq 1 ]]; then
  BASE_IMAGE="${VER_IMAGE_NAME_ROOT}-base-${ARCH}"
  DEVEL_IMAGE_NAME="${VER_IMAGE_NAME_ROOT}-devel-${ARCH}"
  QA_IMAGE_NAME="${IMAGE_NAME_ROOT}-qa-${ARCH}"
  STAGE_IMAGE_NAME="${IMAGE_NAME_ROOT}-stage-${ARCH}"
  QA_IMAGE_NAME_VERSIONED="${VER_IMAGE_NAME_ROOT}-qa-${ARCH}"
  DEVEL_IMAGE_NAME_SQRL="${DEVEL_IMAGE_NAME/$CI_REGISTRY/$SQRL_REGISTRY_URL}"
  QA_IMAGE_NAME_SQRL="${QA_IMAGE_NAME/$CI_REGISTRY/$SQRL_REGISTRY_URL}"
  STAGE_IMAGE_NAME_SQRL="${STAGE_IMAGE_NAME/$CI_REGISTRY/$SQRL_REGISTRY_URL}"

  if [[ "$PULL" -eq 1 ]]; then
    docker pull "${BASE_IMAGE}"
    docker pull "${DEVEL_IMAGE_NAME}"
  fi
  docker tag "$BASE_IMAGE" "${STAGE_IMAGE_NAME}"
  # Create xx.yy-qa image
  docker build --progress=plain -t "${QA_IMAGE_NAME}" -f Dockerfile.qa \
      --platform "linux/${ARCH}" \
      --build-arg "FROM_IMAGE_DEVEL=${DEVEL_IMAGE_NAME}" \
      --build-arg "FROM_IMAGE=${BASE_IMAGE}" \
      --build-arg "FRAMEWORK=${FRAMEWORK}" \
      --build-arg "NVIDIA_PIPELINE_ID=${PIPELINE}" \
      --build-arg "NVIDIA_BUILD_ID=${JOB_ID}" \
      --network=host .
  if [[ $? -ne 0 ]]; then
    echo ABORT Failed to create qa image
    exit 1
  fi

  docker tag "${QA_IMAGE_NAME}" "${QA_IMAGE_NAME_VERSIONED}"
  docker tag "${DEVEL_IMAGE_NAME}" "${DEVEL_IMAGE_NAME_SQRL}"
  docker tag "${QA_IMAGE_NAME}" "${QA_IMAGE_NAME_SQRL}"
  docker tag "${STAGE_IMAGE_NAME}" "${STAGE_IMAGE_NAME_SQRL}"

  if [[ "${PUSH}" -eq 1 ]]; then
    docker push "${STAGE_IMAGE_NAME}"
    docker push "${QA_IMAGE_NAME}"
    docker push "${QA_IMAGE_NAME_VERSIONED}"
    docker push "${DEVEL_IMAGE_NAME_SQRL}"
    docker push "${QA_IMAGE_NAME_SQRL}"
    docker push "${STAGE_IMAGE_NAME_SQRL}"

    # push a xx.yy-stage tag that records the commit we staged from
    ORIGIN_GIT="$(git remote get-url origin)"
    ORIGIN_TMP="${ORIGIN_GIT##*@}"
    ORIGIN_SSH=ssh://git@${ORIGIN_TMP/\//:12051\/}
    git push --delete $ORIGIN_SSH refs/tags/${TAG_ROOT}-stage
    git push $ORIGIN_SSH HEAD:refs/tags/${TAG_ROOT}-stage
  fi

fi # END MODE == DEVEL/STAGE

# Running contamer-scan
if [[ "$CONTAMER_SCAN" -eq 1 ]]; then
   CI_TMPDIR="${PWD}/${CI_JOB_ID}"
   CONTAMERDIR="${CI_TMPDIR}/contamer" && echo "CONTAMERDIR=${CONTAMERDIR}"
   git config --global advice.detachedHead false
   git clone --branch master --single-branch "https://gitlab-ci-token:${CI_JOB_TOKEN}@gitlab-master.nvidia.com/dl/devops/contamer.git" "${CONTAMERDIR}"
   cd "${CONTAMERDIR}"
   pip3 install -r requirements.txt
   docker pull "${BASE_IMAGE_NAME_VERSIONED}-base-${ARCH}"
   docker images | grep anchore
   python3 contamer.py -ls "${BASE_IMAGE_NAME_VERSIONED}-base-${ARCH}"
   if [[ $? -ne 0 ]]; then
       docker images | grep anchore
       echo "${BASE_IMAGE_NAME_VERSIONED}-base-${ARCH} Failed contamer scan"
       exit 1
   fi
   docker images | grep anchore
fi
