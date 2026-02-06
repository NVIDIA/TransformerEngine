FROM nvcr.io/nvidia/pytorch:25.12-py3

RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
    && apt-get install -y sccache \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y transformer-engine
