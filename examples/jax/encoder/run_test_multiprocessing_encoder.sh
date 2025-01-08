# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

for i in $(seq 0 $(($NUM_GPUS-1)))
do
  pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py::TestEncoder::test_te_bf16 --num-process=$NUM_GPUS --process-id=$i &
done
wait

for i in $(seq 0 $(($NUM_GPUS-1)))
do
  pytest -c $TE_PATH/tests/jax/pytest.ini -v $TE_PATH/examples/jax/encoder/test_multiprocessing_encoder.py::TestEncoder::test_te_fp8 --num-process=$NUM_GPUS --process-id=$i &
done
wait
