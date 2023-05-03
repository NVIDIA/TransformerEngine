#!/bin/bash

set -xe

APP=test_multiprocessing_encoder.py
#APP=mp_test.py
NHOSTS=${1:-$(nvidia-smi --list-gpus | wc -l)}

for id in $( seq 1 $((NHOSTS - 1)) );
do
    python $APP --num_process ${NHOSTS} --process_id ${id} &
    echo $!
done

python $APP --num_process ${NHOSTS} --process_id 0 &
echo $!

