#!/bin/bash

set -xe

APP=test_multiprocessing_encoder.py
#APP=mp_test.py
NHOSTS=${1:-$(nvidia-smi --list-gpus | wc -l)}

for id in $( seq 0 $((NHOSTS - 1)) );
do
    python $APP --num-process ${NHOSTS} --process-id ${id} --dry-run &
    echo $!
done

