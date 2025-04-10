#!/bin/bash

function error_exit() {
    echo "Error: $1"
    exit 1
}

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

function initialize_test_variables() {
    RET=0
    FAILED_CASES=""
}

function check_test_results() {
    if [ "$RET" -ne 0 ]; then
        echo "Error in the following sub-tests:$FAILED_CASES"
        exit 1
    fi
    echo "All tests passed"
    exit 0
}
