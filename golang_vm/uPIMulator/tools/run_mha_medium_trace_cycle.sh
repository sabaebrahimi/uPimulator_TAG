#!/usr/bin/env bash
set -euo pipefail

cd /home/saba/master/uPimulator_TAG

export PATH="/home/saba/master/tools/go/bin:${PATH}"
export GOCACHE=/tmp/cosim-go-cache-trace
export PIMDL_PYTHON_LIB=/home/saba/.pyenv/versions/3.10.14/lib
export PIMDL_NUM_TASKLETS=16
export COSIM_XFER_MODE=trace_cycle

python3 golang_vm/uPIMulator/tools/cosim_results.py \
    --model mha \
    --dpus 1 \
    --config PIM-DL-ASPLOS/inference-engine/configs/cosim_mha_medium_1dpu.yaml \
    --results-dir cosim_results/mha_medium \
    --resume

python3 golang_vm/uPIMulator/tools/cosim_results.py \
    --model mha \
    --dpus 8 \
    --config PIM-DL-ASPLOS/inference-engine/configs/cosim_mha_medium_1dpu.yaml \
    --results-dir cosim_results/mha_medium \
    --resume
