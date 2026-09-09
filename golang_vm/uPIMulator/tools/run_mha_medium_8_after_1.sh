#!/usr/bin/env bash
set -euo pipefail

previous_unit=upimulator-mha-medium-1dpu.service
while systemctl is-active --quiet "${previous_unit}"; do
    sleep 30
done

previous_status="$(systemctl show "${previous_unit}" --property=ExecMainStatus --value)"
if [[ "${previous_status}" != "0" ]]; then
    echo "Not starting 8-DPU run: ${previous_unit} exited with status ${previous_status}."
    exit 1
fi

export PATH="/home/saba/master/tools/go/bin:${PATH}"
export PIMDL_PYTHON_LIB=/home/saba/.pyenv/versions/3.10.14/lib
export PIMDL_NUM_TASKLETS=16
export COSIM_XFER_MODE=fixed_bw

exec python3 golang_vm/uPIMulator/tools/cosim_results.py \
    --model mha \
    --dpus 8 \
    --config PIM-DL-ASPLOS/inference-engine/configs/cosim_mha_medium_1dpu.yaml \
    --results-dir cosim_results/mha_medium \
    --resume
