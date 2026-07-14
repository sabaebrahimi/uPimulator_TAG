#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Newer huggingface_hub/httpx rejects proxy URLs with "socks://".
# Normalize to "socks5://" for compatibility.
normalize_proxy_var() {
  local var_name="$1"
  local value="${!var_name:-}"
  if [[ -n "${value}" && "${value}" == socks://* ]]; then
    local fixed="socks5://${value#socks://}"
    export "${var_name}=${fixed}"
    echo "[calibrate_bert] Normalized ${var_name} from socks:// to socks5://"
  fi
}

for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  normalize_proxy_var "${proxy_var}"
done

MODEL_DIR="${HOME}/models/bert"
OUTPUT_DIR="${MODEL_DIR}/serialization_dir"
TOKENIZED_DATASET_DIR="${SCRIPT_DIR}/bert_bookcorpus.token.128"
QUICK_MODE="${QUICK_MODE:-1}"

TRAIN_BS="${TRAIN_BS:-}"
EVAL_BS="${EVAL_BS:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
ENABLE_TRACKING="${ENABLE_TRACKING:-}"

if [[ "${QUICK_MODE}" == "1" ]]; then
  : "${TRAIN_BS:=1}"
  : "${EVAL_BS:=1}"
  : "${NUM_EPOCHS:=1}"
  : "${CHECKPOINT_STEPS:=100}"
  : "${MAX_TRAIN_STEPS:=200}"
  : "${ENABLE_TRACKING:=0}"
  echo "[calibrate_bert] QUICK_MODE=1 -> using lightweight settings."
else
  : "${TRAIN_BS:=6}"
  : "${EVAL_BS:=6}"
  : "${NUM_EPOCHS:=20}"
  : "${CHECKPOINT_STEPS:=2000}"
  : "${ENABLE_TRACKING:=1}"
fi

if [[ "${FORCE_CPU:-0}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  echo "[calibrate_bert] FORCE_CPU=1 -> running on CPU."
else
  # Preflight CUDA to avoid crashing later with "no kernel image is available".
  # If CUDA kernels are incompatible with the GPU, we fall back to CPU.
  if ! python - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)

if not torch.cuda.is_available():
    sys.exit(1)

try:
    x = torch.tensor([1.0], device="cuda")
    y = x + 1
    torch.cuda.synchronize()
except Exception:
    sys.exit(1)

sys.exit(0)
PY
  then
    export CUDA_VISIBLE_DEVICES=""
    echo "[calibrate_bert] CUDA preflight failed; falling back to CPU."
    echo "[calibrate_bert] Set FORCE_CPU=1 to silence this check."
  fi
fi

if [[ ! -d "${TOKENIZED_DATASET_DIR}" ]]; then
  echo "[calibrate_bert] Tokenized dataset not found: ${TOKENIZED_DATASET_DIR}"
  echo "[calibrate_bert] Generating tokenized BookCorpus dataset..."
  python ./bookcorpus_generator.py \
    --store_tokenized_datasets "${TOKENIZED_DATASET_DIR}"
fi

CMD=(
  python ./examples/run_lutrize_mlm_no_trainer.py
  --model_type bert
  --dataset_name bookcorpus
  --dataset_config_name plain_text
  --model_name_or_path "${MODEL_DIR}"
  --per_device_train_batch_size "${TRAIN_BS}"
  --per_device_eval_batch_size "${EVAL_BS}"
  --log_steps 5
  --output_dir "${OUTPUT_DIR}"
  --learning_rate 1e-3
  --reconstruct_rate 1e-3
  --max_seq_length 128
  --num_train_epochs "${NUM_EPOCHS}"
  --checkpointing_steps "${CHECKPOINT_STEPS}"
  --centroid_requires_grad
  --vec_len 2
  --ncentroid 16
  --nsharecodebook 1
  --distance_p 2.0
  --load_tokenized_datasets "${TOKENIZED_DATASET_DIR}"
)

if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  CMD+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

if [[ "${ENABLE_TRACKING}" == "1" ]]; then
  CMD+=(--with_tracking --report_to tensorboard)
fi

"${CMD[@]}"
