#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/swin_unetpp_attn.yaml"
  "configs/swin_unetpp_base.yaml"
  "configs/convnext_unetpp_attn.yaml"
  "configs/convnext_unetpp_base.yaml"
  "configs/swin_deeplab_attn.yaml"
  "configs/swin_deeplab_base.yaml"
  "configs/convnext_deeplab_attn.yaml"
  "configs/convnext_deeplab_base.yaml"
)

# Edit GPU_IDS for your machine. For 3xL40, keep this as (0 1 2).
GPU_IDS=(0 1 2)

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

num_gpus="${#GPU_IDS[@]}"
if [[ "${num_gpus}" -lt 1 ]]; then
  echo "GPU_IDS is empty. Please specify at least one GPU ID."
  exit 1
fi

declare -a GPU_PIDS

for idx in "${!CONFIGS[@]}"; do
  slot="$((idx % num_gpus))"
  gpu="${GPU_IDS[$slot]}"
  cfg="${CONFIGS[$idx]}"

  prev_pid="${GPU_PIDS[$slot]:-}"
  if [[ -n "${prev_pid}" ]]; then
    echo "GPU ${gpu} busy (pid=${prev_pid}). Waiting for it to finish..."
    wait "${prev_pid}"
  fi

  exp_name="$(basename "${cfg}" .yaml)"
  log_path="${LOG_DIR}/${exp_name}.gpu${gpu}.log"

  echo "Launching ${cfg} on GPU ${gpu} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train.py --config "${cfg}" > "${log_path}" 2>&1 &
  GPU_PIDS[$slot]="$!"
done

for pid in "${GPU_PIDS[@]}"; do
  wait "${pid}"
done

wait
echo "All experiments completed."
