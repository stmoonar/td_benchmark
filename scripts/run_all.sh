#!/usr/bin/env bash
# One-command remote validation and report-grade benchmark suite.
#
# Usage:
#   VENV_ACTIVATE=/path/to/venv/bin/activate \
#   TD_PYTHON=/path/to/triton-distributed/python \
#   bash scripts/run_all.sh
# Set GPUS explicitly to bypass automatic idle-GPU selection.
#
# The script intentionally unsets CUDA_DEVICE_MAX_CONNECTIONS so CUDA uses its
# runtime default.  All benchmark schemes use FP8 weights (128x128 blocks),
# group128 token activations, and precomputed routing outside the timed region.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate}"
TD_PYTHON="${TD_PYTHON:-/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python}"
GPUS="${GPUS:-}"
NPROC="${NPROC:-4}"
GPU_IDLE_MAX_MEMORY_MB="${GPU_IDLE_MAX_MEMORY_MB:-1024}"
GPU_IDLE_MAX_UTIL_PERCENT="${GPU_IDLE_MAX_UTIL_PERCENT:-10}"
MASTER_PORT="${MASTER_PORT:-29500}"
NVSHMEM_SIZE="${NVSHMEM_SIZE:-8589934592}"
WARMUP="${WARMUP:-50}"
REPEAT="${REPEAT:-50}"
SWEEP_WARMUP="${SWEEP_WARMUP:-20}"
SWEEP_REPEAT="${SWEEP_REPEAT:-50}"
RUN_SWEEPS="${RUN_SWEEPS:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-${ROOT}/results/server_suite_${STAMP}}"
ZIP_PATH="${ZIP_PATH:-${ROOT}/results/server_suite_${STAMP}.zip}"
SCHEMES="[a1,a2,b1,b2,b3,c3,c4,c5]"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
else
  echo "ERROR: VENV_ACTIVATE does not exist: ${VENV_ACTIVATE}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}:${TD_PYTHON}:${PYTHONPATH:-}"
unset CUDA_DEVICE_MAX_CONNECTIONS

if [[ -z "${GPUS}" ]]; then
  GPU_SELECTION="auto"
  GPUS="$(python "${ROOT}/scripts/select_idle_gpus.py" \
    --max-memory-mb "${GPU_IDLE_MAX_MEMORY_MB}" \
    --max-util-percent "${GPU_IDLE_MAX_UTIL_PERCENT}")"
else
  GPU_SELECTION="explicit"
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -ne "${NPROC}" ]]; then
  echo "ERROR: selected GPUS=${GPUS} has ${#GPU_LIST[@]} devices, but NPROC=${NPROC}" >&2
  exit 2
fi

mkdir -p "${SUITE_DIR}/suite_logs" "${SUITE_DIR}/environment"

exec > >(tee -a "${SUITE_DIR}/suite_logs/run_all.log") 2>&1

echo "suite_dir=${SUITE_DIR}"
echo "zip_path=${ZIP_PATH}"
echo "gpus=${GPUS} nproc=${NPROC} warmup/repeat=${WARMUP}/${REPEAT}"
echo "gpu_selection=${GPU_SELECTION} idle_memory_mb<=${GPU_IDLE_MAX_MEMORY_MB} idle_util_percent<=${GPU_IDLE_MAX_UTIL_PERCENT}"
echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS-unset}"

{
  date --iso-8601=seconds
  uname -a
  python --version
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())"
  python -c "import vllm; print('vllm import ok')"
  python -c "import triton_dist; print('triton_dist', triton_dist.__file__)"
  git -C "${ROOT}" rev-parse HEAD
  git -C "${ROOT}" status --short
  echo "gpu_selection=${GPU_SELECTION}"
  echo "selected_gpus=${GPUS}"
  echo "idle_memory_threshold_mb=${GPU_IDLE_MAX_MEMORY_MB}"
  echo "idle_utilization_threshold_percent=${GPU_IDLE_MAX_UTIL_PERCENT}"
} > "${SUITE_DIR}/environment/runtime.txt"
python -m pip freeze > "${SUITE_DIR}/environment/pip_freeze.txt"
git -C "${ROOT}" diff --binary > "${SUITE_DIR}/environment/source.patch"
nvidia-smi -q > "${SUITE_DIR}/environment/nvidia_smi_q.txt"
nvidia-smi topo -m > "${SUITE_DIR}/environment/nvidia_smi_topo.txt"

echo "[1/5] Python syntax and unit tests"
python -m compileall -q "${ROOT}/moe_bench" "${ROOT}/tests" \
  "${ROOT}/scripts/package_server_results.py" "${ROOT}/scripts/select_idle_gpus.py"
python -m pytest "${ROOT}/tests" -q

common_overrides=(
  --isolate-schemes
  --set "dist.cuda_visible_devices=${GPUS}"
  --set "dist.nproc=${NPROC}"
  --set "dist.master_port=${MASTER_PORT}"
  --set "run.output_root=${SUITE_DIR}"
  --set "env.NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SIZE}"
  --set "schemes.enabled=${SCHEMES}"
)

run_benchmark() {
  local config="$1"
  local tag="$2"
  local warmup="$3"
  local repeat="$4"
  shift 4
  echo "RUN ${tag}: ${config} warmup=${warmup} repeat=${repeat}"
  python -m moe_bench.cli "${ROOT}/${config}" \
    "${common_overrides[@]}" \
    --set "run.tag=${tag}" \
    --set "run.warmup=${warmup}" \
    --set "run.repeat=${repeat}" \
    "$@"
}

echo "[2/5] Four-GPU smoke"
run_benchmark configs/smoke.yaml "${STAMP}_smoke" 5 5

echo "[3/5] Report-grade v1 anchor"
run_benchmark configs/run_v1_production.yaml "${STAMP}_v1" "${WARMUP}" "${REPEAT}"

echo "[4/5] Shape and routing sweeps"
if [[ "${RUN_SWEEPS}" == "1" ]]; then
  run_benchmark configs/sweep_v1.yaml "${STAMP}_sweep_v1" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}"
  run_benchmark configs/sweep_und.yaml "${STAMP}_sweep_und" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}"
  run_benchmark configs/sweep_gen.yaml "${STAMP}_sweep_gen" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}"
else
  echo "RUN_SWEEPS=${RUN_SWEEPS}; sweeps skipped"
fi

echo "[5/5] Validate and package"
python "${ROOT}/scripts/package_server_results.py" \
  --suite-dir "${SUITE_DIR}" \
  --output "${ZIP_PATH}"

echo "DONE"
echo "Download: ${ZIP_PATH}"
echo "Checksum: ${ZIP_PATH}.sha256"
