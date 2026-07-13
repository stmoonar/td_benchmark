#!/usr/bin/env bash
# Focused remote investigation: b2 chunk overlap vs c4 TD tile overlap.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate}"
GPUS="${GPUS:-}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29600}"
GPU_IDLE_MAX_MEMORY_MB="${GPU_IDLE_MAX_MEMORY_MB:-1024}"
GPU_IDLE_MAX_UTIL_PERCENT="${GPU_IDLE_MAX_UTIL_PERCENT:-10}"
NVSHMEM_SIZE="${NVSHMEM_SIZE:-8589934592}"
WARMUP="${WARMUP:-50}"
REPEAT="${REPEAT:-100}"
SWEEP_WARMUP="${SWEEP_WARMUP:-20}"
SWEEP_REPEAT="${SWEEP_REPEAT:-50}"
PROFILE_ITERS="${PROFILE_ITERS:-3}"
RUN_SWEEPS="${RUN_SWEEPS:-1}"
RUN_TORCH_PROFILE="${RUN_TORCH_PROFILE:-1}"
RUN_NSYS="${RUN_NSYS:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="${SUITE_DIR:-${ROOT}/results/tp_overlap_investigation_${STAMP}}"
ZIP_PATH="${ZIP_PATH:-${ROOT}/results/tp_overlap_investigation_${STAMP}.zip}"
CONFIG="configs/investigate_tp_overlap.yaml"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: VENV_ACTIVATE does not exist: ${VENV_ACTIVATE}" >&2
  exit 2
fi
source "${VENV_ACTIVATE}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset CUDA_DEVICE_MAX_CONNECTIONS
NVSHMEM_HOME_RESOLVED="${NVSHMEM_HOME:-$(python -c 'import nvidia.nvshmem; print(next(iter(nvidia.nvshmem.__path__)))')}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/results/.triton_cache_${STAMP}}"
PROFILE_TMPDIR="${PROFILE_TMPDIR:-${ROOT}/results/.profile_tmp_${STAMP}}"
export NVSHMEM_HOME="${NVSHMEM_HOME_RESOLVED}"
export NVSHMEM_LIBDEVICE_PATH="${NVSHMEM_HOME_RESOLVED}/lib"
export LD_LIBRARY_PATH="${NVSHMEM_HOME_RESOLVED}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TRITON_CACHE_DIR PROFILE_TMPDIR
export TMPDIR="${PROFILE_TMPDIR}"
mkdir -p "${SUITE_DIR}/suite_logs" "${SUITE_DIR}/environment" \
  "${SUITE_DIR}/profiles/nsys" "${SUITE_DIR}/analysis" \
  "${TRITON_CACHE_DIR}" "${PROFILE_TMPDIR}"
exec > >(tee -a "${SUITE_DIR}/suite_logs/run_investigation.log") 2>&1

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
  echo "ERROR: GPUS=${GPUS} has ${#GPU_LIST[@]} devices, NPROC=${NPROC}" >&2
  exit 2
fi

echo "suite_dir=${SUITE_DIR}"
echo "zip_path=${ZIP_PATH}"
echo "branch=$(git -C "${ROOT}" branch --show-current)"
echo "gpus=${GPUS} selection=${GPU_SELECTION} nproc=${NPROC}"
echo "comparison=b2_vs_c4 CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS-unset}"
echo "c4_consumer_candidate=BLOCK_M128_BLOCK_N64_WARPS8_STAGES2"
echo "nvshmem_home=${NVSHMEM_HOME}"
echo "triton_cache_dir=${TRITON_CACHE_DIR} tmpdir=${TMPDIR}"

{
  date --iso-8601=seconds
  uname -a
  python --version
  python -c "import sys, torch, vllm, triton_dist; print('python', sys.executable); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__); print('triton_dist', triton_dist.__file__)"
  git -C "${ROOT}" rev-parse HEAD
  git -C "${ROOT}" status --short
  echo "selected_gpus=${GPUS}"
  echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS-unset}"
  echo "NVSHMEM_HOME=${NVSHMEM_HOME}"
  echo "NVSHMEM_LIBDEVICE_PATH=${NVSHMEM_LIBDEVICE_PATH}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  echo "TMPDIR=${TMPDIR}"
  command -v nsys >/dev/null 2>&1 && nsys --version || echo "nsys=missing"
} > "${SUITE_DIR}/environment/runtime.txt"
nvidia-smi -q > "${SUITE_DIR}/environment/nvidia_smi_q.txt"
nvidia-smi topo -m > "${SUITE_DIR}/environment/nvidia_smi_topo.txt"
python -m pip freeze > "${SUITE_DIR}/environment/pip_freeze.txt"
git -C "${ROOT}" diff --binary > "${SUITE_DIR}/environment/source.patch"

common_overrides=(
  --isolate-schemes
  --set "dist.cuda_visible_devices=${GPUS}"
  --set "dist.nproc=${NPROC}"
  --set "dist.master_port=${MASTER_PORT}"
  --set "run.output_root=${SUITE_DIR}"
  --set "env.NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SIZE}"
)

run_case() {
  local tag="$1"
  local warmup="$2"
  local repeat="$3"
  shift 3
  echo "RUN case=${tag} warmup=${warmup} repeat=${repeat}"
  python -m moe_bench.cli "${ROOT}/${CONFIG}" \
    "${common_overrides[@]}" \
    --set "run.tag=${tag}" \
    --set "run.warmup=${warmup}" \
    --set "run.repeat=${repeat}" \
    "$@"
}

echo "[1/6] Focused unit checks"
python -m compileall -q "${ROOT}/moe_bench" "${ROOT}/scripts/analyze_tp_overlap.py"
python -m pytest "${ROOT}/tests/test_config.py" \
  "${ROOT}/tests/test_analyze_tp_overlap.py" \
  "${ROOT}/tests/test_worker_local_fallback.py" -q

echo "[2/6] Matched end-to-end anchors"
run_case "${STAMP}_paired_v1" "${WARMUP}" "${REPEAT}" \
  --set "schemes.enabled=[b2,c4]"
run_case "${STAMP}_paired_v1_no_shared" "${WARMUP}" "${REPEAT}" \
  --set "schemes.enabled=[b2,c4]" \
  --set "shape.shared_experts=0"

echo "[3/6] Scaling and routing sensitivity"
echo "RUN component preprocessing and quantization costs"
CUDA_VISIBLE_DEVICES="${GPUS}" \
NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SIZE}" \
NVSHMEM_REMOTE_TRANSPORT=none \
NVSHMEM_DISABLE_CUDA_VMM=1 \
torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="$((MASTER_PORT + 1))" \
  "${ROOT}/scripts/benchmark_tp_overlap_components.py" \
  --output "${SUITE_DIR}/analysis/component_benchmarks.json" \
  --gemm-block-n 64 \
  --warmup "${SWEEP_WARMUP}" \
  --repeat "${SWEEP_REPEAT}"

if [[ "${RUN_SWEEPS}" == "1" ]]; then
  run_case "${STAMP}_m_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[b2,c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"shape.M","values":[128,512,1024,2048,4096,5120]}]'
  run_case "${STAMP}_routing_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[b2,c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"routing.imbalance_kind","values":["none","zipf","hotspot"]}]'

  echo "[4/6] Granularity ablations"
  run_case "${STAMP}_b2_chunk_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[b2]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"schemes.b2.tunables.n_chunks_gateup","values":[1,2,4]},{"path":"schemes.b2.tunables.n_chunks_down","values":[1,2,4,8]}]'
  run_case "${STAMP}_c4_tile_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"schemes.c4.tunables.n_chunks_rs","values":[4,8,16,32]}]'
  run_case "${STAMP}_c4_gate_block_m_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"schemes.c4.tunables.gemm_block_m","values":[64,128]}]'
  run_case "${STAMP}_c4_gate_group_m_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"schemes.c4.tunables.gemm_group_size_m","values":[1,4,8]}]'
  run_case "${STAMP}_c4_gate_warps_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"schemes.c4.tunables.gemm_num_warps","values":[4,8]}]'
  run_case "${STAMP}_c4_gate_stages_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"schemes.c4.tunables.gemm_num_stages","values":[2,3,4]}]'
elif [[ "${RUN_SWEEPS}" != "joint" && "${RUN_SWEEPS}" != "gemm" ]]; then
  echo "RUN_SWEEPS=${RUN_SWEEPS}; diagnostic sweeps skipped"
fi

if [[ "${RUN_SWEEPS}" == "1" || "${RUN_SWEEPS}" == "joint" || "${RUN_SWEEPS}" == "gemm" ]]; then
  run_case "${STAMP}_c4_gate_occupancy_sweep" "${SWEEP_WARMUP}" "${SWEEP_REPEAT}" \
    --set "schemes.enabled=[c4]" \
    --set "shape.shared_experts=0" \
    --set 'sweep_axes=[{"path":"schemes.c4.tunables.gemm_block_m","values":[64,128]},{"path":"schemes.c4.tunables.gemm_block_n","values":[32,64,128]},{"path":"schemes.c4.tunables.gemm_num_warps","values":[4,8]},{"path":"schemes.c4.tunables.gemm_num_stages","values":[1,2]}]'
fi

echo "[5/6] Profiles"
if [[ "${RUN_TORCH_PROFILE}" == "1" ]]; then
  run_case "${STAMP}_torch_profile_b2" 5 5 \
    --set "schemes.enabled=[b2]" \
    --set "shape.shared_experts=0" \
    --set "profile.torch_profiler=true"
  run_case "${STAMP}_torch_profile_c4" 5 5 \
    --set "schemes.enabled=[c4]" \
    --set "shape.shared_experts=0" \
    --set "profile.torch_profiler=true"
fi

run_nsys_case() {
  local scheme="$1"
  local label="$2"
  shift 2
  local output="${SUITE_DIR}/profiles/nsys/${label}_steady_state"
  local nsys_args=(
    profile
    --force-overwrite=true
    --trace=cuda,nvtx,osrt
    --sample=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --output="${output}"
  )
  if nsys profile --help 2>&1 | grep -q -- '--trace-fork-before-exec'; then
    nsys_args+=(--trace-fork-before-exec=true)
  fi
  echo "NSYS scheme=${scheme} output=${output}.nsys-rep"
  if ! nsys "${nsys_args[@]}" python -m moe_bench.cli "${ROOT}/${CONFIG}" \
      "${common_overrides[@]}" \
      --set "run.tag=${STAMP}_nsys_${label}" \
      --set "run.warmup=5" \
      --set "run.repeat=5" \
      --set "schemes.enabled=[${scheme}]" \
      --set "shape.shared_experts=0" \
      --set "env.MOE_BENCH_NSYS_CAPTURE=1" \
      --set "env.MOE_BENCH_PROFILE_ITERS=${PROFILE_ITERS}" \
      "$@" \
      2>&1 | tee "${output}_console.log"; then
    echo "WARNING: nsys capture failed for ${scheme}"
    echo "nsys capture failed" > "${output}_FAILED.txt"
    return 0
  fi

  local report="${output}.nsys-rep"
  if [[ ! -f "${report}" && -f "${output}.qdrep" ]]; then
    report="${output}.qdrep"
  fi
  if [[ -f "${report}" ]]; then
    for summary in cuda_gpu_kern_sum cuda_api_sum nvtx_gpu_proj_sum; do
      if ! nsys stats --report "${summary}" --format csv "${report}" \
          > "${output}_${summary}.csv"; then
        echo "WARNING: nsys stats report ${summary} unavailable for ${scheme}"
      fi
    done
  fi
}

if [[ "${RUN_NSYS}" == "1" ]]; then
  if command -v nsys >/dev/null 2>&1; then
    run_nsys_case c4 c4
    if [[ "${RUN_SWEEPS}" == "gemm" ]]; then
      run_nsys_case c4 c4_bn128 --set "schemes.c4.tunables.gemm_block_n=128"
    fi
    run_nsys_case b2 b2
  else
    echo "WARNING: nsys is unavailable; torch profiles are still included"
    echo "nsys command not found" > "${SUITE_DIR}/profiles/nsys/MISSING.txt"
  fi
fi

echo "[6/6] Analyze, validate, and package"
python "${ROOT}/scripts/analyze_tp_overlap.py" --suite-dir "${SUITE_DIR}"
python "${ROOT}/scripts/package_server_results.py" \
  --suite-dir "${SUITE_DIR}" \
  --output "${ZIP_PATH}" \
  --package-invalid

echo "DONE zip=${ZIP_PATH}"
echo "SHA256 ${ZIP_PATH}.sha256"
