#!/usr/bin/env bash
# =============================================================================
#   MoE Bench v2 — 一键部署 + 运行脚本
#
#   用法：
#     # 1. 把整个 new_start/ 目录同步到目标机器
#     # 2. 修改下面的配置区
#     # 3. bash scripts/run_all.sh
#
#   前置依赖：
#     - Python 3.12+ venv，已装 torch (CUDA)、vllm、triton
#     - triton_dist 源码目录（clean upstream @ 1b9dc71a 或兼容版本）
#     - NVSHMEM 运行时（c 组需要）
#     - 4 张同型号 GPU
# =============================================================================
set -euo pipefail

# ─── 配置区（按你的机器改这里）─────────────────────────────────────────────────
VENV_ACTIVATE="${VENV_ACTIVATE:-/data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate}"
NEW_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TD_PYTHON="${TD_PYTHON:-/data/cinnzhang_vllm_td_test/triton_distributed-TD+Flux/python}"
GPUS="${GPUS:-9,11,13,15}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
NVSHMEM_SIZE="${NVSHMEM_SIZE:-8589934592}"    # 8GB，大 M 可能需要更多
MAX_CONNECTIONS="${MAX_CONNECTIONS:-32}"       # 32=允许 overlap 并发；1=单队列串行
WARMUP="${WARMUP:-50}"
REPEAT="${REPEAT:-50}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)_bench}"
# ─── 配置区结束 ────────────────────────────────────────────────────────────────

source "$VENV_ACTIVATE"
export PYTHONPATH="${NEW_ROOT}:${TD_PYTHON}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "================================================================"
echo "  MoE Bench v2 一键运行"
echo "  NEW_ROOT     : $NEW_ROOT"
echo "  TD_PYTHON    : $TD_PYTHON"
echo "  GPUs         : $GPUS (nproc=$NPROC)"
echo "  MAX_CONN     : $MAX_CONNECTIONS"
echo "  warmup/repeat: $WARMUP/$REPEAT"
echo "  tag          : $TAG"
echo "================================================================"

# ─── 环境检查 ──────────────────────────────────────────────────────────────────
echo ""
echo "[CHECK] Python packages..."
python -c "import torch; print(f'  torch {torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}')"
python -c "import vllm; print(f'  vllm ok')"
python -c "import triton_dist; print(f'  triton_dist ok')"
python -c "import moe_bench; print(f'  moe_bench ok')"
echo "[CHECK] All imports OK"
echo ""

# ─── 单元测试（快速冒烟）───────────────────────────────────────────────────────
echo "[TEST] Running verification gate tests..."
python -m pytest "${NEW_ROOT}/tests/test_verify_gate.py" -q || { echo "FAIL: gate tests"; exit 1; }
echo ""

# ─── v1 全 10 组 ──────────────────────────────────────────────────────────────
echo "[RUN] v1 shape, 10 schemes, M=5120..."
python -m moe_bench.cli "${NEW_ROOT}/configs/run_v1_production.yaml" \
  --isolate-schemes \
  --set "dist.cuda_visible_devices=${GPUS}" \
  --set "dist.nproc=${NPROC}" \
  --set "dist.master_port=${MASTER_PORT}" \
  --set "run.warmup=${WARMUP}" \
  --set "run.repeat=${REPEAT}" \
  --set "run.tag=${TAG}_v1" \
  --set "env.NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SIZE}" \
  --set "env.CUDA_DEVICE_MAX_CONNECTIONS=${MAX_CONNECTIONS}" \
  --set "schemes.enabled=[a1,a2,b1,b2,b3,c1,c2,c3,c4,c5]" \
  --set "schemes.c3.tunables.fp8_rs_enabled=true" \
  --set "schemes.c4.tunables.n_chunks_rs=32" \
  --set "schemes.c4.tunables.block_quant=true" \
  --set "schemes.c5.tunables.n_chunks_rs=32" \
  --set "schemes.c5.tunables.block_quant=true"

echo ""
echo "[RUN] v1 M-sweep (7 points x 10 schemes)..."
python -m moe_bench.cli "${NEW_ROOT}/configs/run_v1_production.yaml" \
  --isolate-schemes \
  --set "dist.cuda_visible_devices=${GPUS}" \
  --set "dist.nproc=${NPROC}" \
  --set "dist.master_port=${MASTER_PORT}" \
  --set "run.warmup=${WARMUP}" \
  --set "run.repeat=${REPEAT}" \
  --set "run.tag=${TAG}_v1_sweep" \
  --set "env.NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SIZE}" \
  --set "env.CUDA_DEVICE_MAX_CONNECTIONS=${MAX_CONNECTIONS}" \
  --set "schemes.enabled=[a1,a2,b1,b2,b3,c1,c2,c3,c4,c5]" \
  --set "schemes.c3.tunables.fp8_rs_enabled=true" \
  --set "schemes.c4.tunables.n_chunks_rs=32" \
  --set "schemes.c4.tunables.block_quant=true" \
  --set "schemes.c5.tunables.n_chunks_rs=32" \
  --set "schemes.c5.tunables.block_quant=true" \
  --set 'sweep_axes=[{"path":"shape.M","values":[128,512,1024,2048,3072,4096,5120]}]'

echo ""
echo "================================================================"
echo "  完成！结果在 ${NEW_ROOT}/results/ 下"
echo "  grep PASS/FAIL: find results -name results.jsonl -exec grep -l FAIL {} \\;"
echo "================================================================"
