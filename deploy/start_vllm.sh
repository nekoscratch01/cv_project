#!/bin/bash
# vLLM 服务启动脚本（Qwen3-VL-4B-Instruct，FP16 基础版）

set -euo pipefail

# 尝试使用 flash-attn；若缺失则回退 torch SDPA，并提示未来可安装加速。
if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("flash_attn") else 1)
PY
then
  export VLLM_USE_FLASH_ATTENTION=1
  export VLLM_ATTENTION_BACKEND=FLASH_ATTN
  echo "[vLLM] flash-attn detected, using FLASH_ATTN backend."
else
  export VLLM_USE_FLASH_ATTENTION=0
  export VLLM_ATTENTION_BACKEND=TORCH_SDPA
  echo "[vLLM] flash-attn not found, fallback to TORCH_SDPA. Install later for better throughput:"
  echo "pip install \"flash-attn>=2.5.8\" --no-build-isolation"
fi

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image":5}' \
    --enable-prefix-caching

# 量化版本示例（AWQ），需要对应模型
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-VL-4B-Instruct-AWQ \
#     --trust-remote-code \
#     --quantization awq \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --gpu-memory-utilization 0.90 \
#     --max-model-len 8192 \
#     --limit-mm-per-prompt image=5
