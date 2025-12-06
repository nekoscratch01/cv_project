#!/bin/bash
# vLLM 服务启动脚本（Qwen3-VL-4B-Instruct，FP16 基础版）

set -euo pipefail

# 强制使用 flash-attn；若缺失则终止并提示安装
python - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("flash_attn") is None:
    sys.stderr.write(
        "[ERROR] flash-attn not installed. Install first, e.g.:\n"
        "pip install \"flash-attn>=2.5.8\" --no-build-isolation\n"
    )
    sys.exit(1)
PY

export VLLM_USE_FLASH_ATTENTION=1
export VLLM_ATTENTION_BACKEND=flash
echo "[vLLM] flash-attn detected, using flash backend."

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
