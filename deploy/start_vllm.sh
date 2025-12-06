#!/bin/bash
# vLLM 服务启动脚本（Qwen3-VL-4B-Instruct，FP16 基础版）

set -euo pipefail

# 固定禁用 flash-attn（避免编译或缺失时崩溃），使用 PyTorch SDPA。
# 如需开启 flash-attn 加速，未来手动改为：
#   export VLLM_USE_FLASH_ATTENTION=1
#   export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# 并确保已安装 flash-attn（pip install "flash-attn>=2.5.8" --no-build-isolation）
export VLLM_USE_FLASH_ATTENTION=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
echo "[vLLM] flash-attn disabled; using TORCH_SDPA backend. To enable later: set VLLM_USE_FLASH_ATTENTION=1 and VLLM_ATTENTION_BACKEND=FLASH_ATTN (after installing flash-attn)."

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
