#!/bin/bash
# vLLM 服务启动脚本（Qwen3-VL-4B-Instruct，FP16 基础版）

set -euo pipefail

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --trust-remote-code \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-batched-tokens 4096 \
    --limit-mm-per-prompt '{"image":5}'

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
