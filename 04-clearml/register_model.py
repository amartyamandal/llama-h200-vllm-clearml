#!/usr/bin/env python3
"""
Step 4 — Register Llama 3.3 70B in ClearML model registry.

Prerequisites:
  pip install clearml
  clearml-init   # enter your cluster ClearML credentials once

Usage:
  python3 register_model.py

The script reads GPU_TYPE and TENSOR_PARALLEL_SIZE from environment variables
(set in config.env) so you don't need to edit this file directly.
"""
import os
from clearml import Task, OutputModel

GPU_TYPE           = os.environ.get("MY_GPU_TYPE", "H200 (141 GB HBM3e)")
TENSOR_PARALLEL    = int(os.environ.get("MY_TENSOR_PARALLEL_SIZE", "2"))
MIN_GPUS           = TENSOR_PARALLEL
RECOMMENDED_GPUS   = max(4, TENSOR_PARALLEL)

# Verify ClearML is reachable before creating the task
print("Connecting to ClearML...")
task = Task.init(
    project_name="production-llm",
    task_name="register-llama-3.3-70b",
    task_type=Task.TaskTypes.data_processing,
    reuse_last_task_id=False,
)

out_model = OutputModel(
    task=task,
    name="llama-3.3-70b-instruct",
    tags=["llama3", "70b", "instruct", "vllm", "production"],
    framework="PyTorch",
)

out_model.update_design(config_dict={
    "model_id": "meta-llama/Llama-3.3-70B-Instruct",
    "cluster_path": "/models/llama-3.3-70b",
    "format": "safetensors",
    "precision": "bfloat16",
    "context_length": 131072,
    "param_count": "70B",
    "inference_engine": "vLLM",
    "gpu_type": GPU_TYPE,
    "min_gpus": MIN_GPUS,
    "recommended_gpus": RECOMMENDED_GPUS,
})

task.close()
print(f"Model registered successfully.")
print(f"  Model ID : {out_model.id}")
print(f"  GPU type : {GPU_TYPE}  |  Tensor parallel : {TENSOR_PARALLEL}")
print("Open ClearML UI → Models → search 'llama-3.3-70b-instruct' to verify.")
