#!/usr/bin/env python3
"""
Step 7 — Run sample inference and log results to ClearML.

This script:
  1. Connects to ClearML and initialises a task of type "inference"
  2. Retrieves the registered Llama 3.3 70B model from the registry
  3. Sends 3 sample prompts through LiteLLM → vLLM → H200 GPUs
  4. Logs prompts, responses, and token-usage metrics to ClearML
  5. Uploads the full results as a ClearML artifact

Prerequisites:
  pip install clearml
  clearml-init   # configure cluster ClearML endpoint once

Usage:
  source ../config.env
  # Then set the two vars below, or export them alongside config.env:
  export LITELLM_URL="http://<NODE_IP>:<NODE_PORT>"   # from Step 6 verify
  export LITELLM_MASTER_KEY="sk-..."                  # matches config.env
  python3 inference_task.py
"""
import os
import json
import urllib.request
from clearml import Task, InputModel

# ── Read from environment (set by config.env + exports above) ──────────────
LITELLM_URL = os.environ.get("LITELLM_URL", "").rstrip("/")
API_KEY     = os.environ.get("LITELLM_MASTER_KEY", "")

if not LITELLM_URL:
    raise SystemExit(
        "ERROR: LITELLM_URL is not set.\n"
        "Export it before running:\n"
        "  export LITELLM_URL=http://<NODE_IP>:<NODE_PORT>"
    )
if not API_KEY:
    raise SystemExit(
        "ERROR: LITELLM_MASTER_KEY is not set.\n"
        "Source config.env or export it manually."
    )

# ── ClearML task ────────────────────────────────────────────────────────────
print("Initialising ClearML task...")
task = Task.init(
    project_name="production-llm",
    task_name="llama-3.3-70b-inference-demo",
    task_type=Task.TaskTypes.inference,
    reuse_last_task_id=False,
)
logger = task.get_logger()

# Retrieve the model registered in Step 4
model = InputModel(name="llama-3.3-70b-instruct", project="production-llm")
logger.report_text(f"Using model: {model.name}  |  ID: {model.id}")
logger.report_text(f"Endpoint: {LITELLM_URL}")

# ── Sample prompts ───────────────────────────────────────────────────────────
prompts = [
    "Explain transformer self-attention in one paragraph.",
    "What is tensor parallelism and why is it needed for large models?",
    "How does vLLM's continuous batching differ from static batching?",
]

results = []
for i, prompt in enumerate(prompts, start=1):
    print(f"\nQuery {i}/{len(prompts)}: {prompt[:60]}...")
    payload = {
        "model": "llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"{LITELLM_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())

    text  = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})

    # Log to ClearML
    logger.report_text(f"[Query {i}] Prompt:\n{prompt}\n\nResponse:\n{text}")
    logger.report_scalar("Token usage", "prompt_tokens",     usage.get("prompt_tokens", 0),     i)
    logger.report_scalar("Token usage", "completion_tokens", usage.get("completion_tokens", 0), i)
    logger.report_scalar("Token usage", "total_tokens",      usage.get("total_tokens", 0),      i)

    results.append({"prompt": prompt, "response": text, "tokens": usage})
    print(f"  Done. Tokens used: {usage}")

# Upload full results as artifact
task.upload_artifact("inference_results", artifact_object=results)
task.close()

print("\nAll queries complete.")
print("Open ClearML UI → production-llm → Tasks → llama-3.3-70b-inference-demo")
print("You should see: prompts/responses in the log, token-usage graphs, and the inference_results artifact.")
