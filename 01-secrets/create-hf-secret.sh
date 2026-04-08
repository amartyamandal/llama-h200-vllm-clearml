#!/usr/bin/env bash
# Step 1 — Create Hugging Face credentials secret
# Usage: source ../config.env && bash create-hf-secret.sh
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in config.env}"
: "${MY_NAMESPACE:?Set MY_NAMESPACE in config.env}"

echo "=== Step 1: Hugging Face Secret ==="

# Verify the token has access to the gated model
echo "Verifying HF token access to Llama 3.3 70B..."
python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
try:
    api.model_info("meta-llama/Llama-3.3-70B-Instruct", token="$HF_TOKEN")
    print("SUCCESS: Token has access to Llama 3.3 70B")
except Exception as e:
    print(f"FAILED: {e}")
    print("Go to https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct and accept the license.")
    exit(1)
EOF

# Check if secret already exists
if kubectl get secret hf-credentials -n "$MY_NAMESPACE" &>/dev/null; then
    echo "Secret 'hf-credentials' already exists in namespace '$MY_NAMESPACE'. Skipping."
else
    kubectl create secret generic hf-credentials \
        --from-literal=token="$HF_TOKEN" \
        -n "$MY_NAMESPACE"
    echo "Secret created."
fi

# Verify
kubectl get secret hf-credentials -n "$MY_NAMESPACE"
echo "=== Step 1 complete ==="
