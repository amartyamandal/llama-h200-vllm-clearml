#!/usr/bin/env bash
# deploy-all.sh — Full end-to-end deployment of Llama 3.3 70B on H200 + vLLM + ClearML
#
# Usage:
#   cp ../config.env.example ../config.env
#   # Edit ../config.env with your cluster values
#   bash deploy-all.sh
#
# What this script does (in order):
#   1. Sources config.env and validates required variables
#   2. Creates the Kubernetes namespace
#   3. Creates the HF credentials secret
#   4. Creates the model-weights PVC
#   5. Runs the download Job (waits for completion)
#   6. Registers the model in ClearML
#   7. Deploys vLLM and waits until healthy
#   8. Deploys LiteLLM proxy
#   9. Prints the LiteLLM access URL and runs a smoke test
#
# Each step prints a VERIFY gate. The script exits on any error.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load config ──────────────────────────────────────────────────────────────
CONFIG="$REPO_ROOT/config.env"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: $CONFIG not found."
    echo "Run: cp $REPO_ROOT/config.env.example $CONFIG  then fill in your values."
    exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG"

# ── Validate required variables ──────────────────────────────────────────────
REQUIRED=(
    MY_NAMESPACE MY_STORAGE_CLASS MY_TENSOR_PARALLEL_SIZE
    MY_GPU_TAINT_KEY HF_TOKEN LITELLM_MASTER_KEY
    MY_CLEARML_API_URL MY_CLEARML_ACCESS_KEY MY_CLEARML_SECRET_KEY
    MY_VLLM_IMAGE MY_LITELLM_IMAGE MY_PYTHON_IMAGE
)
MISSING=()
for VAR in "${REQUIRED[@]}"; do
    if [[ -z "${!VAR:-}" ]]; then
        MISSING+=("$VAR")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: The following required variables are not set in config.env:"
    printf '  - %s\n' "${MISSING[@]}"
    exit 1
fi

echo "============================================================"
echo " Deploying Llama 3.3 70B on H200 GPUs — vLLM + ClearML"
echo " Namespace        : $MY_NAMESPACE"
echo " Tensor parallel  : $MY_TENSOR_PARALLEL_SIZE"
echo " Storage class    : $MY_STORAGE_CLASS"
echo " vLLM image       : $MY_VLLM_IMAGE"
echo " LiteLLM image    : $MY_LITELLM_IMAGE"
echo " Python image     : $MY_PYTHON_IMAGE"
echo " Internet access  : ${MY_CLUSTER_HAS_INTERNET:-unknown}"
echo "============================================================"
echo ""

# ── Helper: apply a YAML template with env substitution ─────────────────────
apply_template() {
    local tmpl="$1"
    envsubst < "$tmpl" | kubectl apply -f -
}

# ── Step 0: namespace ────────────────────────────────────────────────────────
echo ">>> Creating namespace '$MY_NAMESPACE' (if not exists)..."
kubectl create namespace "$MY_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# ── Step 1: HF secret ────────────────────────────────────────────────────────
echo ""
echo ">>> Step 1: Hugging Face secret"
bash "$REPO_ROOT/01-secrets/create-hf-secret.sh"

# ── Step 2: PVC ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 2: PersistentVolumeClaim for model weights"
if kubectl get pvc llama-model-storage -n "$MY_NAMESPACE" &>/dev/null; then
    echo "    PVC already exists — skipping."
else
    apply_template "$REPO_ROOT/02-storage/model-storage-pvc.yaml"
    echo "    Waiting for PVC to bind..."
    kubectl wait pvc/llama-model-storage -n "$MY_NAMESPACE" \
        --for=jsonpath='{.status.phase}'=Bound --timeout=120s
fi
echo "    PVC status: $(kubectl get pvc llama-model-storage -n "$MY_NAMESPACE" -o jsonpath='{.status.phase}')"

# ── Step 3: Download job ──────────────────────────────────────────────────────
echo ""
echo ">>> Step 3: Download Llama 3.3 70B weights (~140 GB)"
if kubectl get job llama-download -n "$MY_NAMESPACE" &>/dev/null; then
    echo "    Download job already exists. Checking status..."
    SUCCEEDED=$(kubectl get job llama-download -n "$MY_NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")
    if [[ "$SUCCEEDED" != "1" ]]; then
        echo "    Job not yet succeeded. Following logs..."
        kubectl logs -f job/llama-download -n "$MY_NAMESPACE" || true
    else
        echo "    Download already complete."
    fi
else
    envsubst < "$REPO_ROOT/03-download/download-job.yaml" | kubectl apply -f -
    echo "    Following download logs (30-60 min for first run)..."
    kubectl logs -f job/llama-download -n "$MY_NAMESPACE"
fi
# Wait for job success
kubectl wait job/llama-download -n "$MY_NAMESPACE" \
    --for=condition=complete --timeout=7200s
echo "    Download complete."

# ── Step 4: ClearML model registration ───────────────────────────────────────
echo ""
echo ">>> Step 4: Registering model in ClearML"
export CLEARML_API_HOST="$MY_CLEARML_API_URL"
export CLEARML_WEB_HOST="${MY_CLEARML_WEB_URL:-}"
export CLEARML_FILES_HOST="${MY_CLEARML_FILES_URL:-}"
export CLEARML_API_ACCESS_KEY="$MY_CLEARML_ACCESS_KEY"
export CLEARML_API_SECRET_KEY="$MY_CLEARML_SECRET_KEY"
python3 "$REPO_ROOT/04-clearml/register_model.py" || {
    echo "    WARNING: ClearML registration failed. Check ClearML credentials."
    echo "    Continuing deployment..."
}

# ── Step 5: vLLM deployment ───────────────────────────────────────────────────
echo ""
echo ">>> Step 5: Deploying vLLM inference server"
apply_template "$REPO_ROOT/05-vllm/vllm-deployment.yaml"
echo "    Waiting for vLLM pod to become Ready (can take 3-5 min for model load)..."
kubectl rollout status deployment/vllm-llama-70b -n "$MY_NAMESPACE" --timeout=600s
echo "    Verifying health endpoint..."
kubectl exec deployment/vllm-llama-70b -n "$MY_NAMESPACE" -- \
    curl -sf http://localhost:8000/health && echo "    vLLM: healthy"

# ── Step 6: LiteLLM proxy ─────────────────────────────────────────────────────
echo ""
echo ">>> Step 6: Deploying LiteLLM proxy"

# Render and apply the ConfigMap
RENDERED_CONFIG=$(mktemp /tmp/litellm-config-XXXX.yaml)
envsubst < "$REPO_ROOT/06-litellm/litellm-config.yaml" > "$RENDERED_CONFIG"
kubectl create configmap litellm-config \
    --from-file=config.yaml="$RENDERED_CONFIG" \
    -n "$MY_NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
rm -f "$RENDERED_CONFIG"

apply_template "$REPO_ROOT/06-litellm/litellm-deployment.yaml"
echo "    Waiting for LiteLLM pod to become Ready..."
kubectl rollout status deployment/litellm-proxy -n "$MY_NAMESPACE" --timeout=120s

# ── Print access URL ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Deployment complete!"
echo "============================================================"
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc litellm-proxy -n "$MY_NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}')
LITELLM_URL="http://$NODE_IP:$NODE_PORT"
echo ""
echo "  LiteLLM URL : $LITELLM_URL"
echo "  API Key     : $LITELLM_MASTER_KEY"
echo ""
echo "  Quick smoke test:"
curl -s -X POST "$LITELLM_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d '{"model":"llama-3.3-70b-instruct","messages":[{"role":"user","content":"Reply with one word: working"}],"max_tokens":10}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Model says:', d['choices'][0]['message']['content'])" \
    || echo "  Smoke test failed — check vLLM logs: kubectl logs deployment/vllm-llama-70b -n $MY_NAMESPACE"
echo ""
echo "  To log a ClearML inference run:"
echo "    export LITELLM_URL=$LITELLM_URL"
echo "    python3 $REPO_ROOT/07-clearml-logging/inference_task.py"
