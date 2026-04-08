#!/usr/bin/env bash
# verify-stack.sh — Quick health check of the full deployed stack
#
# Usage:
#   source ../config.env && bash verify-stack.sh
set -euo pipefail

: "${MY_NAMESPACE:?Source config.env first}"

PASS=0; FAIL=0
check() {
    local label="$1"; shift
    if "$@" &>/dev/null; then
        echo "  [PASS] $label"
        ((PASS++))
    else
        echo "  [FAIL] $label"
        ((FAIL++))
    fi
}

echo "============================================================"
echo " Stack verification — namespace: $MY_NAMESPACE"
echo "============================================================"

echo ""
echo "--- Kubernetes resources ---"
check "PVC llama-model-storage Bound" \
    bash -c "[[ \$(kubectl get pvc llama-model-storage -n $MY_NAMESPACE -o jsonpath='{.status.phase}') == 'Bound' ]]"
check "Secret hf-credentials exists" \
    kubectl get secret hf-credentials -n "$MY_NAMESPACE"
check "Download job succeeded" \
    bash -c "[[ \$(kubectl get job llama-download -n $MY_NAMESPACE -o jsonpath='{.status.succeeded}' 2>/dev/null) == '1' ]]"
check "vLLM deployment Running (1/1)" \
    bash -c "[[ \$(kubectl get deployment vllm-llama-70b -n $MY_NAMESPACE -o jsonpath='{.status.readyReplicas}') == '1' ]]"
check "LiteLLM deployment Running (1/1)" \
    bash -c "[[ \$(kubectl get deployment litellm-proxy -n $MY_NAMESPACE -o jsonpath='{.status.readyReplicas}') == '1' ]]"

echo ""
echo "--- Service health ---"
check "vLLM health endpoint" \
    kubectl exec deployment/vllm-llama-70b -n "$MY_NAMESPACE" -- \
        curl -sf http://localhost:8000/health

NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc litellm-proxy -n "$MY_NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}')
LITELLM_URL="http://$NODE_IP:$NODE_PORT"

check "LiteLLM health endpoint ($LITELLM_URL)" \
    curl -sf "$LITELLM_URL/health"

echo ""
echo "--- GPU allocation ---"
echo "  GPU pods in namespace:"
kubectl get pods -n "$MY_NAMESPACE" -o wide | grep -v "Completed" || true
echo ""
echo "  GPU memory (nvidia-smi):"
kubectl exec deployment/vllm-llama-70b -n "$MY_NAMESPACE" -- \
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /' || echo "    (nvidia-smi unavailable)"

echo ""
echo "============================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "============================================================"
[[ "$FAIL" -eq 0 ]]
