#!/usr/bin/env bash
# cleanup.sh — Remove all resources created by deploy-all.sh
#
# WARNING: Deleting the PVC erases the 140 GB of downloaded model weights.
#          You will need to re-run the download job to restore them.
#
# Usage:
#   source ../config.env && bash cleanup.sh
set -euo pipefail

: "${MY_NAMESPACE:?Source config.env first}"

echo "============================================================"
echo " Cleanup: removing all Llama 3.3 70B resources"
echo " Namespace: $MY_NAMESPACE"
echo "============================================================"
read -r -p "Are you sure? This will delete the 140 GB PVC. [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

echo "Removing LiteLLM proxy..."
kubectl delete deployment litellm-proxy -n "$MY_NAMESPACE" --ignore-not-found
kubectl delete svc         litellm-proxy -n "$MY_NAMESPACE" --ignore-not-found
kubectl delete configmap   litellm-config -n "$MY_NAMESPACE" --ignore-not-found

echo "Removing vLLM server..."
kubectl delete deployment vllm-llama-70b -n "$MY_NAMESPACE" --ignore-not-found
kubectl delete svc         vllm-llama-70b -n "$MY_NAMESPACE" --ignore-not-found

echo "Removing download job..."
kubectl delete job llama-download -n "$MY_NAMESPACE" --ignore-not-found

echo "Removing PVC (deletes 140 GB of weights)..."
kubectl delete pvc llama-model-storage -n "$MY_NAMESPACE" --ignore-not-found

echo "Removing HF secret..."
kubectl delete secret hf-credentials -n "$MY_NAMESPACE" --ignore-not-found

echo ""
echo "Cleanup complete."
echo "The namespace '$MY_NAMESPACE' itself was NOT deleted."
echo "To also delete it: kubectl delete namespace $MY_NAMESPACE"
