# Step 0 — Discover Your Cluster

Run every command below before touching any YAML or script. Record the values
in `config.env` (copy from `../config.env.example`).

---

## 0.1 — Confirm cluster access and permissions

```bash
kubectl cluster-info
kubectl get nodes -o wide
kubectl auth can-i '*' '*' --all-namespaces
# Expected: "yes" — if "no": STOP, you need ClusterAdmin first
```

---

## 0.2 — Discover GPU nodes and counts

```bash
kubectl get nodes -o custom-columns=\
"NAME:.metadata.name,\
GPUs:.status.allocatable.nvidia\.com/gpu,\
MEMORY:.status.allocatable.memory,\
OS:.status.nodeInfo.osImage"
```

Record in `config.env`:
```
MY_GPU_NODE_NAMES=
MY_GPU_COUNT_PER_NODE=
MY_TOTAL_GPUS_AVAILABLE=
```

---

## 0.3 — Confirm GPU model (H200 vs H100)

```bash
kubectl debug node/<YOUR_GPU_NODE_NAME> -it \
  --image=nvidia/cuda:12.4.0-base-ubuntu22.04 -- nvidia-smi
```

Look for:
- **H200**: "NVIDIA H200" and ~141 GB / ~143360 MiB  ✅
- **H100**: "NVIDIA H100" and ~80 GB / ~81920 MiB — still works but tensor-parallel-size must be 4

| GPU  | VRAM   | Min GPUs for 70B bf16 | Recommended |
|------|--------|-----------------------|-------------|
| H200 | 141 GB | 2                     | 4           |
| H100 | 80 GB  | 2 (tight)             | 4           |

Record in `config.env`:
```
MY_GPU_TYPE=
MY_GPU_MEMORY_GB=
MY_TENSOR_PARALLEL_SIZE=    # 2 for H200, 4 for H100
```

---

## 0.4 — Find ClearML services

```bash
kubectl get pods --all-namespaces | grep -i clearml
kubectl get svc -n <CLEARML_NAMESPACE>
```

Then open the ClearML web UI → Settings → Workspace → Create new credentials.

Record in `config.env`:
```
MY_CLEARML_NAMESPACE=
MY_CLEARML_API_URL=
MY_CLEARML_WEB_URL=
MY_CLEARML_FILES_URL=
MY_CLEARML_ACCESS_KEY=
MY_CLEARML_SECRET_KEY=
```

---

## 0.5 — Storage class

```bash
kubectl get storageclass
# Use the one marked (default), or ask the cluster owner
```

Record in `config.env`:
```
MY_STORAGE_CLASS=
```

---

## 0.6 — Check current GPU usage

```bash
kubectl get pods --all-namespaces -o wide | grep -i gpu
kubectl describe nodes | grep -A10 "Allocated resources" | grep -A3 "nvidia"
```

Record:
```
MY_FREE_GPUS=    # must be >= MY_TENSOR_PARALLEL_SIZE to proceed
```

---

## 0.7 — Test internet egress (model + image downloads)

```bash
kubectl run test-egress --rm -it --restart=Never \
  --image=busybox -- wget -qO- https://huggingface.co --timeout=10
# Hangs or fails → cluster is air-gapped. Complete 0.7a and 0.7b below.
```

Record:
```
MY_CLUSTER_HAS_INTERNET=    # "yes" or "no"
```

---

## 0.7a — (Air-gapped only) Discover the internal container registry

Skip this section if `MY_CLUSTER_HAS_INTERNET=yes`.

Most corporate clusters pull images from an internal registry, not Docker Hub directly.

```bash
# Check for existing pull secrets — they often name the registry
kubectl get secrets --all-namespaces | grep -i "registry\|pull\|docker"

# Check what images existing pods are using
kubectl get pods --all-namespaces \
  -o jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' \
  | sort -u | head -20
# Look for patterns like:
#   artifactory.yourcompany.com/docker/nginx:latest
#   registry.internal.corp/images/python:3.11
#   123456789.dkr.ecr.us-east-1.amazonaws.com/app:v1
# The domain before the first "/" is your internal registry
```

Record in `config.env`:
```
MY_CONTAINER_REGISTRY=      # e.g. "artifactory.corp.com/docker"
MY_IMAGE_PULL_SECRET=       # e.g. "registry-credentials" (or "none" if not required)
```

---

## 0.7b — (Air-gapped only) Pre-load the vLLM, LiteLLM, and Python images

Skip this section if `MY_CLUSTER_HAS_INTERNET=yes`.

**Do this on a machine WITH internet access** (laptop, bastion, CI runner), then push to your internal registry:

```bash
# Pull the three images needed by this stack
docker pull vllm/vllm-openai:latest           # ~8-10 GB — takes a while
docker pull ghcr.io/berriai/litellm:main-stable
docker pull python:3.11-slim

# Tag for your internal registry
export MY_CONTAINER_REGISTRY="artifactory.corp.com/docker"   # replace this

docker tag vllm/vllm-openai:latest            $MY_CONTAINER_REGISTRY/vllm-openai:latest
docker tag ghcr.io/berriai/litellm:main-stable $MY_CONTAINER_REGISTRY/litellm:main-stable
docker tag python:3.11-slim                   $MY_CONTAINER_REGISTRY/python:3.11-slim

# Push to internal registry (authenticate first if needed)
docker push $MY_CONTAINER_REGISTRY/vllm-openai:latest
docker push $MY_CONTAINER_REGISTRY/litellm:main-stable
docker push $MY_CONTAINER_REGISTRY/python:3.11-slim
```

If you can't use Docker directly, ask the platform team how to onboard new images — there's always a process (they got ClearML in somehow).

Record the full image paths in `config.env`:
```
MY_VLLM_IMAGE=      # e.g. "artifactory.corp.com/docker/vllm-openai:latest"
MY_LITELLM_IMAGE=   # e.g. "artifactory.corp.com/docker/litellm:main-stable"
MY_PYTHON_IMAGE=    # e.g. "artifactory.corp.com/docker/python:3.11-slim"
```

---

## 0.8 — Set up your deployment namespace

```bash
# Option A: same namespace as ClearML
#   Pro: short DNS names. Con: mixes with ClearML pods.
MY_NAMESPACE=$MY_CLEARML_NAMESPACE

# Option B: dedicated namespace (recommended)
MY_NAMESPACE="llm-serving"
kubectl create namespace $MY_NAMESPACE

# If the cluster requires an image pull secret, copy it to the new namespace
# (pods can only use secrets in their own namespace)
kubectl get secret $MY_IMAGE_PULL_SECRET -n $MY_CLEARML_NAMESPACE -o yaml \
  | sed "s/namespace: .*/namespace: $MY_NAMESPACE/" \
  | kubectl apply -f -

# Check for ResourceQuotas that might block GPU requests
kubectl get resourcequota -n $MY_NAMESPACE
kubectl get limitrange -n $MY_NAMESPACE
kubectl describe resourcequota -n $MY_NAMESPACE | grep -i gpu
# If "nvidia.com/gpu" has a limit, it must be >= MY_TENSOR_PARALLEL_SIZE
# If it's too low: ask the cluster admin to raise it for your namespace
```

Record:
```
MY_NAMESPACE=
```

---

## 0.9 — GPU node taint

```bash
kubectl describe nodes | grep -i taint
```

Record:
```
MY_GPU_TAINT_KEY=    # e.g. "nvidia.com/gpu" or empty string if no taint
```

---

## 0.10 — Pre-flight checklist

Before moving to Step 1, confirm every item:

```
[ ] MY_GPU_TYPE             = ___________
[ ] MY_GPU_MEMORY_GB        = ___________
[ ] MY_TENSOR_PARALLEL_SIZE = ___________
[ ] MY_GPU_TAINT_KEY        = ___________
[ ] MY_CLEARML_NAMESPACE    = ___________
[ ] MY_CLEARML_API_URL      = ___________
[ ] MY_NAMESPACE            = ___________
[ ] MY_STORAGE_CLASS        = ___________
[ ] MY_FREE_GPUS            = ___________ (>= MY_TENSOR_PARALLEL_SIZE)
[ ] MY_CLUSTER_HAS_INTERNET = ___________
[ ] MY_CONTAINER_REGISTRY   = ___________ (or "docker.io" if public internet)
[ ] MY_IMAGE_PULL_SECRET    = ___________ (or "none")
[ ] MY_VLLM_IMAGE           = ___________
[ ] MY_LITELLM_IMAGE        = ___________
[ ] MY_PYTHON_IMAGE         = ___________
[ ] HF account with Meta Llama license accepted
[ ] HF_TOKEN                = hf_...
[ ] LITELLM_MASTER_KEY      = sk-...
```

**If any item is blank → STOP and fill it in.**
