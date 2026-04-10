# Deploying a 70B LLM on Multi-GPU Kubernetes with vLLM + ClearML

A step-by-step production deployment guide for multi-tenant Kubernetes clusters
with ClearML Enterprise, corporate proxy for internet access, and enterprise
NFS storage.

---

## Cluster profile this guide is written for

| Property | Your cluster |
|----------|-------------|
| GPU type | H200 (141 GB HBM3e) or H100 (80 GB HBM3) |
| Cluster access | ClusterAdmin on Kubernetes |
| ClearML access | Tenant admin on ClearML Enterprise (multi-tenant) |
| MIG status | Enabled on some nodes — must be avoided for full-GPU workloads |
| Internet egress | Via **corporate HTTP/HTTPS proxy** (port 3128) — not direct |
| Image pulling | Public registries (Docker Hub, NGC) via managed pull secrets |
| Storage backend | Enterprise NFS (Powerscale/Isilon) with existing shared PVC (~1 TiB) |
| ClearML queue | Not yet created — you will create it |

---

## Step 0 — Discover your cluster

Run every command below and fill in the values. **Do not skip this step.**

### 0.1 — Confirm Kubernetes access

```bash
kubectl cluster-info
kubectl auth can-i '*' '*' --all-namespaces
# Must show: "yes"
# If "no": STOP. Get ClusterAdmin before proceeding.
```

### 0.2 — Discover GPU nodes

```bash
kubectl get nodes -o custom-columns=\
"NAME:.metadata.name,\
GPUs:.status.allocatable.nvidia\.com/gpu,\
MEMORY:.status.allocatable.memory"
```

```
MY_GPU_NODES=               # list node names with GPUs
MY_GPUS_PER_NODE=           # e.g. "8"
```

### 0.3 — Confirm GPU type and memory

```bash
# Pick any GPU node from the list above
kubectl debug node/<GPU_NODE_NAME> -it --image=nvidia/cuda:12.4.0-base-ubuntu22.04 -- nvidia-smi
```

Look for the GPU name and memory:
- **"NVIDIA H200" + ~141 GB** → Use `TENSOR_PARALLEL_SIZE=2` minimum.
- **"NVIDIA H100" + ~80 GB** → Use `TENSOR_PARALLEL_SIZE=4` minimum.

```
MY_GPU_TYPE=                # "H200" or "H100"
MY_GPU_MEMORY_GB=           # 141 or 80
MY_TENSOR_PARALLEL_SIZE=    # 2 (H200) or 4 (H100)
```

### 0.4 — Identify MIG nodes (must avoid)

```bash
# Check which nodes have MIG enabled
for node in $(kubectl get nodes -o name | sed 's|node/||'); do
  echo "--- $node ---"
  kubectl debug node/$node -it --image=nvidia/cuda:12.4.0-base-ubuntu22.04 -- nvidia-smi mig -lgi 2>/dev/null || echo "No MIG"
done
```

A 70B model in bfloat16 needs ~140 GB in GPU memory — no MIG profile is large enough.
You need full GPUs.

**Note:** A node showing `nvidia.com/gpu: 0` but with MIG enabled is NOT broken.
MIG replaces `nvidia.com/gpu` with `nvidia.com/mig-*` resources. Run
`kubectl describe node <NODE> | grep nvidia.com` to see the MIG slice resources.

```
MY_MIG_NODES=               # node names WITH MIG (avoid these)
MY_FULL_GPU_NODES=          # node names WITHOUT MIG (target these)
```

### 0.5 — Confirm GPU availability on non-MIG nodes

```bash
for node in <MY_FULL_GPU_NODES>; do
  echo "--- $node ---"
  kubectl describe node $node | grep -A5 "Allocated resources" | grep nvidia
done
```

```
MY_FREE_GPUS=               # must be >= MY_TENSOR_PARALLEL_SIZE
```

### 0.6 — Check node labels and taints

```bash
kubectl describe nodes | grep -A3 "Taints:"
kubectl get nodes --show-labels | grep -i "gpu\|nvidia\|accelerator"
```

```
MY_GPU_TAINT_KEY=           # e.g. "nvidia.com/gpu" or "none"
MY_GPU_NODE_LABEL=          # existing label that distinguishes full-GPU from MIG nodes
```

If no distinguishing label exists, create one:

```bash
kubectl label node <FULL_GPU_NODE_NAME> gpu-mode=full
# Repeat for each non-MIG GPU node
```

### 0.7 — Discover the corporate proxy

```bash
# Find proxy settings from an existing ClearML agent pod
kubectl get pods --all-namespaces | grep -i "clearml\|agent"
kubectl describe pod <CLEARML_AGENT_POD> -n <NAMESPACE> | grep -i proxy
```

You should see `http_proxy`, `https_proxy`, and `no_proxy` values.

```
MY_HTTP_PROXY=              # e.g. "http://10.x.x.x:3128/"
MY_HTTPS_PROXY=             # e.g. "http://10.x.x.x:3128/"
MY_NO_PROXY=                # e.g. ".clearml.internal.com,127.0.0.1,localhost"
```

Verify the proxy works:

```bash
kubectl run test-proxy --rm -it --restart=Never --image=python:3.11-slim \
  --env="http_proxy=$MY_HTTP_PROXY" \
  --env="https_proxy=$MY_HTTPS_PROXY" \
  --env="no_proxy=$MY_NO_PROXY" \
  -- bash -c "pip install -q requests && python3 -c \"import requests; r=requests.get('https://huggingface.co'); print('PROXY WORKS - status:', r.status_code)\""
# Expected: "PROXY WORKS - status: 200"
# If this fails: proxy IP/port may be wrong. Re-check from the agent pod.
```

### 0.8 — Discover the existing shared PVC

The cluster likely has a shared workspace PVC already mounted by ClearML.

```bash
# Find existing PVCs
kubectl get pvc --all-namespaces | grep -i "shared\|workspace\|model"

# Get details of the PVC
kubectl describe pvc <PVC_NAME> -n <NAMESPACE> | grep -E "Capacity|StorageClass|VolumeName"

# Find the mount path used by existing pods
kubectl get pods -n <NAMESPACE> -o jsonpath='{range .items[*]}{.metadata.name}: {range .spec.containers[*].volumeMounts[*]}{.mountPath} {end}{"\n"}{end}'
```

```
MY_SHARED_PVC_NAME=         # e.g. "shared-workspace"
MY_SHARED_PVC_NAMESPACE=    # namespace where it lives
MY_SHARED_MOUNT_PATH=       # e.g. "/shared-workspace"
MY_STORAGE_CLASS=           # from the PVC's StorageClass field
```

### 0.9 — Discover image pull secrets

```bash
kubectl get secrets --all-namespaces | grep -i "pull\|registry\|docker\|image"
```

```
MY_PULL_SECRET_NAME=        # e.g. "dockerhub-credentials"
MY_PULL_SECRET_NAMESPACE=   # where the pull secret lives
```

### 0.10 — Confirm ClearML tenant access

```bash
clearml-init
# Enter your API server, web server, file server URLs and credentials

python3 -c "
from clearml import Task
t = Task.init(project_name='_connection_test', task_name='verify_access')
print('Tenant connected. Project ID:', t.project)
t.close()
print('SUCCESS')
"
```

```
MY_CLEARML_API=             # API server URL
MY_CLEARML_WEB=             # Web UI URL
MY_CLEARML_FILES=           # File server URL
```

### 0.11 — Summary checklist

```
[ ] MY_GPU_TYPE               = ___________
[ ] MY_GPU_MEMORY_GB          = ___________
[ ] MY_TENSOR_PARALLEL_SIZE   = ___________
[ ] MY_MIG_NODES              = ___________ (will AVOID)
[ ] MY_FULL_GPU_NODES         = ___________ (will TARGET)
[ ] MY_FREE_GPUS              = ___________ (>= TENSOR_PARALLEL_SIZE)
[ ] MY_GPU_TAINT_KEY          = ___________
[ ] MY_GPU_NODE_LABEL         = ___________
[ ] MY_HTTP_PROXY             = ___________
[ ] MY_HTTPS_PROXY            = ___________
[ ] MY_NO_PROXY               = ___________
[ ] MY_SHARED_PVC_NAME        = ___________
[ ] MY_SHARED_PVC_NAMESPACE   = ___________
[ ] MY_SHARED_MOUNT_PATH      = ___________
[ ] MY_PULL_SECRET_NAME       = ___________
[ ] MY_CLEARML_API            = ___________
```

**If any item is blank, STOP and fill it in.**

---

## Step 1 — Prepare the deployment namespace

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Check if your tenant already has a workload namespace with a ClearML agent
kubectl get pods --all-namespaces | grep -i "clearml.*agent\|k8sglue"

# If a namespace already has an agent for your tenant, use it.
# If not, create one.
```

### Execute (only if creating a new namespace)

```bash
MY_NAMESPACE="llm-serving"

kubectl get namespace $MY_NAMESPACE 2>/dev/null && echo "ALREADY EXISTS" || echo "OK to create"
kubectl create namespace $MY_NAMESPACE

# Copy pull secret to your namespace
kubectl get secret $MY_PULL_SECRET_NAME -n $MY_PULL_SECRET_NAMESPACE -o yaml \
  | grep -v "namespace:\|uid:\|resourceVersion:\|creationTimestamp:" \
  | kubectl apply -n $MY_NAMESPACE -f -

# If the shared PVC is in a different namespace, you may need to create
# a new PVC in YOUR namespace. Check if the PVC is accessible:
kubectl get pvc $MY_SHARED_PVC_NAME -n $MY_NAMESPACE 2>/dev/null \
  && echo "PVC accessible" \
  || echo "PVC not in this namespace — see Step 2"
```

### 🔍 VERIFY AFTER

```bash
kubectl get namespace $MY_NAMESPACE        # STATUS = Active
kubectl get secret $MY_PULL_SECRET_NAME -n $MY_NAMESPACE  # Must exist
```

```
MY_NAMESPACE=               # your working namespace
```

---

## Step 2 — Ensure model storage is available

You have two options depending on whether the existing shared PVC is in your namespace.

### Option A — Shared PVC is in your namespace (or you deploy in its namespace)

Skip PVC creation. Use the existing PVC directly. Note the mount path:

```
MY_PVC_NAME=$MY_SHARED_PVC_NAME
MY_MODEL_DIR="$MY_SHARED_MOUNT_PATH/llama-3.3-70b"
```

### Option B — You need a new PVC in your namespace

### 🔍 VERIFY BEFORE PROCEEDING

```bash
kubectl get storageclass $MY_STORAGE_CLASS
kubectl get pvc -n $MY_NAMESPACE | grep -i model
# If a model PVC already exists and is Bound, skip this step
```

### Execute

```yaml
# model-storage-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llm-model-storage
  namespace: <MY_NAMESPACE>            # <-- REPLACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
  storageClassName: <MY_STORAGE_CLASS>  # <-- REPLACE
```

```bash
grep "<" model-storage-pvc.yaml && echo "ERROR: Replace placeholders!" || echo "OK"
kubectl apply -f model-storage-pvc.yaml
```

### 🔍 VERIFY AFTER

```bash
kubectl get pvc -n $MY_NAMESPACE
# STATUS must be "Bound"
```

```
MY_PVC_NAME=                # "shared-workspace" (Option A) or "llm-model-storage" (Option B)
```

---

## Step 3 — Download model weights via proxy

The cluster has internet access through a corporate proxy. You'll run a download
job with the proxy env vars set. The weights download directly into the PVC.

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm PVC is Bound
kubectl get pvc $MY_PVC_NAME -n $MY_NAMESPACE -o jsonpath='{.status.phase}'
# Must show: "Bound"

# Confirm proxy works (from Step 0.7)
# If you haven't tested yet, run the proxy test now.

# If using a gated model (Llama), confirm your HF token has access:
# On your LOCAL machine (which has internet):
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('meta-llama/Llama-3.3-70B-Instruct', token='YOUR_TOKEN')
print(f'Model: {info.id}, Size: {info.siblings[0].size if info.siblings else \"unknown\"}')
print('TOKEN HAS ACCESS')
"
# If 401/403: go to https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
# and accept the Meta license first.
#
# If you don't have/want a HF account: use Qwen/Qwen2.5-72B-Instruct instead
# (ungated, no license needed, comparable quality)
```

### Execute

Create a Kubernetes Secret for your HF token (skip if using ungated model):

```bash
kubectl create secret generic hf-credentials \
  --from-literal=token=<YOUR_HF_TOKEN> \
  -n $MY_NAMESPACE
```

Create `download-job.yaml` — **replace ALL `<>` placeholders:**

```yaml
# download-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download
  namespace: <MY_NAMESPACE>                    # <-- REPLACE
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>          # <-- REPLACE
      containers:
        - name: downloader
          image: python:3.11-slim
          env:
            - name: http_proxy
              value: "<MY_HTTP_PROXY>"          # <-- REPLACE
            - name: https_proxy
              value: "<MY_HTTPS_PROXY>"         # <-- REPLACE
            - name: no_proxy
              value: "<MY_NO_PROXY>"            # <-- REPLACE
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
                  optional: true               # optional if using ungated model
          command:
            - /bin/sh
            - -c
            - |
              pip install -q huggingface_hub && \
              python3 -c "
              import os
              from huggingface_hub import snapshot_download

              # === CONFIGURE MODEL HERE ===
              # For Llama 3.3 70B (gated — needs HF token):
              REPO = 'meta-llama/Llama-3.3-70B-Instruct'
              TOKEN = os.environ.get('HF_TOKEN')

              # For Qwen 2.5 72B (ungated — no token needed):
              # REPO = 'Qwen/Qwen2.5-72B-Instruct'
              # TOKEN = None
              # =============================

              LOCAL_DIR = '<MY_SHARED_MOUNT_PATH>/llama-3.3-70b'   # <-- REPLACE

              print(f'Downloading {REPO} to {LOCAL_DIR}')
              print('This will download ~130 GB. Be patient.')
              snapshot_download(
                  repo_id=REPO,
                  local_dir=LOCAL_DIR,
                  token=TOKEN,
                  ignore_patterns=['*.gguf', '*.bin'],
              )
              print('Download complete')
              "
          volumeMounts:
            - name: model-storage
              mountPath: <MY_SHARED_MOUNT_PATH>  # <-- REPLACE
          resources:
            requests:
              memory: "8Gi"
              cpu: "2"
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: <MY_PVC_NAME>             # <-- REPLACE
```

```bash
# Verify no placeholders
grep "<" download-job.yaml && echo "ERROR: Replace all <> placeholders!" || echo "OK"

kubectl apply -f download-job.yaml

# Watch progress (30-60 minutes depending on proxy bandwidth)
kubectl logs -f job/model-download -n $MY_NAMESPACE

# If download fails mid-way (network timeout, proxy disconnect):
# kubectl delete job model-download -n $MY_NAMESPACE
# kubectl apply -f download-job.yaml
# huggingface_hub resumes automatically — it won't re-download completed files
```

### 🔍 VERIFY AFTER

```bash
# Confirm job completed
kubectl get job model-download -n $MY_NAMESPACE -o jsonpath='{.status.succeeded}'
# Must show: "1"

# Verify model files exist and are complete
kubectl run verify-weights --rm -it --restart=Never \
  --image=busybox \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "verify",
        "image": "busybox",
        "command": ["sh", "-c", "echo '=== Files ===' && ls /storage/llama-3.3-70b/*.safetensors 2>/dev/null | wc -l && echo 'safetensors files' && echo '=== Size ===' && du -sh /storage/llama-3.3-70b/"],
        "volumeMounts": [{"name":"ms","mountPath":"/storage"}]
      }],
      "volumes": [{"name":"ms","persistentVolumeClaim":{"claimName":"<MY_PVC_NAME>"}}]
    }
  }' \
  -n $MY_NAMESPACE
# Expected: ~30 safetensors files, total ~130 GB
# If fewer files or much less than 130 GB: download is incomplete, re-run the job
```

```
MY_MODEL_PATH=              # full path inside PVC, e.g. "/shared-workspace/llama-3.3-70b"
MY_MODEL_NAME=              # name for vLLM, e.g. "llama-3.3-70b-instruct"
```

---

## Step 4 — Register the model in ClearML

### 🔍 VERIFY BEFORE PROCEEDING

```bash
python3 -c "from clearml import Task; print('ClearML SDK OK')"
```

### Execute

```python
# register_model.py
from clearml import Task, OutputModel

# ============================================================
# CONFIGURE THESE
# ============================================================
PROJECT_NAME = "production-llm"
MODEL_NAME   = "llama-3.3-70b-instruct"       # <-- match MY_MODEL_NAME
CLUSTER_PATH = "/shared-workspace/llama-3.3-70b"  # <-- match MY_MODEL_PATH
GPU_TYPE     = "H200 (141 GB HBM3e)"          # <-- or "H100 (80 GB HBM3)"
MIN_GPUS     = 2                               # <-- match MY_TENSOR_PARALLEL_SIZE
# ============================================================

task = Task.init(
    project_name=PROJECT_NAME,
    task_name=f"register-{MODEL_NAME}",
    task_type=Task.TaskTypes.data_processing,
    reuse_last_task_id=False,
)

out_model = OutputModel(
    task=task,
    name=MODEL_NAME,
    tags=["70b", "instruct", "vllm", "production"],
    framework="PyTorch",
)

out_model.update_design(config_dict={
    "cluster_path": CLUSTER_PATH,
    "format": "safetensors",
    "precision": "bfloat16",
    "inference_engine": "vLLM",
    "gpu_type": GPU_TYPE,
    "min_gpus": MIN_GPUS,
})

task.close()
print(f"Model registered: {out_model.id}")
```

```bash
python3 register_model.py
```

### 🔍 VERIFY AFTER

Open ClearML Web UI → Models → search for your model name. Confirm it appears.

---

## Step 5 — Deploy vLLM inference server

vLLM runs as a container image. Kubernetes pulls it via managed pull secrets.
You do NOT install vLLM on cluster nodes.

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# 1. Enough free GPUs on non-MIG nodes?
for node in <MY_FULL_GPU_NODES>; do
  echo "--- $node ---"
  kubectl describe node $node | grep -A5 "Allocated resources" | grep nvidia
done

# 2. Model weights PVC is Bound?
kubectl get pvc $MY_PVC_NAME -n $MY_NAMESPACE -o jsonpath='{.status.phase}'

# 3. Pull secret exists in your namespace?
kubectl get secret $MY_PULL_SECRET_NAME -n $MY_NAMESPACE
```

### Execute

Create `vllm-deployment.yaml` — **replace ALL `<>` placeholders:**

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: <MY_NAMESPACE>
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-inference
  template:
    metadata:
      labels:
        app: vllm-inference
    spec:
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>
      nodeSelector:
        gpu-mode: full                           # <-- match YOUR non-MIG node label
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          command:
            - python3
            - -m
            - vllm.entrypoints.openai.api_server
          args:
            - --model=<MY_MODEL_PATH>            # <-- REPLACE (e.g. /shared-workspace/llama-3.3-70b)
            - --served-model-name=<MY_MODEL_NAME>
            - --tensor-parallel-size=<MY_TENSOR_PARALLEL_SIZE>
            - --dtype=bfloat16
            - --max-model-len=32768
            - --gpu-memory-utilization=0.90
            - --port=8000
            - --host=0.0.0.0
            - --trust-remote-code
            - --enable-chunked-prefill
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "<MY_TENSOR_PARALLEL_SIZE>"
            requests:
              nvidia.com/gpu: "<MY_TENSOR_PARALLEL_SIZE>"
              memory: "64Gi"
              cpu: "8"
          volumeMounts:
            - name: model-storage
              mountPath: <MY_SHARED_MOUNT_PATH>  # <-- REPLACE (e.g. /shared-workspace)
            - name: shm
              mountPath: /dev/shm
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 30
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 300
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 5
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: <MY_PVC_NAME>
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "16Gi"
      tolerations:                               # <-- remove this block if no GPU taints
        - key: <MY_GPU_TAINT_KEY>
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-inference
  namespace: <MY_NAMESPACE>
spec:
  selector:
    app: vllm-inference
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

**Why these settings matter:**
- `nodeSelector: gpu-mode: full` → avoids MIG nodes
- `/dev/shm` → required for NCCL multi-GPU communication
- `--gpu-memory-utilization=0.90` → uses 90% of GPU VRAM for weights + KV cache
- `readinessProbe` at 180s → vLLM needs 3-5 min to load ~130 GB of weights
- `livenessProbe` at 300s → prevents K8s from killing the pod during loading

```bash
grep "<" vllm-deployment.yaml && echo "ERROR: Replace placeholders!" || echo "OK"

kubectl apply -f vllm-deployment.yaml
kubectl get pods -n $MY_NAMESPACE -l app=vllm-inference -w
kubectl logs -f deployment/vllm-inference -n $MY_NAMESPACE
# Wait for: "INFO:     Application startup complete."
```

### 🔍 VERIFY AFTER

```bash
# Pod is Running and Ready
kubectl get pods -n $MY_NAMESPACE -l app=vllm-inference

# Landed on a non-MIG node
kubectl get pod -l app=vllm-inference -n $MY_NAMESPACE -o wide

# Health check
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- \
  curl -s http://localhost:8000/health

# GPU usage
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- nvidia-smi
# Expected: 2+ GPUs, ~90% memory used

# Test inference
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- \
  curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<MY_MODEL_NAME>","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Service DNS works
kubectl run test-svc --rm -it --restart=Never --image=busybox -n $MY_NAMESPACE -- \
  wget -qO- http://vllm-inference:8000/health --timeout=10
```

**If pod is Pending:** `kubectl describe pod -l app=vllm-inference -n $MY_NAMESPACE | grep -A10 Events`
**If CrashLoopBackOff:** `kubectl logs -l app=vllm-inference -n $MY_NAMESPACE --tail=50`

---

## Step 6 — Deploy LiteLLM proxy

### 🔍 VERIFY BEFORE PROCEEDING

```bash
kubectl run test-vllm --rm -it --restart=Never --image=busybox -n $MY_NAMESPACE -- \
  wget -qO- http://vllm-inference:8000/health --timeout=10
# Must show: {"status":"ok"}
```

### Execute

Create `litellm-config.yaml`:

```yaml
model_list:
  - model_name: <MY_MODEL_NAME>
    litellm_params:
      model: openai/<MY_MODEL_NAME>
      api_base: http://vllm-inference.<MY_NAMESPACE>.svc.cluster.local:8000/v1
      api_key: "not-used"

litellm_settings:
  drop_params: true
  request_timeout: 300

general_settings:
  master_key: "<GENERATE_A_RANDOM_KEY>"
```

```bash
kubectl create configmap litellm-config \
  --from-file=config.yaml=litellm-config.yaml \
  -n $MY_NAMESPACE
```

Create `litellm-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-proxy
  namespace: <MY_NAMESPACE>
spec:
  replicas: 1
  selector:
    matchLabels:
      app: litellm-proxy
  template:
    metadata:
      labels:
        app: litellm-proxy
    spec:
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>
      containers:
        - name: litellm
          image: ghcr.io/berriai/litellm:main-stable
          args:
            - --config=/app/config.yaml
            - --port=4000
          ports:
            - containerPort: 4000
          resources:
            requests:
              memory: "512Mi"
              cpu: "1"
          volumeMounts:
            - name: config
              mountPath: /app/config.yaml
              subPath: config.yaml
          readinessProbe:
            httpGet:
              path: /health
              port: 4000
            initialDelaySeconds: 10
            periodSeconds: 5
      volumes:
        - name: config
          configMap:
            name: litellm-config
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-proxy
  namespace: <MY_NAMESPACE>
spec:
  selector:
    app: litellm-proxy
  ports:
    - port: 4000
      targetPort: 4000
  type: NodePort
```

```bash
grep "<" litellm-deployment.yaml && echo "ERROR: Replace placeholders!" || echo "OK"
kubectl apply -f litellm-deployment.yaml
```

### 🔍 VERIFY AFTER

```bash
# Get access URL
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc litellm-proxy -n $MY_NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
echo "LiteLLM URL: http://$NODE_IP:$NODE_PORT"

# End-to-end test
curl -s -X POST http://$NODE_IP:$NODE_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_MASTER_KEY>" \
  -d "{
    \"model\": \"<MY_MODEL_NAME>\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What model are you?\"}],
    \"max_tokens\": 100
  }"
# If you get a coherent response: THE FULL STACK IS WORKING.
```

---

## Step 7 — Log an inference run to ClearML

```python
# inference_task.py
import os, json, urllib.request
from clearml import Task, InputModel

LITELLM_URL  = "http://<NODE_IP>:<NODE_PORT>"    # <-- REPLACE
API_KEY      = "<YOUR_MASTER_KEY>"                # <-- REPLACE
MODEL_NAME   = "<MY_MODEL_NAME>"                  # <-- REPLACE
PROJECT_NAME = "production-llm"

task = Task.init(
    project_name=PROJECT_NAME,
    task_name=f"{MODEL_NAME}-inference-demo",
    task_type=Task.TaskTypes.inference,
    reuse_last_task_id=False,
)
logger = task.get_logger()

model = InputModel(name=MODEL_NAME, project=PROJECT_NAME)
logger.report_text(f"Model: {model.name}  ID: {model.id}")

prompts = [
    "Explain transformer attention in one paragraph.",
    "What is tensor parallelism?",
    "How does continuous batching work in LLM inference?",
]

results = []
for i, prompt in enumerate(prompts, 1):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200, "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"{LITELLM_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    logger.report_text(f"Query {i}: {prompt}\nResponse: {text}")
    logger.report_scalar("Token usage", "prompt_tokens", usage.get("prompt_tokens", 0), i)
    logger.report_scalar("Token usage", "completion_tokens", usage.get("completion_tokens", 0), i)
    results.append({"prompt": prompt, "response": text, "tokens": usage})
    print(f"Query {i} done. Tokens: {usage}")

task.upload_artifact("inference_results", artifact_object=results)
task.close()
print(f"Done. Check ClearML UI -> {PROJECT_NAME}")
```

---

## Step 8 — Monitor GPU health

```bash
# GPU status
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- nvidia-smi

# DCGM diagnostics (if available)
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- dcgmi diag -r 1

# Real-time metrics
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- \
  dcgmi dmon -e 100,101,102,203,204 -d 2000
# 100=SM%  101=mem%  102=mem(MB)  203=temp(C)  204=power(W)

# NCCL topology
kubectl logs deployment/vllm-inference -n $MY_NAMESPACE | grep -i "nccl\|nvlink"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Pod `Pending` | Not enough GPUs or label mismatch | `kubectl describe pod` → Events |
| Pod `CrashLoopBackOff` | OOM or bad args | Check logs. Reduce `--max-model-len` to `16384` |
| vLLM killed after 2 min | Liveness probe too aggressive | Increase `initialDelaySeconds` to `600` |
| NCCL shared memory error | Missing `/dev/shm` mount | Add `shm` emptyDir volume |
| Pod on MIG node | nodeSelector wrong | Fix label in `nodeSelector` |
| LiteLLM 502 | vLLM not ready | Wait 5 min, check vLLM logs |
| Download job fails | Proxy env vars missing or wrong | Check `http_proxy`/`https_proxy` in job spec |
| `pip install` fails in job | Proxy not set | Add proxy env vars to the download job |
| `ImagePullBackOff` | Pull secret missing | Copy secret to namespace (Step 1) |
| MIG node shows 0 GPUs | Normal MIG behavior | Not a failure — MIG uses `nvidia.com/mig-*` resources |
| nvidia-smi 0% util | No requests | Normal when idle |

---

## Cleanup

```bash
kubectl delete deployment litellm-proxy -n $MY_NAMESPACE
kubectl delete svc litellm-proxy -n $MY_NAMESPACE
kubectl delete configmap litellm-config -n $MY_NAMESPACE
kubectl delete deployment vllm-inference -n $MY_NAMESPACE
kubectl delete svc vllm-inference -n $MY_NAMESPACE
kubectl delete job model-download -n $MY_NAMESPACE
kubectl delete secret hf-credentials -n $MY_NAMESPACE
# Only if you created a separate PVC (Option B):
# kubectl delete pvc llm-model-storage -n $MY_NAMESPACE
```

---

## Reference: local lab → production

| Topic | Local lab | Production |
|-------|-----------|------------|
| Model storage | Local filesystem | K8s PVC on NFS (Powerscale) |
| Inference engine | Custom serve.py | vLLM (PagedAttention, continuous batching) |
| GPU | 1× consumer GPU | 2-8× H200 (141 GB HBM3e each) |
| Context window | 128 chars | 32K–131K tokens |
| Concurrency | 1 request | Hundreds |
| Model download | Python script (direct) | K8s Job via corporate proxy |
| Secrets | Hardcoded | K8s Secrets |
| Networking | localhost | K8s Service DNS |
| Multi-GPU | N/A | NCCL over NVLink (/dev/shm) |
| GPU targeting | N/A | nodeSelector to avoid MIG |
| Monitoring | nvidia-smi | DCGM + Prometheus |

The API is identical: `POST /v1/chat/completions` with `Authorization: Bearer <key>`.

---

## Next steps

1. **Ingress** — DNS name + TLS in front of LiteLLM
2. **Virtual keys** — per-user rate limiting in LiteLLM
3. **vLLM metrics** — `--enable-metrics`, scrape with Prometheus
4. **ClearML Pipeline** — auto-redeploy on new model registration
5. **Scale up** — higher `--tensor-parallel-size` for 131K context
6. **Dev tools** — point VS Code / Continue.dev at the LiteLLM endpoint
7. **ClearML queue** — integrate with tenant queue for managed lifecycle
