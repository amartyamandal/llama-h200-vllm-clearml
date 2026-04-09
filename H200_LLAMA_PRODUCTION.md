# Deploying a 70B LLM on Multi-GPU Kubernetes with vLLM + ClearML

A step-by-step production deployment guide for air-gapped, multi-tenant Kubernetes
clusters with GPU scheduling managed by ClearML Enterprise.

---

## Cluster profile this guide is written for

| Property | Your cluster |
|----------|-------------|
| GPU type | H200 (141 GB HBM3e) or H100 (80 GB HBM3) |
| Cluster access | ClusterAdmin on Kubernetes |
| ClearML access | Tenant admin on ClearML Enterprise (multi-tenant) |
| MIG status | Enabled on some nodes — must be avoided for full-GPU workloads |
| Internet egress | **None** — pods cannot reach external sites |
| Image pulling | Public registries (Docker Hub, NGC) via managed pull secrets |
| Storage backend | Enterprise NFS (e.g. Powerscale/Isilon) |
| ClearML queue | Not yet created — you will create it |

If your cluster differs, adjust the marked parameters accordingly.

---

## Before you begin: what you need from the platform team

Even with ClusterAdmin and ClearML tenant admin, some things require
platform team involvement. Ask these first and **do not proceed until
you have answers:**

### Must ask (you cannot solve these yourself)

**1. Model weights ingestion path**

> "Pods in this cluster cannot reach the internet. I need to stage ~140 GB
> of LLM weights onto the NFS storage backend. What is the approved path
> for getting large external files into the cluster? Is there a bastion
> host, NFS mount point, S3 staging bucket, or a data transfer process?"

**2. Model source / licensing**

> "Does the organization have a Hugging Face account or token for
> downloading gated models (e.g. Llama 3.3 70B)? Has anyone already
> downloaded LLM weights that I can reuse from shared storage? If
> Hugging Face is not approved, should I use an ungated model instead?"

**3. Long-running workloads policy**

> "An inference server (vLLM) needs to run 24/7 as an always-on service,
> not a batch job that completes and exits. Should I deploy it through
> ClearML queues, or as a standalone Kubernetes Deployment?"

### Can verify yourself (do these while waiting for answers)

Everything in Step 0 below.

---

## Step 0 — Discover your cluster

Run every command below and fill in the values. These are referenced
throughout the guide. **Do not skip this step.**

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

Look for the GPU name and memory in the output:
- **"NVIDIA H200" + ~141 GB** → H200. Use `TENSOR_PARALLEL_SIZE=2` minimum.
- **"NVIDIA H100" + ~80 GB** → H100. Use `TENSOR_PARALLEL_SIZE=4` minimum.

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

Any node that shows MIG instances is a MIG node. Your workload needs **full GPUs**,
not MIG slices. A 70B model in bfloat16 needs ~140 GB — no MIG profile is large enough.

```
MY_MIG_NODES=               # node names WITH MIG (avoid these)
MY_FULL_GPU_NODES=          # node names WITHOUT MIG (target these)
```

### 0.5 — Confirm GPU availability on non-MIG nodes

```bash
# Check allocated vs allocatable GPUs on your target nodes
for node in MY_FULL_GPU_NODES; do    # replace with actual node names
  echo "--- $node ---"
  kubectl describe node $node | grep -A5 "Allocated resources" | grep nvidia
done
```

```
MY_FREE_GPUS=               # free GPUs on non-MIG nodes (must be >= TENSOR_PARALLEL_SIZE)
```

### 0.6 — Check node labels and taints

```bash
# Check taints on GPU nodes (needed for tolerations in your pod specs)
kubectl describe nodes | grep -A3 "Taints:"

# Check existing labels (ClearML agent uses labels to target nodes)
kubectl get nodes --show-labels | grep -i "gpu\|nvidia\|accelerator"
```

```
MY_GPU_TAINT_KEY=           # e.g. "nvidia.com/gpu" or "gpu=true" or "none"
MY_GPU_NODE_LABEL=          # e.g. "nvidia.com/gpu.product=NVIDIA-H200" or custom label
```

If your non-MIG full-GPU nodes don't have a label that distinguishes them from
MIG nodes, create one now:

```bash
# Label your full-GPU nodes (skip if a distinguishing label already exists)
kubectl label node <FULL_GPU_NODE_NAME> gpu-mode=full
# Repeat for each non-MIG GPU node
```

### 0.7 — Discover ClearML tenant namespaces

```bash
# Find namespaces with ClearML components
kubectl get pods --all-namespaces | grep -i "clearml\|agent\|k8sglue\|app-gateway"

# List all ClearML-related services
kubectl get svc --all-namespaces | grep -i clearml
```

```
MY_CLEARML_NAMESPACES=      # namespaces where ClearML pods run
MY_CLEARML_AGENT_NAMESPACE= # namespace where the agent for YOUR tenant runs (if any)
```

### 0.8 — Discover storage classes

```bash
kubectl get storageclass -o custom-columns="NAME:.metadata.name,PROVISIONER:.provisioner,DEFAULT:.metadata.annotations.storageclass\.kubernetes\.io/is-default-class"

# Identify which class provisions on your NFS backend (Powerscale/Isilon)
# Look for provisioner containing "powerscale", "isilon", "nfs", or "csi-isilon"
```

```
MY_STORAGE_CLASS=           # storage class that provisions on NFS
```

### 0.9 — Discover image pull secrets

```bash
# Find managed pull secrets
kubectl get secrets --all-namespaces | grep -i "pull\|registry\|docker\|image"

# Check what images existing pods use (to confirm public registries work)
kubectl get pods --all-namespaces -o jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' | sort -u | head -20
```

```
MY_PULL_SECRET_NAME=        # e.g. "dockerhub-credentials" or "regcred"
MY_PULL_SECRET_NAMESPACE=   # where the pull secret lives
```

### 0.10 — Confirm ClearML tenant access

```bash
# From your local machine — configure ClearML CLI
clearml-init
# Enter your API server, web server, file server URLs and credentials

# Verify connection
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
[ ] MY_MIG_NODES              = ___________ (will AVOID these)
[ ] MY_FULL_GPU_NODES         = ___________ (will TARGET these)
[ ] MY_FREE_GPUS              = ___________ (>= TENSOR_PARALLEL_SIZE)
[ ] MY_GPU_TAINT_KEY          = ___________
[ ] MY_GPU_NODE_LABEL         = ___________ (label distinguishing full-GPU from MIG nodes)
[ ] MY_STORAGE_CLASS          = ___________
[ ] MY_PULL_SECRET_NAME       = ___________
[ ] MY_CLEARML_API            = ___________
[ ] MY_CLEARML_WEB            = ___________
[ ] MY_CLEARML_FILES          = ___________
[ ] Platform team response: model ingestion path = ___________
[ ] Platform team response: HF token / model source = ___________
[ ] Platform team response: long-running workload policy = ___________
```

**If any critical item is blank, STOP.**

---

## Step 1 — Prepare the deployment namespace

You need a Kubernetes namespace where vLLM, LiteLLM, and model storage will live.
This may already exist (your ClearML tenant's workload namespace) or you may
need to create one.

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Check if your tenant already has a workload namespace
kubectl get namespaces | grep -i "clearml\|tenant\|llm\|mlops\|gpu"

# Check if any of those namespaces already have a ClearML agent
kubectl get pods -n <CANDIDATE_NAMESPACE> | grep -i agent

# If a namespace already has a ClearML agent configured for your tenant,
# that's your workload namespace. Use it.
#
# If no suitable namespace exists, create one.
```

### Execute (only if creating a new namespace)

```bash
MY_NAMESPACE="llm-serving"   # choose a name

# Check it doesn't already exist
kubectl get namespace $MY_NAMESPACE 2>/dev/null && echo "ALREADY EXISTS" || echo "OK to create"

# Create
kubectl create namespace $MY_NAMESPACE

# Copy the image pull secret into your namespace
# (pods can only use secrets from their own namespace)
kubectl get secret $MY_PULL_SECRET_NAME -n $MY_PULL_SECRET_NAMESPACE -o yaml \
  | grep -v "namespace:" \
  | grep -v "uid:" \
  | grep -v "resourceVersion:" \
  | grep -v "creationTimestamp:" \
  | kubectl apply -n $MY_NAMESPACE -f -

# Verify the pull secret exists in your namespace
kubectl get secret $MY_PULL_SECRET_NAME -n $MY_NAMESPACE
```

### 🔍 VERIFY AFTER

```bash
kubectl get namespace $MY_NAMESPACE
# STATUS must be "Active"

kubectl get secret $MY_PULL_SECRET_NAME -n $MY_NAMESPACE
# Must exist
```

```
MY_NAMESPACE=               # your final working namespace
```

---

## Step 2 — Create persistent storage for model weights

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm your storage class exists
kubectl get storageclass $MY_STORAGE_CLASS
# Must return a row (not "NotFound")

# Check if a PVC for model storage already exists
kubectl get pvc -n $MY_NAMESPACE | grep -i "model\|llm\|llama"
# If it exists and is Bound with enough capacity, skip this step
```

### Execute

Create `model-storage-pvc.yaml` — **replace placeholders:**

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
      storage: 300Gi
  storageClassName: <MY_STORAGE_CLASS>  # <-- REPLACE
```

```bash
# Verify no placeholders remain
grep "<" model-storage-pvc.yaml && echo "ERROR: Replace placeholders!" || echo "OK"

kubectl apply -f model-storage-pvc.yaml
```

### 🔍 VERIFY AFTER

```bash
kubectl get pvc llm-model-storage -n $MY_NAMESPACE
# STATUS must be "Bound"
# If "Pending" after 2 minutes:
kubectl describe pvc llm-model-storage -n $MY_NAMESPACE | tail -10
```

---

## Step 3 — Stage model weights (air-gapped cluster)

Since pods cannot reach the internet, model weights must be staged from
outside the cluster. The exact method depends on what the platform team
told you. Below are the common patterns.

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm you know the ingestion path (from platform team answer)
# Options:
# A) Bastion host / jump box with NFS mount to Powerscale
# B) S3-compatible object store that Powerscale can read
# C) Direct NFS mount accessible from your workstation
# D) Kubectl cp into a running pod (slow for 140 GB but works)
```

### Option A — Bastion host with NFS mount (most common in enterprise)

From the bastion host that has both internet access AND NFS access:

```bash
# On the bastion host:

# 1. Install huggingface_hub
pip install huggingface_hub

# 2. Log in (if using a gated model like Llama)
huggingface-cli login
# Paste your token

# 3. Download to the NFS mount point
#    Ask the platform team for the exact mount path
#    e.g. /mnt/powerscale/llm-models/ or /nfs/shared/models/
huggingface-cli download \
  meta-llama/Llama-3.3-70B-Instruct \
  --local-dir /mnt/powerscale/llm-models/llama-3.3-70b \
  --exclude "*.gguf" "*.bin"

# This downloads ~140 GB. Takes 30-60 minutes.
# If using an ungated model (no HF account needed):
#   huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
#     --local-dir /mnt/powerscale/llm-models/qwen2.5-72b
```

### Option B — Meta direct download (no HF account)

```bash
# On a machine with internet access:
# 1. Go to https://llama.meta.com/llama-downloads/
# 2. Accept the license, receive a signed URL via email
# 3. Run Meta's download script:
./download.sh
# 4. Transfer the downloaded directory to the NFS mount
```

### Option C — Kubectl cp fallback (slow but always works)

If you have no bastion host but can download to your local machine:

```bash
# On your local machine (with internet):
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
  --local-dir ./llama-3.3-70b \
  --exclude "*.gguf" "*.bin"

# Start a temporary pod that mounts the PVC
kubectl run model-loader --restart=Never \
  --image=busybox \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "loader",
        "image": "busybox",
        "command": ["sleep", "86400"],
        "volumeMounts": [{"name":"ms","mountPath":"/models"}]
      }],
      "volumes": [{"name":"ms","persistentVolumeClaim":{"claimName":"llm-model-storage"}}]
    }
  }' \
  -n $MY_NAMESPACE

# Wait for the pod to be Running
kubectl get pod model-loader -n $MY_NAMESPACE -w

# Copy weights into the PVC (this will be SLOW for 140 GB)
kubectl cp ./llama-3.3-70b $MY_NAMESPACE/model-loader:/models/llama-3.3-70b

# When done, delete the loader pod
kubectl delete pod model-loader -n $MY_NAMESPACE
```

### 🔍 VERIFY AFTER (regardless of which option you used)

```bash
# Verify the model files are on the PVC
kubectl run verify-weights --rm -it --restart=Never \
  --image=busybox \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "verify",
        "image": "busybox",
        "command": ["sh", "-c", "ls -lh /models/ && echo --- && ls /models/llama-3.3-70b/ | head -20 && echo --- && du -sh /models/llama-3.3-70b/"],
        "volumeMounts": [{"name":"ms","mountPath":"/models"}]
      }],
      "volumes": [{"name":"ms","persistentVolumeClaim":{"claimName":"llm-model-storage"}}]
    }
  }' \
  -n $MY_NAMESPACE

# Expected output:
# - A directory listing showing *.safetensors files
# - Total size ~140 GB
# If you see less than 100 GB or no safetensors files: download is incomplete
```

```
MY_MODEL_PATH=              # path inside the PVC, e.g. "/models/llama-3.3-70b"
MY_MODEL_NAME=              # what you'll name it in vLLM, e.g. "llama-3.3-70b-instruct"
```

---

## Step 4 — Register the model in ClearML

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm ClearML CLI is configured and working
python3 -c "from clearml import Task; print('ClearML SDK OK')"
```

### Execute

Create `register_model.py` — **update the values in the config section:**

```python
# register_model.py
from clearml import Task, OutputModel

# ============================================================
# CONFIGURE THESE
# ============================================================
PROJECT_NAME     = "production-llm"
MODEL_NAME       = "llama-3.3-70b-instruct"       # <-- match MY_MODEL_NAME
CLUSTER_PATH     = "/models/llama-3.3-70b"         # <-- match MY_MODEL_PATH
GPU_TYPE         = "H200 (141 GB HBM3e)"           # <-- or "H100 (80 GB HBM3)"
MIN_GPUS         = 2                                # <-- match MY_TENSOR_PARALLEL_SIZE
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
    "recommended_gpus": 4,
})

task.close()
print(f"Model registered: {out_model.id}")
```

```bash
python3 register_model.py
```

### 🔍 VERIFY AFTER

Open ClearML Web UI → Models → search for your model name. Confirm it appears
with correct metadata (path, GPU type, minimum GPUs).

---

## Step 5 — Deploy vLLM inference server

vLLM runs as a container image (`vllm/vllm-openai`) inside a Kubernetes pod.
You do NOT install vLLM on the cluster nodes — Kubernetes pulls the image and
runs it. The managed pull secrets handle Docker Hub authentication.

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# 1. Confirm enough free GPUs on non-MIG nodes
for node in <MY_FULL_GPU_NODES>; do
  echo "--- $node ---"
  kubectl describe node $node | grep -A5 "Allocated resources" | grep nvidia
done
# Free GPUs must be >= MY_TENSOR_PARALLEL_SIZE

# 2. Confirm model weights PVC is Bound
kubectl get pvc llm-model-storage -n $MY_NAMESPACE -o jsonpath='{.status.phase}'
# Must show: "Bound"

# 3. Confirm pull secret exists in your namespace
kubectl get secret $MY_PULL_SECRET_NAME -n $MY_NAMESPACE
# Must exist

# 4. (Optional) Confirm the vLLM image is pullable
kubectl run test-vllm-pull --rm -it --restart=Never \
  --overrides="{\"spec\":{\"imagePullSecrets\":[{\"name\":\"$MY_PULL_SECRET_NAME\"}]}}" \
  --image=vllm/vllm-openai:latest -- echo "Image pull OK"
# If this fails: the managed credentials may not include Docker Hub,
# or the image name has changed. Check with platform team.
```

### Execute

Create `vllm-deployment.yaml` — **replace ALL placeholders marked with `<>`:**

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: <MY_NAMESPACE>                       # <-- REPLACE
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
        - name: <MY_PULL_SECRET_NAME>              # <-- REPLACE (remove block if not needed)
      nodeSelector:
        gpu-mode: full                             # <-- targets non-MIG nodes (from Step 0.4)
                                                   #     change key/value to match YOUR label
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          command:
            - python3
            - -m
            - vllm.entrypoints.openai.api_server
          args:
            - --model=<MY_MODEL_PATH>              # <-- REPLACE (e.g. /models/llama-3.3-70b)
            - --served-model-name=<MY_MODEL_NAME>  # <-- REPLACE (e.g. llama-3.3-70b-instruct)
            - --tensor-parallel-size=<MY_TENSOR_PARALLEL_SIZE>  # <-- REPLACE (2 or 4)
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
              nvidia.com/gpu: "<MY_TENSOR_PARALLEL_SIZE>"   # <-- REPLACE (must match above)
            requests:
              nvidia.com/gpu: "<MY_TENSOR_PARALLEL_SIZE>"   # <-- REPLACE
              memory: "64Gi"
              cpu: "8"
          volumeMounts:
            - name: model-storage
              mountPath: /models
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
            claimName: llm-model-storage
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "16Gi"
      tolerations:
        - key: <MY_GPU_TAINT_KEY>                  # <-- REPLACE (or remove block if no taints)
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-inference
  namespace: <MY_NAMESPACE>                        # <-- REPLACE
spec:
  selector:
    app: vllm-inference
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

**Key elements explained:**
- `nodeSelector: gpu-mode: full` — ensures the pod lands on non-MIG nodes. Change to match your label from Step 0.4.
- `/dev/shm` volume — required for NCCL multi-GPU communication. Without it, tensor parallelism crashes.
- `--gpu-memory-utilization=0.90` — uses 90% of each GPU's 141 GB for weights + KV cache.
- `readinessProbe` at 180s — vLLM takes 3-5 minutes to load 140 GB of weights. Don't let K8s kill it early.
- `livenessProbe` at 300s — even longer grace period to prevent restarts during loading.

```bash
# Verify no placeholders remain
grep "<" vllm-deployment.yaml && echo "ERROR: Replace all <> placeholders!" || echo "OK"

kubectl apply -f vllm-deployment.yaml

# Watch the pod start (3-5 minutes for weight loading)
kubectl get pods -n $MY_NAMESPACE -l app=vllm-inference -w

# Follow logs to see loading progress
kubectl logs -f deployment/vllm-inference -n $MY_NAMESPACE
# Wait for: "INFO:     Application startup complete."
```

### 🔍 VERIFY AFTER

```bash
# Pod status
kubectl get pods -n $MY_NAMESPACE -l app=vllm-inference
# STATUS=Running, READY=1/1

# If Pending: check why
kubectl describe pod -l app=vllm-inference -n $MY_NAMESPACE | grep -A10 "Events"
# Common: not enough GPUs, nodeSelector doesn't match, taint not tolerated

# If CrashLoopBackOff: check logs
kubectl logs -l app=vllm-inference -n $MY_NAMESPACE --tail=50

# Health check
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- \
  curl -s http://localhost:8000/health
# Expected: {"status":"ok"}

# GPU usage check
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- nvidia-smi
# Expected: 2+ GPUs showing ~90% memory used

# Confirm it landed on a non-MIG node
kubectl get pod -l app=vllm-inference -n $MY_NAMESPACE -o wide
# NODE column should show one of MY_FULL_GPU_NODES, NOT a MIG node

# Test inference
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- \
  curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<MY_MODEL_NAME>","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
# Expected: a coherent response from the LLM

# Test service DNS
kubectl run test-svc --rm -it --restart=Never --image=busybox -n $MY_NAMESPACE -- \
  wget -qO- http://vllm-inference:8000/health --timeout=10
# Expected: {"status":"ok"}
```

---

## Step 6 — Deploy LiteLLM proxy

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm vLLM service is healthy
kubectl run test-vllm --rm -it --restart=Never --image=busybox -n $MY_NAMESPACE -- \
  wget -qO- http://vllm-inference:8000/health --timeout=10
# Must show: {"status":"ok"}
# If not: STOP. Fix Step 5 first.
```

### Execute

Create `litellm-config.yaml` — **replace placeholders:**

```yaml
# litellm-config.yaml
model_list:
  - model_name: <MY_MODEL_NAME>                                                    # <-- REPLACE
    litellm_params:
      model: openai/<MY_MODEL_NAME>                                                # <-- REPLACE
      api_base: http://vllm-inference.<MY_NAMESPACE>.svc.cluster.local:8000/v1     # <-- REPLACE namespace
      api_key: "not-used"

litellm_settings:
  drop_params: true
  request_timeout: 300

general_settings:
  master_key: "<GENERATE_A_RANDOM_KEY>"       # <-- REPLACE (e.g. sk-xxxxxxxxxxxx)
```

```bash
# Create ConfigMap
kubectl create configmap litellm-config \
  --from-file=config.yaml=litellm-config.yaml \
  -n $MY_NAMESPACE
```

Create `litellm-deployment.yaml` — **replace placeholders:**

```yaml
# litellm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-proxy
  namespace: <MY_NAMESPACE>             # <-- REPLACE
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
        - name: <MY_PULL_SECRET_NAME>    # <-- REPLACE (remove block if not needed)
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
  namespace: <MY_NAMESPACE>              # <-- REPLACE
spec:
  selector:
    app: litellm-proxy
  ports:
    - port: 4000
      targetPort: 4000
  type: NodePort
```

```bash
# Verify no placeholders
grep "<" litellm-deployment.yaml && echo "ERROR: Replace placeholders!" || echo "OK"

kubectl apply -f litellm-deployment.yaml
kubectl get pods -n $MY_NAMESPACE -l app=litellm-proxy -w
```

### 🔍 VERIFY AFTER

```bash
# Get access URL
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc litellm-proxy -n $MY_NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
echo "LiteLLM URL: http://$NODE_IP:$NODE_PORT"

# Health check
curl -s http://$NODE_IP:$NODE_PORT/health

# End-to-end test
curl -s -X POST http://$NODE_IP:$NODE_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_MASTER_KEY>" \
  -d "{
    \"model\": \"<MY_MODEL_NAME>\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What model are you?\"}],
    \"max_tokens\": 100
  }"
# Expected: a coherent response from the LLM
# If you get this: THE FULL STACK IS WORKING.
```

---

## Step 7 — Log an inference run to ClearML

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm LiteLLM is reachable from your local machine
curl -s http://<LITELLM_URL>/health
# Must show healthy
```

### Execute

Create `inference_task.py` — **update the config section:**

```python
# inference_task.py
import os, json, urllib.request
from clearml import Task, InputModel

# ============================================================
# CONFIGURE THESE
# ============================================================
LITELLM_URL  = "http://<NODE_IP>:<NODE_PORT>"    # <-- REPLACE
API_KEY      = "<YOUR_MASTER_KEY>"                # <-- REPLACE
MODEL_NAME   = "<MY_MODEL_NAME>"                  # <-- REPLACE
PROJECT_NAME = "production-llm"
# ============================================================

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
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    logger.report_text(f"Query {i}: {prompt}\nResponse: {text}")
    logger.report_scalar("Token usage", "prompt_tokens", usage.get("prompt_tokens", 0), i)
    logger.report_scalar("Token usage", "completion_tokens", usage.get("completion_tokens", 0), i)
    results.append({"prompt": prompt, "response": text, "tokens": usage})
    print(f"Query {i} done. Tokens: {usage}")

task.upload_artifact("inference_results", artifact_object=results)
task.close()
print(f"Done. Check ClearML UI -> {PROJECT_NAME} -> Tasks")
```

```bash
python3 inference_task.py
```

### 🔍 VERIFY AFTER

Open ClearML Web UI → your project → Tasks. Confirm: prompts and responses in the
log, token usage graphs, and the inference_results artifact.

---

## Step 8 — Monitor GPU health

```bash
# Quick GPU status
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- nvidia-smi
# Look for:
#   Memory: ~90% used per GPU
#   Temperature: under 80°C
#   Power: up to 700W (H200 SXM) under load

# DCGM diagnostics (if available)
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- dcgmi diag -r 1

# Real-time metrics stream
kubectl exec deployment/vllm-inference -n $MY_NAMESPACE -- \
  dcgmi dmon -e 100,101,102,203,204 -d 2000
# 100=SM util%  101=mem util%  102=mem used(MB)  203=temp(C)  204=power(W)

# Check NCCL topology (multi-GPU communication path)
kubectl logs deployment/vllm-inference -n $MY_NAMESPACE | grep -i "nccl\|nvlink\|topology"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Pod `Pending` | Not enough free GPUs, or nodeSelector doesn't match | `kubectl describe pod` → Events section |
| Pod `Pending` + "didn't match node selector" | Label mismatch | Verify label: `kubectl get nodes --show-labels \| grep <label>` |
| Pod `CrashLoopBackOff` | OOM or bad args | Check logs. If OOM: reduce `--max-model-len` to `16384` |
| vLLM killed after 2 min | Liveness probe too aggressive | Increase `livenessProbe.initialDelaySeconds` to `600` |
| NCCL shared memory error | Missing `/dev/shm` mount | Add the `shm` emptyDir volume (see Step 5 YAML) |
| Pod scheduled on MIG node | nodeSelector missing or wrong | Add/fix `nodeSelector` with your non-MIG label |
| LiteLLM returns 502 | vLLM not ready | Wait 5 min. Check vLLM logs for "Application startup complete" |
| Context length exceeded | Prompt too long | Increase `--max-model-len` (needs more GPU memory) |
| `ImagePullBackOff` | Pull secret missing in namespace | Copy secret: Step 1 instructions |
| `ErrImagePull` + auth error | Wrong secret or expired token | `kubectl describe pod` → check image name and secret |
| nvidia-smi shows 0% util | No requests in flight | Normal when idle. Send a test request. |

---

## Cleanup

```bash
# Remove in reverse order
kubectl delete deployment litellm-proxy -n $MY_NAMESPACE
kubectl delete svc litellm-proxy -n $MY_NAMESPACE
kubectl delete configmap litellm-config -n $MY_NAMESPACE
kubectl delete deployment vllm-inference -n $MY_NAMESPACE
kubectl delete svc vllm-inference -n $MY_NAMESPACE
kubectl delete pvc llm-model-storage -n $MY_NAMESPACE   # WARNING: deletes model weights
```

---

## Reference: local lab → production mapping

| Topic | Local lab | Production (this guide) |
|-------|-----------|------------------------|
| Model storage | Local filesystem | Kubernetes PVC on NFS |
| Inference engine | Custom serve.py | vLLM (PagedAttention, continuous batching) |
| GPU | 1× consumer GPU | 2-8× H200 (141 GB HBM3e each) |
| Context window | 128 chars | 32K–131K tokens |
| Concurrency | 1 request | Hundreds (continuous batching) |
| API gateway | Docker LiteLLM | K8s LiteLLM Deployment |
| Model download | Python script | Air-gap staging via bastion/NFS |
| Secrets | Hardcoded | Kubernetes Secrets |
| Networking | localhost | K8s Service DNS |
| Multi-GPU | N/A | NCCL over NVLink (/dev/shm) |
| GPU targeting | N/A | nodeSelector to avoid MIG nodes |
| Monitoring | nvidia-smi | DCGM + Prometheus |

The API clients call is identical in both environments:
`POST /v1/chat/completions` with `Authorization: Bearer <key>`.

---

## Next steps

1. **Add Ingress** — DNS name + TLS in front of LiteLLM
2. **LiteLLM virtual keys** — per-user rate limiting and usage tracking
3. **vLLM metrics** — add `--enable-metrics`, scrape with Prometheus
4. **ClearML Pipeline** — auto-redeploy when a new model is registered
5. **Scale up** — try higher `--tensor-parallel-size` for full 131K context
6. **Point dev tools** — configure VS Code / Continue.dev to use the LiteLLM endpoint
7. **ClearML queue integration** — if the platform team confirms ClearML should manage
   the inference lifecycle, create a dedicated queue for long-running serving tasks
