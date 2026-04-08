# Deploying Llama 3.3 70B on H200 GPUs with vLLM + ClearML

A step-by-step production deployment guide. Every step includes a **🔍 VERIFY BEFORE PROCEEDING** gate so you confirm your cluster's actual configuration before running anything.

> **Your starting point:** ClusterAdmin access to a K8s cluster with H200 GPUs, ClearML already installed, and hands-on experience from the sovereign-lab (Phase 1/2/3 pipeline).

> **What this guide produces:** Llama 3.3 70B running via vLLM on H200 GPUs, an OpenAI-compatible API behind a LiteLLM proxy, and the model tracked in ClearML — the production-scale version of your home lab.

---

## How this maps to what you already know

| Local lab | Production (this guide) |
| --------- | ----------------------- |
| TinyGPT ~500K params, pure PyTorch | Llama 3.3 70B, loaded by vLLM |
| `serve.py` (stdlib HTTP server) | vLLM OpenAI server (production-grade) |
| GTX 1070 Ti, 8 GB VRAM, float32 | H200, 141 GB HBM3e, bfloat16 |
| Model registered in local ClearML | Model registered in cluster ClearML |
| LiteLLM on port 4100 (Docker) | LiteLLM on cluster (Kubernetes Deployment) |
| `phase3_inference/run.py` | This guide |

The concepts are identical. The only differences are scale and the inference engine.

---

## Step 0 — Discover YOUR cluster

Before touching anything, gather the facts about the cluster you're working on. Every value you collect here will be referenced in later steps.

### 0.1 — Confirm cluster access and your permissions

```bash
# Can you reach the cluster?
kubectl cluster-info

# What nodes exist?
kubectl get nodes -o wide

# Do you actually have ClusterAdmin?
kubectl auth can-i '*' '*' --all-namespaces
# Expected: "yes"
# If "no": STOP. You need ClusterAdmin before proceeding.
```

### 0.2 — Discover GPU nodes and count

```bash
# List nodes with GPU counts
kubectl get nodes -o custom-columns=\
"NAME:.metadata.name,\
GPUs:.status.allocatable.nvidia\.com/gpu,\
MEMORY:.status.allocatable.memory,\
OS:.status.nodeInfo.osImage"
```

**🔍 VERIFY:** Write down your answers:
```
MY_GPU_NODE_NAMES=          # e.g. "gpu-node-01, gpu-node-02"
MY_GPU_COUNT_PER_NODE=      # e.g. "8"
MY_TOTAL_GPUS_AVAILABLE=    # e.g. "16"
```

### 0.3 — Confirm these are actually H200s (not H100s)

```bash
# SSH into a GPU node or exec into any pod on a GPU node and run:
kubectl debug node/<YOUR_GPU_NODE_NAME> -it --image=nvidia/cuda:12.4.0-base-ubuntu22.04 -- nvidia-smi

# Look for the GPU name and memory:
# "NVIDIA H200" and "141 GB" (or ~143360 MiB)
#
# If you see "NVIDIA H100" and "80 GB" (or ~81920 MiB):
#   You have H100s, not H200s. The guide still works but:
#   - You need MINIMUM 2 GPUs (2×80 = 160 GB) just for weights
#   - Recommended: 4 GPUs for usable KV cache
#   - Change --tensor-parallel-size=4 in Step 5
```

**🔍 VERIFY:** Write down:
```
MY_GPU_TYPE=                # "H200" or "H100"
MY_GPU_MEMORY_GB=           # 141 (H200) or 80 (H100)
```

**GPU count decision based on what you just found:**

| GPU type | VRAM each | Minimum GPUs for Llama 70B bf16 | Recommended |
|----------|-----------|--------------------------------|-------------|
| H200 | 141 GB | 2 (282 GB total) | 4 |
| H100 | 80 GB | 2 (160 GB — tight) | 4 |

```
MY_TENSOR_PARALLEL_SIZE=    # 2 for H200, 4 for H100 (or adjust based on available GPUs)
```

### 0.4 — Discover the ClearML namespace and services

```bash
# Find where ClearML is running
kubectl get pods --all-namespaces | grep -i clearml

# Note the namespace
kubectl get svc -n <CLEARML_NAMESPACE>
```

**🔍 VERIFY:** Write down:
```
MY_CLEARML_NAMESPACE=       # e.g. "mlops-platform", "clearml", "default"
MY_CLEARML_API_URL=         # e.g. "http://clearml-api:8008" or NodePort URL
MY_CLEARML_WEB_URL=         # e.g. "http://clearml-web:8080"
MY_CLEARML_FILES_URL=       # e.g. "http://clearml-files:8081"
```

Open `MY_CLEARML_WEB_URL` in your browser. Log in. Create API credentials:
`Settings → Workspace → Create new credentials`

```
MY_CLEARML_ACCESS_KEY=      # from ClearML UI
MY_CLEARML_SECRET_KEY=      # from ClearML UI
```

### 0.5 — Discover available storage classes

```bash
kubectl get storageclass
```

**🔍 VERIFY:** Write down:
```
MY_STORAGE_CLASS=           # e.g. "gp3", "standard", "local-path", "ceph-rbd"
```

If you're unsure which one to use, pick the one marked `(default)` in the output.
If none is marked default, ask the cluster owner.

### 0.6 — Check existing GPU usage

```bash
# Are any GPUs currently in use?
kubectl get pods --all-namespaces -o wide | grep -i gpu

# Check GPU allocations per node
kubectl describe nodes | grep -A10 "Allocated resources" | grep -A3 "nvidia"
```

**🔍 VERIFY:** Write down:
```
MY_FREE_GPUS=               # e.g. "6 out of 8 available"
```

If fewer GPUs are free than `MY_TENSOR_PARALLEL_SIZE`, you need to wait or coordinate with other users.

### 0.7 — Check network egress (can the cluster pull images and download models?)

```bash
# Test: can a pod reach the internet?
kubectl run test-egress --rm -it --restart=Never --image=busybox -- wget -qO- https://huggingface.co --timeout=10
# If this hangs or fails: the cluster may be air-gapped.
# You'll need to pre-load the vLLM image and model weights. See 0.7a and 0.7b below.
```

**🔍 VERIFY:** Write down:
```
MY_CLUSTER_HAS_INTERNET=    # "yes" or "no"
```

### 0.7a — (If air-gapped / sandboxed) Discover the internal container registry

Most corporate clusters pull images from an internal registry, not Docker Hub.

```bash
# Check if there's an existing pull secret that tells you the registry URL
kubectl get secrets --all-namespaces | grep -i "registry\|pull\|docker"

# Check what registry existing pods use
kubectl get pods --all-namespaces -o jsonpath='{range .items[*]}{.spec.containers[*].image}{"\n"}{end}' | sort -u | head -20
# Look for patterns like:
#   artifactory.yourcompany.com/docker/nginx:latest
#   registry.internal.corp/images/python:3.11
#   123456789.dkr.ecr.us-east-1.amazonaws.com/app:v1
# The domain before the first "/" is your internal registry
```

**🔍 VERIFY:** Write down:
```
MY_CONTAINER_REGISTRY=      # e.g. "artifactory.corp.com/docker" or "docker.io" if public
MY_IMAGE_PULL_SECRET=       # e.g. "registry-credentials" or "none" if public
```

### 0.7b — (If air-gapped / sandboxed) Pre-load the vLLM image

**This is important:** vLLM is NOT pre-installed on the cluster. It runs as a container image that Kubernetes pulls when you create the Deployment in Step 5. If the cluster can't reach Docker Hub, you need to get the image into the internal registry first.

```bash
# On a machine WITH internet access (your laptop, a bastion host, a CI runner):
docker pull vllm/vllm-openai:latest
docker pull ghcr.io/berriai/litellm:main-stable
docker pull python:3.11-slim

# Tag them for your internal registry
docker tag vllm/vllm-openai:latest $MY_CONTAINER_REGISTRY/vllm-openai:latest
docker tag ghcr.io/berriai/litellm:main-stable $MY_CONTAINER_REGISTRY/litellm:main-stable
docker tag python:3.11-slim $MY_CONTAINER_REGISTRY/python:3.11-slim

# Push to internal registry
docker push $MY_CONTAINER_REGISTRY/vllm-openai:latest
docker push $MY_CONTAINER_REGISTRY/litellm:main-stable
docker push $MY_CONTAINER_REGISTRY/python:3.11-slim

# NOTE: vllm-openai:latest is a large image (~8-10 GB). This push will take a while.
```

If you can't use Docker directly, ask your platform team how they onboard new images. There's always a process — they got ClearML's images in somehow.

**🔍 VERIFY:** Write down the image names you'll use in later steps:
```
MY_VLLM_IMAGE=              # e.g. "artifactory.corp.com/docker/vllm-openai:latest" or "vllm/vllm-openai:latest"
MY_LITELLM_IMAGE=           # e.g. "artifactory.corp.com/docker/litellm:main-stable" or "ghcr.io/berriai/litellm:main-stable"
MY_PYTHON_IMAGE=            # e.g. "artifactory.corp.com/docker/python:3.11-slim" or "python:3.11-slim"
```

### 0.8 — Set up your deployment namespace

This is where all your LLM serving components will live. You have two choices:

```bash
# Option A: Deploy in the SAME namespace as ClearML
# Pros: vLLM can reach ClearML services by short name (e.g. "clearml-api:8008")
# Cons: Your pods mix with ClearML pods, harder to clean up
MY_NAMESPACE=$MY_CLEARML_NAMESPACE

# Option B: Deploy in a DEDICATED namespace (recommended for production)
# Pros: Clean separation, easier RBAC, easier to tear down
# Cons: Need full DNS names for cross-namespace communication
MY_NAMESPACE="llm-serving"
```

#### If creating a new namespace:

```bash
# Check it doesn't already exist
kubectl get namespace $MY_NAMESPACE 2>/dev/null && echo "ALREADY EXISTS" || echo "OK to create"

# Create it
kubectl create namespace $MY_NAMESPACE

# If your cluster requires an image pull secret, copy it to the new namespace
# (pods can only use secrets in their own namespace)
kubectl get secret $MY_IMAGE_PULL_SECRET -n $MY_CLEARML_NAMESPACE -o yaml \
  | sed "s/namespace: .*/namespace: $MY_NAMESPACE/" \
  | kubectl apply -f -

# Verify the namespace is active
kubectl get namespace $MY_NAMESPACE
# STATUS should be "Active"
```

#### If your cluster uses ResourceQuotas or LimitRanges:

```bash
# Check if there are quotas that might block your deployment
kubectl get resourcequota -n $MY_NAMESPACE
kubectl get limitrange -n $MY_NAMESPACE

# If quotas exist, verify they allow GPU requests:
kubectl describe resourcequota -n $MY_NAMESPACE | grep -i gpu
# If "nvidia.com/gpu" has a limit, make sure it's >= MY_TENSOR_PARALLEL_SIZE
# If it's too low, ask the cluster admin to increase it for your namespace
```

**🔍 VERIFY:** Write down:
```
MY_NAMESPACE=               # where you'll deploy everything
```

### 0.9 — Summary checklist

Before proceeding, confirm you have ALL of these:

```
[ ] MY_GPU_TYPE             = ___________
[ ] MY_GPU_MEMORY_GB        = ___________
[ ] MY_TENSOR_PARALLEL_SIZE = ___________
[ ] MY_CLEARML_NAMESPACE    = ___________
[ ] MY_CLEARML_API_URL      = ___________
[ ] MY_NAMESPACE            = ___________
[ ] MY_STORAGE_CLASS        = ___________
[ ] MY_FREE_GPUS            = ___________ (must be >= MY_TENSOR_PARALLEL_SIZE)
[ ] MY_CLUSTER_HAS_INTERNET = ___________
[ ] MY_CONTAINER_REGISTRY   = ___________ (or "docker.io" if public internet)
[ ] MY_IMAGE_PULL_SECRET    = ___________ (or "none")
[ ] MY_VLLM_IMAGE           = ___________ (full image path for vLLM)
[ ] MY_LITELLM_IMAGE        = ___________ (full image path for LiteLLM)
[ ] MY_PYTHON_IMAGE         = ___________ (full image path for python:3.11-slim)
[ ] Hugging Face token OR Meta direct download URL OR ungated model chosen OR weights pre-staged
[ ] ClearML access key and secret key
```

**If any item is blank, STOP and fill it in before continuing.**

> **How vLLM gets onto the cluster:** vLLM is not a system package you install
> on nodes. It's a container image (`vllm/vllm-openai`) that Kubernetes pulls
> and runs as a pod — just like ClearML, LiteLLM, and every other component.
> If the cluster has internet, Kubernetes pulls it from Docker Hub automatically.
> If the cluster is air-gapped, you pre-load it into your internal registry
> (Step 0.7b) and reference that registry in your Deployment YAML.

---

## Step 1 — Create the Hugging Face secret

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm your HF token works and has access to the gated model
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.model_info('meta-llama/Llama-3.3-70B-Instruct', token='YOUR_TOKEN_HERE')
print('SUCCESS: Token has access to Llama 3.3 70B')
"
# If you see "401" or "403": your token doesn't have access.
# Go to https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct and accept the license.
```

### Execute

```bash
# Check if the secret already exists (don't create duplicates)
kubectl get secret hf-credentials -n $MY_NAMESPACE 2>/dev/null && echo "SECRET ALREADY EXISTS" || echo "OK to create"

# Create it
kubectl create secret generic hf-credentials \
  --from-literal=token=YOUR_HF_TOKEN_HERE \
  -n $MY_NAMESPACE

# Verify it was created
kubectl get secret hf-credentials -n $MY_NAMESPACE
```

---

## Step 2 — Create persistent storage for model weights

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm your storage class exists and supports the access mode
kubectl get storageclass $MY_STORAGE_CLASS
# Look at PROVISIONER column — this tells you what backend it uses

# Check if someone already created this PVC (maybe from a previous attempt)
kubectl get pvc llama-model-storage -n $MY_NAMESPACE 2>/dev/null && echo "PVC ALREADY EXISTS — skip this step" || echo "OK to create"
```

### Execute

Create the file `model-storage-pvc.yaml` — **substitute YOUR values:**

```yaml
# model-storage-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llama-model-storage
  namespace: MY_NAMESPACE              # <-- REPLACE with your namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 300Gi
  storageClassName: MY_STORAGE_CLASS   # <-- REPLACE with your storage class
```

```bash
# Before applying, confirm the YAML has your actual values (not placeholders)
cat model-storage-pvc.yaml | grep -E "namespace:|storageClassName:"
# You should see YOUR values, not "MY_NAMESPACE" or "MY_STORAGE_CLASS"

kubectl apply -f model-storage-pvc.yaml

# Wait for it to bind
kubectl get pvc llama-model-storage -n $MY_NAMESPACE -w
# Expected: STATUS = "Bound"
# If stuck on "Pending" for >2 minutes:
kubectl describe pvc llama-model-storage -n $MY_NAMESPACE | tail -10
```

---

## Step 3 — Download Llama 3.3 70B weights

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm the PVC is bound
kubectl get pvc llama-model-storage -n $MY_NAMESPACE -o jsonpath='{.status.phase}'
# Must show: "Bound"
# If not "Bound": STOP. Fix the PVC issue from Step 2 first.

# Confirm the HF secret exists
kubectl get secret hf-credentials -n $MY_NAMESPACE -o jsonpath='{.data.token}' | base64 -d | head -c 10
# Should show the first 10 chars of your token (e.g. "hf_xxxxxx")
```

### Execute

Create `download-job.yaml` — **substitute YOUR namespace:**

```yaml
# download-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llama-download
  namespace: MY_NAMESPACE              # <-- REPLACE
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      # Uncomment if your cluster requires an image pull secret:
      # imagePullSecrets:
      #   - name: MY_IMAGE_PULL_SECRET  # <-- REPLACE
      containers:
        - name: downloader
          image: MY_PYTHON_IMAGE         # <-- REPLACE (e.g. "python:3.11-slim" or internal registry path)
          command:
            - /bin/sh
            - -c
            - |
              pip install -q huggingface_hub && \
              python3 -c "
              from huggingface_hub import snapshot_download
              import os
              print(f'Starting download to /models/llama-3.3-70b')
              print(f'This will download ~140 GB. Be patient.')
              snapshot_download(
                  repo_id='meta-llama/Llama-3.3-70B-Instruct',
                  local_dir='/models/llama-3.3-70b',
                  token=os.environ['HF_TOKEN'],
                  ignore_patterns=['*.gguf', '*.bin'],
              )
              print('Download complete')
              "
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
          volumeMounts:
            - name: model-storage
              mountPath: /models
          resources:
            requests:
              memory: "8Gi"
              cpu: "2"
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: llama-model-storage
```

```bash
# Confirm no placeholder values remain
grep "MY_NAMESPACE" download-job.yaml && echo "ERROR: Replace MY_NAMESPACE!" || echo "OK"

kubectl apply -f download-job.yaml

# Watch progress — this takes 30-60 minutes
kubectl logs -f job/llama-download -n $MY_NAMESPACE

# If it fails and you need to retry:
# kubectl delete job llama-download -n $MY_NAMESPACE
# kubectl apply -f download-job.yaml
# (huggingface_hub resumes partial downloads automatically)
```

### 🔍 VERIFY AFTER COMPLETION

```bash
# Confirm the download finished
kubectl get job llama-download -n $MY_NAMESPACE -o jsonpath='{.status.succeeded}'
# Must show: "1"

# Verify the model files exist on the PV
kubectl run verify-download --rm -it --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "verify",
        "image": "busybox",
        "command": ["sh", "-c", "ls -lh /models/llama-3.3-70b/ && du -sh /models/llama-3.3-70b/"],
        "volumeMounts": [{"name":"ms","mountPath":"/models"}]
      }],
      "volumes": [{"name":"ms","persistentVolumeClaim":{"claimName":"llama-model-storage"}}]
    }
  }' \
  --image=busybox -n $MY_NAMESPACE
# Expected: files totaling ~140 GB, including *.safetensors files
```

---

## Step 4 — Register the model in ClearML

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm ClearML is reachable from your local machine
python3 -c "
from clearml import Task
print('ClearML SDK version:', Task.__version__)
print('Connection test...')
t = Task.init(project_name='_test', task_name='_connection_test', reuse_last_task_id=True)
t.close()
print('SUCCESS: ClearML is reachable')
"
# If this fails: run 'clearml-init' and enter your cluster ClearML credentials
```

### Execute

Create `register_model.py`:

```python
# register_model.py
from clearml import Task, OutputModel

# Update these if your GPU type is different
GPU_TYPE = "H200 (141 GB HBM3e)"   # <-- CHANGE if using H100s
MIN_GPUS = 2                        # <-- CHANGE if using H100s (set to 4)

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
    "recommended_gpus": 4,
})

task.close()
print(f"Model registered. ID: {out_model.id}")
```

```bash
python3 register_model.py
```

### 🔍 VERIFY AFTER COMPLETION

Open ClearML web UI → Models → search for `llama-3.3-70b-instruct`. Confirm it appears with the correct metadata.

---

## Step 5 — Deploy vLLM inference server

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm enough GPUs are free for your tensor-parallel-size
echo "Need $MY_TENSOR_PARALLEL_SIZE GPUs. Checking availability..."
kubectl describe nodes | grep -A5 "nvidia.com/gpu"
# "Allocatable" minus "Allocated" must be >= MY_TENSOR_PARALLEL_SIZE

# Confirm the model weights PVC is bound
kubectl get pvc llama-model-storage -n $MY_NAMESPACE -o jsonpath='{.status.phase}'
# Must show: "Bound"

# Confirm the vLLM image is pullable (test on a non-GPU node)
kubectl run test-pull --rm -it --restart=Never --image=vllm/vllm-openai:latest -- echo "Image pulled OK" 2>/dev/null
# If this hangs: the cluster can't reach Docker Hub. You need to pre-load the image.

# Confirm the GPU toleration matches your cluster's taint
kubectl describe nodes | grep -i taint
# Note the exact taint key (e.g. "nvidia.com/gpu", "gpu=true", or no taint at all)
# If your cluster uses a DIFFERENT taint key, change the toleration in the YAML below
```

```
MY_GPU_TAINT_KEY=           # e.g. "nvidia.com/gpu" (default) or whatever your cluster uses
```

### Execute

Create `vllm-deployment.yaml` — **substitute YOUR values in the marked lines:**

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b
  namespace: MY_NAMESPACE                # <-- REPLACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama-70b
  template:
    metadata:
      labels:
        app: vllm-llama-70b
    spec:
      # Uncomment if your cluster requires an image pull secret:
      # imagePullSecrets:
      #   - name: MY_IMAGE_PULL_SECRET  # <-- REPLACE
      containers:
        - name: vllm
          image: MY_VLLM_IMAGE           # <-- REPLACE (e.g. "vllm/vllm-openai:latest" or internal registry path)
          command:
            - python3
            - -m
            - vllm.entrypoints.openai.api_server
          args:
            - --model=/models/llama-3.3-70b
            - --served-model-name=llama-3.3-70b-instruct
            - --tensor-parallel-size=MY_TENSOR_PARALLEL_SIZE   # <-- REPLACE (2 for H200, 4 for H100)
            - --dtype=bfloat16
            - --max-model-len=32768
            - --gpu-memory-utilization=0.90
            - --port=8000
            - --host=0.0.0.0
            - --trust-remote-code
            - --enable-chunked-prefill
          ports:
            - containerPort: 8000
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
          resources:
            limits:
              nvidia.com/gpu: "MY_TENSOR_PARALLEL_SIZE"       # <-- REPLACE (must match above)
            requests:
              nvidia.com/gpu: "MY_TENSOR_PARALLEL_SIZE"       # <-- REPLACE (must match above)
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
            claimName: llama-model-storage
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "16Gi"
      tolerations:
        - key: MY_GPU_TAINT_KEY             # <-- REPLACE (e.g. "nvidia.com/gpu")
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-70b
  namespace: MY_NAMESPACE                  # <-- REPLACE
spec:
  selector:
    app: vllm-llama-70b
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

```bash
# Final check: no placeholder values remain
grep -n "MY_" vllm-deployment.yaml
# If ANY lines show "MY_" prefixes: STOP and replace them with your actual values

kubectl apply -f vllm-deployment.yaml

# Watch the pod start (3-5 minutes to load 140 GB of weights)
kubectl get pods -n $MY_NAMESPACE -l app=vllm-llama-70b -w

# Follow logs to see loading progress
kubectl logs -f deployment/vllm-llama-70b -n $MY_NAMESPACE
# Wait for: "INFO:     Application startup complete."
```

### 🔍 VERIFY AFTER DEPLOYMENT

```bash
# Check pod is Running (not CrashLoopBackOff or Pending)
kubectl get pods -n $MY_NAMESPACE -l app=vllm-llama-70b
# STATUS must be "Running" and READY must be "1/1"

# If STATUS is "Pending": check why
kubectl describe pod -l app=vllm-llama-70b -n $MY_NAMESPACE | grep -A5 "Events"
# Common causes: not enough GPUs, PVC can't mount, image can't pull

# If STATUS is "CrashLoopBackOff": check logs
kubectl logs -l app=vllm-llama-70b -n $MY_NAMESPACE --tail=50

# Health check
kubectl exec deployment/vllm-llama-70b -n $MY_NAMESPACE -- curl -s http://localhost:8000/health
# Expected: {"status":"ok"}

# Verify GPU usage
kubectl exec deployment/vllm-llama-70b -n $MY_NAMESPACE -- nvidia-smi
# Expected: 2+ GPUs showing ~90% memory used, model name in process list

# Test inference
kubectl exec deployment/vllm-llama-70b -n $MY_NAMESPACE -- \
  curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-instruct","messages":[{"role":"user","content":"Hello, what model are you?"}],"max_tokens":50}'
# Expected: a coherent response identifying itself as Llama

# Check the service is reachable by name
kubectl run test-svc --rm -it --restart=Never --image=busybox -n $MY_NAMESPACE -- \
  wget -qO- http://vllm-llama-70b:8000/health --timeout=10
# Expected: {"status":"ok"}
```

---

## Step 6 — Deploy LiteLLM proxy

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm vLLM service is responding
kubectl run test-vllm --rm -it --restart=Never --image=busybox -n $MY_NAMESPACE -- \
  wget -qO- http://vllm-llama-70b:8000/health --timeout=10
# Must show: {"status":"ok"}
# If not: STOP. Fix vLLM deployment in Step 5 first.
```

### Execute

Create `litellm-config.yaml`:

```yaml
# litellm-config.yaml
model_list:
  - model_name: llama-3.3-70b-instruct
    litellm_params:
      model: openai/llama-3.3-70b-instruct
      api_base: http://vllm-llama-70b.MY_NAMESPACE.svc.cluster.local:8000/v1   # <-- REPLACE MY_NAMESPACE
      api_key: "not-used"

litellm_settings:
  drop_params: true
  request_timeout: 300

general_settings:
  master_key: "sk-CHANGE-THIS-TO-A-REAL-KEY"    # <-- CHANGE THIS
```

```bash
# Verify no placeholders remain
grep "MY_NAMESPACE\|CHANGE-THIS" litellm-config.yaml
# MY_NAMESPACE should be gone. "CHANGE-THIS" is OK if you've set a real key.

# Create ConfigMap
kubectl create configmap litellm-config \
  --from-file=config.yaml=litellm-config.yaml \
  -n $MY_NAMESPACE

# Verify
kubectl get configmap litellm-config -n $MY_NAMESPACE
```

Create `litellm-deployment.yaml`:

```yaml
# litellm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-proxy
  namespace: MY_NAMESPACE            # <-- REPLACE
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
      # Uncomment if your cluster requires an image pull secret:
      # imagePullSecrets:
      #   - name: MY_IMAGE_PULL_SECRET  # <-- REPLACE
      containers:
        - name: litellm
          image: MY_LITELLM_IMAGE        # <-- REPLACE (e.g. "ghcr.io/berriai/litellm:main-stable" or internal registry path)
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
  namespace: MY_NAMESPACE            # <-- REPLACE
spec:
  selector:
    app: litellm-proxy
  ports:
    - port: 4000
      targetPort: 4000
  type: NodePort                     # Change to LoadBalancer if your cluster supports it
```

```bash
# Verify no placeholders
grep -n "MY_" litellm-deployment.yaml && echo "ERROR: Replace placeholders!" || echo "OK"

kubectl apply -f litellm-deployment.yaml

# Wait for pod to be ready
kubectl get pods -n $MY_NAMESPACE -l app=litellm-proxy -w
```

### 🔍 VERIFY AFTER DEPLOYMENT

```bash
# Get the access URL
# For NodePort:
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc litellm-proxy -n $MY_NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
echo "LiteLLM URL: http://$NODE_IP:$NODE_PORT"

# For LoadBalancer:
# LB_IP=$(kubectl get svc litellm-proxy -n $MY_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
# echo "LiteLLM URL: http://$LB_IP:4000"

# Health check
curl -s http://$NODE_IP:$NODE_PORT/health
# Expected: {"status":"healthy"}

# Full end-to-end test: prompt → LiteLLM → vLLM → H200 GPUs → response
curl -s -X POST http://$NODE_IP:$NODE_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-CHANGE-THIS-TO-A-REAL-KEY" \
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
    "max_tokens": 100
  }'
# Expected: a coherent response about tensor parallelism
# If you get a response: CONGRATULATIONS. The full stack is working.
```

---

## Step 7 — Log an inference run to ClearML

### 🔍 VERIFY BEFORE PROCEEDING

```bash
# Confirm ClearML credentials are configured on your local machine
python3 -c "from clearml import Task; print('ClearML SDK OK')"

# Confirm LiteLLM is reachable from your local machine
# (use the URL from Step 6 verification)
curl -s http://YOUR_LITELLM_URL/health
# Must show: {"status":"healthy"}
```

### Execute

Create `inference_task.py` — **substitute YOUR values:**

```python
# inference_task.py
import os
import json
import urllib.request
from clearml import Task, InputModel

# ============================================================
# CONFIGURE THESE — use the values from YOUR cluster discovery
# ============================================================
LITELLM_URL = "http://NODE_IP:NODE_PORT"            # <-- REPLACE
API_KEY     = "sk-CHANGE-THIS-TO-A-REAL-KEY"         # <-- REPLACE
# ============================================================

task = Task.init(
    project_name="production-llm",
    task_name="llama-3.3-70b-inference-demo",
    task_type=Task.TaskTypes.inference,
    reuse_last_task_id=False,
)
logger = task.get_logger()

# Retrieve the registered model
model = InputModel(name="llama-3.3-70b-instruct", project="production-llm")
logger.report_text(f"Using model: {model.name}  ID: {model.id}")

prompts = [
    "Explain transformer attention in one paragraph.",
    "What is tensor parallelism?",
    "How does vLLM's continuous batching work?",
]

results = []
for i, prompt in enumerate(prompts, 1):
    payload = {
        "model": "llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        f"{LITELLM_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
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
print("Done. Check ClearML UI → production-llm → Tasks.")
```

```bash
python3 inference_task.py
```

### 🔍 VERIFY AFTER COMPLETION

Open ClearML web UI → `production-llm` → Tasks → `llama-3.3-70b-inference-demo`.
Confirm you see: prompts and responses in the log, token usage graphs, and the inference_results artifact.

---

## Step 8 — Monitor GPU health (your ClusterAdmin responsibility)

```bash
# Quick GPU status check
kubectl exec deployment/vllm-llama-70b -n $MY_NAMESPACE -- nvidia-smi

# What to look for:
# - Memory: ~90% used per GPU (weights + KV cache)
# - Utilisation: spikes during inference, 0% when idle
# - Temperature: under 80°C (H200 SXM throttles at ~83°C)
# - Power: up to 700W per H200 SXM under load

# If DCGM is installed on the cluster:
kubectl exec deployment/vllm-llama-70b -n $MY_NAMESPACE -- dcgmi diag -r 1
# Level 1 = quick health check
# Level 3 = comprehensive (takes ~2 minutes, stresses GPUs)

# Real-time metrics stream:
kubectl exec deployment/vllm-llama-70b -n $MY_NAMESPACE -- \
  dcgmi dmon -e 100,101,102,203,204 -d 2000
# 100 = SM utilisation (%)
# 101 = memory utilisation (%)
# 102 = memory used (MB) — should show ~126000 per GPU for 70B bf16
# 203 = GPU temperature (°C)
# 204 = power usage (W)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Pod stuck in `Pending` | Not enough free GPUs | `kubectl describe pod` to confirm. Wait for GPUs or reduce `tensor-parallel-size` |
| Pod `CrashLoopBackOff` | OOM or bad config | Check `kubectl logs`. If OOM: reduce `--max-model-len` to 16384 or use more GPUs |
| vLLM starts but `Killed` after 2 min | Liveness probe too aggressive | Increase `livenessProbe.initialDelaySeconds` to 600 |
| NCCL shared memory error | Missing `/dev/shm` mount | Add the `shm` emptyDir volume (see Step 5 YAML) |
| LiteLLM returns 502 | vLLM not ready yet | Wait 5 minutes. Check vLLM pod logs for "Application startup complete" |
| "Context length exceeded" | Prompt + max_tokens > `--max-model-len` | Increase `--max-model-len` (needs more GPU memory) or shorten prompt |
| Download job fails at 50% | Network timeout | Delete job, re-apply. `huggingface_hub` resumes automatically |
| `nvidia-smi` shows 0% utilisation | No requests coming in | Normal when idle. Send a test request to confirm |
| Image pull fails | Cluster can't reach Docker Hub | Use internal registry path from Step 0.7b. Verify with: `kubectl describe pod` → look for "Failed to pull image" |
| `ImagePullBackOff` | Wrong registry URL or missing pull secret | Check `MY_CONTAINER_REGISTRY` and `MY_IMAGE_PULL_SECRET`. Run: `kubectl get events -n $MY_NAMESPACE` |
| `ErrImagePull` with auth error | Pull secret not in your namespace | Copy secret: `kubectl get secret NAME -n SOURCE_NS -o yaml \| sed 's/namespace:.*/namespace: YOUR_NS/' \| kubectl apply -f -` |

---

## Cleanup (when you're done)

```bash
# Remove everything in reverse order
kubectl delete deployment litellm-proxy -n $MY_NAMESPACE
kubectl delete svc litellm-proxy -n $MY_NAMESPACE
kubectl delete configmap litellm-config -n $MY_NAMESPACE
kubectl delete deployment vllm-llama-70b -n $MY_NAMESPACE
kubectl delete svc vllm-llama-70b -n $MY_NAMESPACE
kubectl delete job llama-download -n $MY_NAMESPACE
kubectl delete pvc llama-model-storage -n $MY_NAMESPACE    # WARNING: deletes 140GB of downloaded weights
kubectl delete secret hf-credentials -n $MY_NAMESPACE
```

---

## Key differences from the local lab

| Topic | Local lab | Production |
| ----- | --------- | ---------- |
| Model weights storage | Local filesystem | Kubernetes PersistentVolume |
| Inference engine | Custom serve.py (stdlib) | vLLM (PagedAttention, continuous batching) |
| GPU type | 1× GTX 1070 Ti (8 GB) | 2-8× H200 (141 GB HBM3e each) |
| GPU memory total | 8 GB | 282–1,128 GB |
| Context window | 128 chars | 32K–131K tokens |
| Concurrency | 1 request at a time | Hundreds (continuous batching) |
| LiteLLM deployment | Docker container | Kubernetes Deployment |
| Model download | Phase 1 script | Kubernetes Job |
| API key security | Hardcoded demo key | Kubernetes Secret |
| Networking | localhost | Kubernetes Service DNS |
| Multi-GPU comms | N/A (single GPU) | NCCL over NVLink (/dev/shm required) |
| Monitoring | nvidia-smi manually | DCGM + Prometheus + Grafana |

The API your clients call is identical:
`POST /v1/chat/completions` with `Authorization: Bearer <key>`.
That is the whole point of the LiteLLM abstraction layer.

---

## Next steps

1. **Add an Ingress** — DNS name + TLS in front of LiteLLM
2. **Add virtual keys** — per-user rate limiting and spend tracking in LiteLLM
3. **Enable vLLM metrics** — `--enable-metrics` flag, scrape with Prometheus
4. **ClearML Pipeline** — auto-redeploy when a new model version is registered
5. **Scale up** — try `--tensor-parallel-size=4` for full 131K context and higher concurrency
6. **Point VS Code at it** — Configure Continue.dev or Copilot alternatives to use your LiteLLM endpoint
