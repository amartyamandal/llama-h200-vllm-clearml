# Deploying a 70B LLM on Multi-GPU Kubernetes with vLLM + ClearML

Production deployment guide for multi-tenant ClearML Enterprise clusters
with corporate proxy, enterprise NFS storage, and GPU scheduling managed
by ClearML queues.

**Deployment flow:**
1. Download model weights → K8s Job (with proxy env vars)
2. Register model in ClearML → K8s Job (with ClearML SDK)
3. Deploy inference server → **ClearML vLLM Model Deployment App** (via ClearML UI)
4. API gateway → ClearML AI Application Gateway (automatic)

> **No ClearML CLI required on your workstation.** All ClearML interactions
> happen either through the Web UI or through K8s Jobs that run the ClearML
> SDK inside the cluster.

---

## Cluster profile

| Property | Your cluster |
|----------|-------------|
| GPU type | H200 (141 GB HBM3e) or H100 (80 GB HBM3) |
| Cluster access | ClusterAdmin on Kubernetes |
| ClearML access | Tenant admin (Web UI) |
| MIG status | Enabled on some nodes — must be avoided |
| Internet egress | Via corporate HTTP/HTTPS proxy (port 3128) |
| Image pulling | Docker Hub / NGC via managed pull secrets |
| Storage | Enterprise NFS (Powerscale) with existing shared PVC (~1 TiB) |
| ClearML CLI | **Not installed** on bastion or control plane |

---

## Step 0 — Discover your cluster

### 0.1 — Confirm access

```bash
kubectl cluster-info
kubectl auth can-i '*' '*' --all-namespaces
# Must show: "yes"
```

### 0.2 — Discover GPU nodes

```bash
kubectl get nodes -o custom-columns=\
"NAME:.metadata.name,\
GPUs:.status.allocatable.nvidia\.com/gpu,\
MEMORY:.status.allocatable.memory"
```

```
MY_GPU_NODES=
MY_GPUS_PER_NODE=
```

### 0.3 — Confirm GPU type

```bash
kubectl debug node/<GPU_NODE_NAME> -it --image=nvidia/cuda:12.4.0-base-ubuntu22.04 -- nvidia-smi
```

- **"NVIDIA H200" + ~141 GB** → `TENSOR_PARALLEL_SIZE=2`
- **"NVIDIA H100" + ~80 GB** → `TENSOR_PARALLEL_SIZE=4`

```
MY_GPU_TYPE=
MY_GPU_MEMORY_GB=
MY_TENSOR_PARALLEL_SIZE=
```

### 0.4 — Identify MIG nodes

```bash
for node in $(kubectl get nodes -o name | sed 's|node/||'); do
  echo "--- $node ---"
  kubectl debug node/$node -it --image=nvidia/cuda:12.4.0-base-ubuntu22.04 -- nvidia-smi mig -lgi 2>/dev/null || echo "No MIG"
done
```

**Note:** A MIG node showing `nvidia.com/gpu: 0` is normal — MIG replaces
`nvidia.com/gpu` with `nvidia.com/mig-*` resources. This is not a failure.

```
MY_MIG_NODES=               # AVOID these
MY_FULL_GPU_NODES=          # TARGET these
```

### 0.5 — Check free GPUs on non-MIG nodes

```bash
for node in <MY_FULL_GPU_NODES>; do
  echo "--- $node ---"
  kubectl describe node $node | grep -A5 "Allocated resources" | grep nvidia
done
```

```
MY_FREE_GPUS=               # must be >= MY_TENSOR_PARALLEL_SIZE
```

### 0.6 — Node labels and taints

```bash
kubectl describe nodes | grep -A3 "Taints:"
kubectl get nodes --show-labels | grep -i "gpu\|nvidia"
```

If non-MIG nodes lack a distinguishing label:

```bash
kubectl label node <FULL_GPU_NODE_NAME> gpu-mode=full
```

```
MY_GPU_TAINT_KEY=
MY_GPU_NODE_LABEL=
```

### 0.7 — Discover the corporate proxy

```bash
kubectl get pods --all-namespaces | grep -i "clearml\|agent"
kubectl describe pod <CLEARML_AGENT_POD> -n <NAMESPACE> | grep -i proxy
```

```
MY_HTTP_PROXY=
MY_HTTPS_PROXY=
MY_NO_PROXY=
```

Verify:

```bash
kubectl run test-proxy --rm -it --restart=Never --image=python:3.11-slim \
  --env="http_proxy=$MY_HTTP_PROXY" \
  --env="https_proxy=$MY_HTTPS_PROXY" \
  --env="no_proxy=$MY_NO_PROXY" \
  -- bash -c "pip install -q requests && python3 -c \"import requests; print('PROXY OK:', requests.get('https://huggingface.co').status_code)\""
```

### 0.8 — Discover the existing shared PVC

```bash
kubectl get pvc --all-namespaces | grep -i "shared\|workspace"
kubectl describe pvc <PVC_NAME> -n <NAMESPACE> | grep -E "Capacity|StorageClass|VolumeName"
```

```
MY_SHARED_PVC_NAME=
MY_SHARED_PVC_NAMESPACE=
MY_SHARED_MOUNT_PATH=
```

### 0.9 — Image pull secrets

```bash
kubectl get secrets --all-namespaces | grep -i "pull\|registry\|docker"
```

```
MY_PULL_SECRET_NAME=
MY_PULL_SECRET_NAMESPACE=
```

### 0.10 — ClearML tenant access

Open the ClearML Web UI. Go to `Settings → Workspace → API Credentials → Create new credentials`.

```
MY_CLEARML_API=
MY_CLEARML_WEB=
MY_CLEARML_FILES=
MY_CLEARML_ACCESS_KEY=
MY_CLEARML_SECRET_KEY=
```

### 0.11 — Check if vLLM Model Deployment App is available

ClearML Web UI → `Applications`. Look for **"vLLM Model Deployment"**.

```
CLEARML_VLLM_APP_AVAILABLE=   # "yes" or "no"
```

### 0.12 — Check existing queues

ClearML Web UI → `Orchestration → Queues`.

```
MY_EXISTING_GPU_QUEUES=
MY_TARGET_QUEUE=
```

### 0.13 — Summary checklist

```
[ ] MY_GPU_TYPE                 = ___________
[ ] MY_TENSOR_PARALLEL_SIZE     = ___________
[ ] MY_MIG_NODES                = ___________
[ ] MY_FULL_GPU_NODES           = ___________
[ ] MY_FREE_GPUS                = ___________
[ ] MY_GPU_NODE_LABEL           = ___________
[ ] MY_HTTP_PROXY               = ___________
[ ] MY_HTTPS_PROXY              = ___________
[ ] MY_SHARED_PVC_NAME          = ___________
[ ] MY_SHARED_MOUNT_PATH        = ___________
[ ] MY_PULL_SECRET_NAME         = ___________
[ ] MY_CLEARML_API              = ___________
[ ] MY_CLEARML_ACCESS_KEY       = ___________
[ ] CLEARML_VLLM_APP_AVAILABLE  = ___________
[ ] MY_TARGET_QUEUE             = ___________
```

---

## Step 1 — Determine your working namespace

```bash
kubectl get pods --all-namespaces | grep -i "clearml.*agent\|k8sglue"
```

The namespace where the ClearML agent runs is where workloads get spawned. Use it.

```
MY_NAMESPACE=
```

Copy pull secret if needed:

```bash
kubectl get secret $MY_PULL_SECRET_NAME -n $MY_NAMESPACE 2>/dev/null \
  && echo "Already exists" \
  || kubectl get secret $MY_PULL_SECRET_NAME -n $MY_PULL_SECRET_NAMESPACE -o yaml \
     | grep -v "namespace:\|uid:\|resourceVersion:\|creationTimestamp:" \
     | kubectl apply -n $MY_NAMESPACE -f -
```

---

## Step 2 — Download model weights via proxy

### 🔍 VERIFY BEFORE PROCEEDING

```bash
kubectl get pvc $MY_SHARED_PVC_NAME -n $MY_NAMESPACE -o jsonpath='{.status.phase}'
# Must show: "Bound"

# Check if model already downloaded:
kubectl run check-models --rm -it --restart=Never --image=busybox \
  --overrides="{\"spec\":{\"containers\":[{\"name\":\"c\",\"image\":\"busybox\",\"command\":[\"sh\",\"-c\",\"ls -la /storage/ 2>/dev/null\"],\"volumeMounts\":[{\"name\":\"ms\",\"mountPath\":\"/storage\"}]}],\"volumes\":[{\"name\":\"ms\",\"persistentVolumeClaim\":{\"claimName\":\"$MY_SHARED_PVC_NAME\"}}]}}" \
  -n $MY_NAMESPACE
# If llama directory with safetensors files exists: SKIP THIS STEP
```

### Execute

```bash
# HF token secret (skip for ungated models)
kubectl create secret generic hf-credentials \
  --from-literal=token=<YOUR_HF_TOKEN> \
  -n $MY_NAMESPACE
```

Create `download-job.yaml` — **replace ALL `<>` placeholders:**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download
  namespace: <MY_NAMESPACE>
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>
      containers:
        - name: downloader
          image: python:3.11-slim
          env:
            - name: http_proxy
              value: "<MY_HTTP_PROXY>"
            - name: https_proxy
              value: "<MY_HTTPS_PROXY>"
            - name: no_proxy
              value: "<MY_NO_PROXY>"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
                  optional: true
          command:
            - /bin/sh
            - -c
            - |
              pip install -q huggingface_hub && \
              python3 -c "
              import os
              from huggingface_hub import snapshot_download
              REPO = 'meta-llama/Llama-3.3-70B-Instruct'
              TOKEN = os.environ.get('HF_TOKEN')
              # Ungated alternative (no token needed):
              # REPO = 'Qwen/Qwen2.5-72B-Instruct'
              # TOKEN = None
              LOCAL_DIR = '<MY_SHARED_MOUNT_PATH>/llama-3.3-70b'
              print(f'Downloading {REPO} to {LOCAL_DIR} (~130 GB)')
              snapshot_download(repo_id=REPO, local_dir=LOCAL_DIR, token=TOKEN,
                  ignore_patterns=['*.gguf', '*.bin'])
              print('Download complete')
              "
          volumeMounts:
            - name: model-storage
              mountPath: <MY_SHARED_MOUNT_PATH>
          resources:
            requests:
              memory: "8Gi"
              cpu: "2"
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: <MY_SHARED_PVC_NAME>
```

```bash
grep "<" download-job.yaml && echo "ERROR: Replace placeholders!" || echo "OK"
kubectl apply -f download-job.yaml
kubectl logs -f job/model-download -n $MY_NAMESPACE
# Wait for "Download complete" (30-60 min)
# If it fails mid-way: delete job, re-apply. HF hub resumes automatically.
```

### 🔍 VERIFY AFTER

```bash
kubectl get job model-download -n $MY_NAMESPACE -o jsonpath='{.status.succeeded}'
# Must show: "1"
```

```
MY_MODEL_PATH=              # e.g. "/shared-workspace/llama-3.3-70b"
MY_MODEL_NAME=              # e.g. "llama-3.3-70b-instruct"
```

---

## Step 3 — Register model in ClearML (via K8s Job)

ClearML CLI is not installed anywhere accessible. Run registration as a K8s Job.

### Execute

Create ClearML credentials secret:

```bash
kubectl create secret generic clearml-credentials \
  --from-literal=api_server=<MY_CLEARML_API> \
  --from-literal=web_server=<MY_CLEARML_WEB> \
  --from-literal=files_server=<MY_CLEARML_FILES> \
  --from-literal=access_key=<MY_CLEARML_ACCESS_KEY> \
  --from-literal=secret_key=<MY_CLEARML_SECRET_KEY> \
  -n $MY_NAMESPACE
```

Create `register-model-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: register-model
  namespace: <MY_NAMESPACE>
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>
      containers:
        - name: register
          image: python:3.11-slim
          env:
            - name: http_proxy
              value: "<MY_HTTP_PROXY>"
            - name: https_proxy
              value: "<MY_HTTPS_PROXY>"
            - name: no_proxy
              value: "<MY_NO_PROXY>,<CLEARML_API_HOSTNAME>"
            - name: CLEARML_API_HOST
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: api_server }
            - name: CLEARML_WEB_HOST
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: web_server }
            - name: CLEARML_FILES_HOST
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: files_server }
            - name: CLEARML_API_ACCESS_KEY
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: access_key }
            - name: CLEARML_API_SECRET_KEY
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: secret_key }
          command:
            - /bin/sh
            - -c
            - |
              pip install -q clearml && \
              python3 -c "
              from clearml import Task, OutputModel
              task = Task.init(project_name='production-llm',
                  task_name='register-llama-3.3-70b',
                  task_type=Task.TaskTypes.data_processing,
                  reuse_last_task_id=False)
              out_model = OutputModel(task=task,
                  name='<MY_MODEL_NAME>',
                  tags=['70b', 'instruct', 'vllm', 'production'],
                  framework='PyTorch')
              out_model.update_design(config_dict={
                  'cluster_path': '<MY_MODEL_PATH>',
                  'format': 'safetensors',
                  'precision': 'bfloat16',
                  'inference_engine': 'vLLM',
                  'gpu_type': '<MY_GPU_TYPE>',
                  'min_gpus': <MY_TENSOR_PARALLEL_SIZE>,
              })
              task.close()
              print(f'Model registered: {out_model.id}')
              "
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
```

**Important:** Add your ClearML API hostname to `no_proxy` so the SDK connects
directly instead of going through the corporate proxy.

```bash
grep "<" register-model-job.yaml && echo "ERROR!" || echo "OK"
kubectl apply -f register-model-job.yaml
kubectl logs -f job/register-model -n $MY_NAMESPACE
```

### 🔍 VERIFY AFTER

ClearML Web UI → Models → search for your model name.

---

## Step 4 — Deploy inference via ClearML

### Path A — ClearML vLLM Model Deployment App (preferred)

If `CLEARML_VLLM_APP_AVAILABLE = yes`:

#### 4A.1 — Ensure a GPU queue exists

ClearML Web UI → `Orchestration → Queues`. You need a queue with:
- `>= MY_TENSOR_PARALLEL_SIZE` full GPUs (not MIG slices)
- `/dev/shm` shared memory configured
- `nodeSelector` targeting non-MIG nodes
- Shared PVC mounted

If no suitable queue exists, ask the platform team:

> "I need a ClearML queue with 2 full H200 GPUs (not MIG), /dev/shm shared
> memory, and the shared workspace PVC mounted at /shared-workspace."

#### 4A.2 — Launch vLLM Model Deployment

ClearML Web UI → `Applications → vLLM Model Deployment → New Instance`:

1. **Queue:** Select your GPU queue
2. **Model:** `<MY_MODEL_PATH>` (cluster path) or `meta-llama/Llama-3.3-70B-Instruct` (HF ID)
3. **CLI arguments:**
   ```
   --tensor-parallel-size <MY_TENSOR_PARALLEL_SIZE>
   --dtype bfloat16
   --max-model-len 32768
   --gpu-memory-utilization 0.90
   --enable-chunked-prefill
   ```
4. **Trust Remote Code:** Enable
5. **AI Gateway Route:** Select if available
6. **Launch**

#### 4A.3 — Wait for startup

Watch the Console log for: `INFO: Application startup complete.` (3-5 minutes)

#### 4A.4 — Get endpoint

The app dashboard shows your **Endpoint URL** and a **curl command**.

### 🔍 VERIFY

```bash
curl -s -X POST <ENDPOINT_URL>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <CLEARML_TOKEN>" \
  -d '{"model":"<MY_MODEL_NAME>","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

**If this returns a coherent response: the full stack is working.**

---

### Path B — Standalone K8s Deployment (fallback)

Use only if vLLM App is NOT available.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  namespace: <MY_NAMESPACE>
spec:
  replicas: 1
  selector:
    matchLabels: { app: vllm-inference }
  template:
    metadata:
      labels: { app: vllm-inference }
    spec:
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>
      nodeSelector:
        gpu-mode: full
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - --model=<MY_MODEL_PATH>
            - --served-model-name=<MY_MODEL_NAME>
            - --tensor-parallel-size=<MY_TENSOR_PARALLEL_SIZE>
            - --dtype=bfloat16
            - --max-model-len=32768
            - --gpu-memory-utilization=0.90
            - --port=8000
            - --host=0.0.0.0
            - --trust-remote-code
            - --enable-chunked-prefill
          ports: [{ containerPort: 8000 }]
          resources:
            limits: { "nvidia.com/gpu": "<MY_TENSOR_PARALLEL_SIZE>" }
            requests: { "nvidia.com/gpu": "<MY_TENSOR_PARALLEL_SIZE>", memory: "64Gi", cpu: "8" }
          volumeMounts:
            - { name: model-storage, mountPath: "<MY_SHARED_MOUNT_PATH>" }
            - { name: shm, mountPath: /dev/shm }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 180
            periodSeconds: 10
            failureThreshold: 30
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 300
            periodSeconds: 30
            failureThreshold: 5
      volumes:
        - name: model-storage
          persistentVolumeClaim: { claimName: "<MY_SHARED_PVC_NAME>" }
        - name: shm
          emptyDir: { medium: Memory, sizeLimit: "16Gi" }
      tolerations:
        - { key: "<MY_GPU_TAINT_KEY>", operator: Exists, effect: NoSchedule }
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-inference
  namespace: <MY_NAMESPACE>
spec:
  selector: { app: vllm-inference }
  ports: [{ port: 8000, targetPort: 8000 }]
  type: ClusterIP
```

For LiteLLM proxy with Path B, see Appendix A at the end.

---

## Step 5 — Monitor GPU health

```bash
# Find the vLLM pod (Path A: ClearML-created; Path B: your deployment)
kubectl get pods -n $MY_NAMESPACE | grep -i "vllm\|clearml-id"

kubectl exec <VLLM_POD> -n $MY_NAMESPACE -- nvidia-smi
kubectl exec <VLLM_POD> -n $MY_NAMESPACE -- dcgmi diag -r 1
kubectl exec <VLLM_POD> -n $MY_NAMESPACE -- dcgmi dmon -e 100,101,102,203,204 -d 2000
kubectl logs <VLLM_POD> -n $MY_NAMESPACE | grep -i "nccl\|nvlink"
```

---

## Step 6 — Log inference run to ClearML (via K8s Job)

Create `inference-test-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-test
  namespace: <MY_NAMESPACE>
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>
      containers:
        - name: test
          image: python:3.11-slim
          env:
            - name: http_proxy
              value: "<MY_HTTP_PROXY>"
            - name: https_proxy
              value: "<MY_HTTPS_PROXY>"
            - name: no_proxy
              value: "<MY_NO_PROXY>,<CLEARML_API_HOSTNAME>,<VLLM_ENDPOINT_HOSTNAME>"
            - name: CLEARML_API_HOST
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: api_server }
            - name: CLEARML_WEB_HOST
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: web_server }
            - name: CLEARML_FILES_HOST
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: files_server }
            - name: CLEARML_API_ACCESS_KEY
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: access_key }
            - name: CLEARML_API_SECRET_KEY
              valueFrom:
                secretKeyRef: { name: clearml-credentials, key: secret_key }
          command:
            - /bin/sh
            - -c
            - |
              pip install -q clearml requests && \
              python3 -c "
              import requests
              from clearml import Task
              VLLM_URL = '<ENDPOINT_URL>'
              AUTH = 'Bearer <TOKEN>'
              MODEL = '<MY_MODEL_NAME>'
              task = Task.init(project_name='production-llm',
                  task_name=f'{MODEL}-inference-test',
                  task_type=Task.TaskTypes.inference,
                  reuse_last_task_id=False)
              logger = task.get_logger()
              prompts = ['Explain transformer attention.', 'What is tensor parallelism?',
                         'How does continuous batching work?']
              for i, p in enumerate(prompts, 1):
                  r = requests.post(f'{VLLM_URL}/v1/chat/completions',
                      json={'model': MODEL, 'messages': [{'role':'user','content':p}], 'max_tokens':200},
                      headers={'Content-Type':'application/json','Authorization':AUTH}, timeout=120)
                  d = r.json()
                  text = d['choices'][0]['message']['content']
                  u = d.get('usage',{})
                  logger.report_text(f'Q{i}: {p}\nA: {text}')
                  logger.report_scalar('Tokens','prompt',u.get('prompt_tokens',0),i)
                  logger.report_scalar('Tokens','completion',u.get('completion_tokens',0),i)
                  print(f'Q{i} done. {u}')
              task.close()
              print('Done. Check ClearML UI.')
              "
          resources:
            requests: { memory: "2Gi", cpu: "1" }
```

```bash
grep "<" inference-test-job.yaml && echo "ERROR!" || echo "OK"
kubectl apply -f inference-test-job.yaml
kubectl logs -f job/inference-test -n $MY_NAMESPACE
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Download: `pip install` fails | Proxy env vars missing | Add `http_proxy`/`https_proxy` |
| Download: HF 401/403 | Bad token or license not accepted | Re-check token; accept license at HF |
| Register: connection refused | ClearML API not in `no_proxy` | Add ClearML hostname to `no_proxy` |
| vLLM `Pending` | Not enough GPUs or label mismatch | `kubectl describe pod` → Events |
| vLLM `CrashLoopBackOff` | OOM | Reduce `--max-model-len` to `16384` |
| vLLM killed after 2 min | Liveness probe too early | Increase `initialDelaySeconds` to `600` |
| NCCL shared memory error | Missing `/dev/shm` | Add emptyDir volume |
| Pod on MIG node | nodeSelector wrong | Fix label |
| ClearML App: no GPU queues | Queue not configured | Ask platform team |
| MIG node shows 0 GPUs | Normal MIG behavior | Not a failure |
| Inference test: connection refused | Wrong endpoint or vLLM not ready | Check URL; wait 5 min |

---

## Cleanup

```bash
kubectl delete job inference-test register-model model-download -n $MY_NAMESPACE
kubectl delete secret clearml-credentials hf-credentials -n $MY_NAMESPACE
# Path A: stop vLLM app from ClearML UI
# Path B: kubectl delete deployment vllm-inference svc vllm-inference -n $MY_NAMESPACE
```

---

## Reference: local lab → production

| Topic | Local lab | Production |
|-------|-----------|------------|
| Model storage | Local filesystem | K8s PVC on NFS |
| Model download | Python script (direct) | K8s Job via proxy |
| Model registration | clearml-init + local script | K8s Job with ClearML SDK |
| Inference engine | Custom serve.py | vLLM (ClearML App or K8s Deployment) |
| API gateway | LiteLLM (Docker) | ClearML AI App Gateway |
| GPU | 1x consumer GPU | 2-8x H200 (141 GB each) |
| Multi-GPU | N/A | NCCL over NVLink (/dev/shm) |
| GPU targeting | N/A | nodeSelector to avoid MIG |
| Monitoring | nvidia-smi | DCGM + ClearML dashboard |

---

## Appendix A — LiteLLM proxy (Path B only)

```yaml
# litellm-config.yaml
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
  master_key: "<GENERATE_A_KEY>"
```

```bash
kubectl create configmap litellm-config --from-file=config.yaml=litellm-config.yaml -n $MY_NAMESPACE
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-proxy
  namespace: <MY_NAMESPACE>
spec:
  replicas: 1
  selector:
    matchLabels: { app: litellm-proxy }
  template:
    metadata:
      labels: { app: litellm-proxy }
    spec:
      imagePullSecrets:
        - name: <MY_PULL_SECRET_NAME>
      containers:
        - name: litellm
          image: ghcr.io/berriai/litellm:main-stable
          args: ["--config=/app/config.yaml", "--port=4000"]
          ports: [{ containerPort: 4000 }]
          resources:
            requests: { memory: "512Mi", cpu: "1" }
          volumeMounts:
            - { name: config, mountPath: /app/config.yaml, subPath: config.yaml }
          readinessProbe:
            httpGet: { path: /health, port: 4000 }
            initialDelaySeconds: 10
            periodSeconds: 5
      volumes:
        - name: config
          configMap: { name: litellm-config }
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-proxy
  namespace: <MY_NAMESPACE>
spec:
  selector: { app: litellm-proxy }
  ports: [{ port: 4000, targetPort: 4000 }]
  type: NodePort
```

---

## Next steps

1. **vLLM metrics** — `--enable-metrics`, scrape with Prometheus
2. **Scale up** — higher `--tensor-parallel-size` for 131K context
3. **Dev tools** — point VS Code / Continue.dev at the endpoint
4. **Additional models** — deploy Qwen, Mistral alongside Llama
