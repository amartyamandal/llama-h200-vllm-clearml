# Llama 3.3 70B on H200 GPUs — vLLM + ClearML

Production deployment of **Llama 3.3 70B Instruct** on Kubernetes H200 GPU nodes using vLLM as the inference engine, LiteLLM as the OpenAI-compatible proxy, and ClearML for model tracking.

The result is a fully functional `POST /v1/chat/completions` endpoint backed by H200 GPUs — drop-in compatible with any OpenAI client.

---

## What gets deployed

```
Client
  │  POST /v1/chat/completions
  ▼
LiteLLM Proxy  (K8s Deployment, port 4000)
  │  routes requests + handles auth
  ▼
vLLM Server    (K8s Deployment, port 8000)
  │  PagedAttention + continuous batching
  ▼
H200 GPUs      (2–4 GPUs via tensor parallelism, NCCL/NVLink)
  │
  ├── Llama 3.3 70B weights  (PersistentVolumeClaim, 300 Gi)
  └── ClearML                (model registry + inference logging)
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| Kubernetes cluster | ClusterAdmin access |
| GPU nodes | H200 (141 GB) or H100 (80 GB) — see GPU notes below |
| ClearML | Already installed on the cluster |
| Hugging Face account | License accepted at [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| Local tools | `kubectl`, `python3`, `envsubst` (`gettext` package) |
| Container images | Public internet: pulled automatically. Air-gapped: see [Step 0.7a/0.7b](00-cluster-discovery/DISCOVERY.md) |

**GPU sizing:**

| GPU  | VRAM   | Minimum `tensor-parallel-size` | Recommended |
|------|--------|-------------------------------|-------------|
| H200 | 141 GB | 2                             | 4           |
| H100 | 80 GB  | 2 (tight)                     | 4           |

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd <repo-name>

# 2. Create your config
cp config.env.example config.env
$EDITOR config.env          # fill in every value

# 3. Run cluster discovery (read 00-cluster-discovery/DISCOVERY.md first)
#    Verify all config.env values match your actual cluster

# 4. Deploy everything
source config.env
bash scripts/deploy-all.sh

# 5. Verify the stack
bash scripts/verify-stack.sh

# 6. Log an inference run to ClearML
export LITELLM_URL="http://<NODE_IP>:<NODE_PORT>"   # printed by deploy-all.sh
python3 07-clearml-logging/inference_task.py
```

---

## Repository layout

```
.
├── README.md                          ← you are here
├── H200_LLAMA_PRODUCTION.md           ← full step-by-step guide with verify gates
├── config.env.example                 ← all variables in one place (copy → config.env)
│
├── 00-cluster-discovery/
│   └── DISCOVERY.md                   ← Step 0: gather cluster facts before touching anything
│
├── 01-secrets/
│   └── create-hf-secret.sh            ← Step 1: HF token → K8s Secret
│
├── 02-storage/
│   └── model-storage-pvc.yaml         ← Step 2: 300 Gi PVC for model weights
│
├── 03-download/
│   └── download-job.yaml              ← Step 3: K8s Job — downloads ~140 GB from HF Hub
│
├── 04-clearml/
│   └── register_model.py              ← Step 4: registers model in ClearML registry
│
├── 05-vllm/
│   └── vllm-deployment.yaml           ← Step 5: vLLM Deployment + ClusterIP Service
│
├── 06-litellm/
│   ├── litellm-config.yaml            ← Step 6a: LiteLLM config (becomes a ConfigMap)
│   └── litellm-deployment.yaml        ← Step 6b: LiteLLM Deployment + NodePort Service
│
├── 07-clearml-logging/
│   └── inference_task.py              ← Step 7: send prompts, log to ClearML
│
└── scripts/
    ├── deploy-all.sh                  ← end-to-end deploy (runs steps 1–6)
    ├── verify-stack.sh                ← health checks across the whole stack
    └── cleanup.sh                     ← tear everything down safely
```

---

## Step-by-step (manual)

Follow the numbered directories in order. Each has its own YAML or script.
All values are driven by `config.env` — never edit values into YAML files directly.

| Step | Directory | What happens |
|------|-----------|--------------|
| 0 | `00-cluster-discovery/` | Collect cluster facts, fill `config.env` |
| 0.7a/b | `00-cluster-discovery/` | **(Air-gapped only)** Find internal registry, pre-load images |
| 1 | `01-secrets/` | Create `hf-credentials` K8s Secret |
| 2 | `02-storage/` | Create 300 Gi PVC |
| 3 | `03-download/` | K8s Job downloads ~140 GB of weights |
| 4 | `04-clearml/` | Register model in ClearML |
| 5 | `05-vllm/` | Deploy vLLM server (2–4 H200 GPUs) |
| 6 | `06-litellm/` | Deploy LiteLLM proxy with API key auth |
| 7 | `07-clearml-logging/` | Run inference, log metrics to ClearML |

For detailed verify gates at each step, see [H200_LLAMA_PRODUCTION.md](H200_LLAMA_PRODUCTION.md).

---

## Applying YAML files manually

All YAML files use `${VARIABLE}` placeholders substituted from `config.env`.

```bash
source config.env

# Example — apply the PVC:
envsubst < 02-storage/model-storage-pvc.yaml | kubectl apply -f -

# Example — apply vLLM deployment:
envsubst < 05-vllm/vllm-deployment.yaml | kubectl apply -f -
```

`deploy-all.sh` does this automatically for every file.

---

## Smoke test

After deployment, send a request from your local machine:

```bash
source config.env

NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc litellm-proxy -n $MY_NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')

curl -s -X POST http://$NODE_IP:$NODE_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
    "max_tokens": 100
  }'
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Pod stuck in `Pending` | Not enough free GPUs | `kubectl describe pod` — wait for GPUs or reduce `tensor-parallel-size` |
| `CrashLoopBackOff` | OOM or bad arg | `kubectl logs` — if OOM, reduce `--max-model-len` to 16384 or use more GPUs |
| vLLM `Killed` after 2 min | Liveness probe fires too early | Increase `livenessProbe.initialDelaySeconds` to 600 |
| NCCL shared memory error | `/dev/shm` not mounted | Add the `shm` emptyDir volume (already in `05-vllm/vllm-deployment.yaml`) |
| LiteLLM returns 502 | vLLM not ready | Wait 5 min, check vLLM logs for "Application startup complete" |
| Download job fails mid-way | Network timeout | Delete job, re-apply — `huggingface_hub` resumes automatically |
| `ImagePullBackOff` | Wrong registry URL or missing pull secret | Check `MY_VLLM_IMAGE` / `MY_LITELLM_IMAGE` in `config.env`. Run: `kubectl get events -n $MY_NAMESPACE` |
| `ErrImagePull` with auth error | Pull secret not copied to your namespace | Copy secret: `kubectl get secret NAME -n SOURCE -o yaml \| sed 's/namespace:.*/namespace: YOUR_NS/' \| kubectl apply -f -` |
| Image pull fails on public cluster | Cluster can't reach Docker Hub | Follow Step 0.7a/0.7b to pre-load images via internal registry |

---

## Cleanup

```bash
source config.env
bash scripts/cleanup.sh
# WARNING: this deletes the 300 Gi PVC and all 140 GB of weights
```

---

## Next steps

1. **Add Ingress** — DNS name + TLS termination in front of LiteLLM
2. **Virtual keys** — per-user rate limiting and spend tracking in LiteLLM
3. **vLLM metrics** — add `--enable-metrics` flag, scrape with Prometheus
4. **ClearML Pipeline** — auto-redeploy when a new model version is registered
5. **Scale up** — try `--tensor-parallel-size=4` for full 131K context
6. **Point your IDE at it** — configure Continue.dev or Copilot alternatives to use your LiteLLM endpoint

---

> For the complete step-by-step guide with every `🔍 VERIFY BEFORE PROCEEDING` gate, see [H200_LLAMA_PRODUCTION.md](H200_LLAMA_PRODUCTION.md).
