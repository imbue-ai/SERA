"""Modal app deploying vLLM as a persistent web endpoint for SWE-bench evaluation.

Deploy base model:
    uv run modal deploy serve_vllm.py

Deploy fine-tuned model from sera-models volume:
    VLLM_MODEL=/models/flask-specialist/converted VLLM_SERVED_NAME=flask-specialist uv run modal deploy serve_vllm.py

URL:     https://imbue--swebench-vllm-serve.modal.run
Models:  curl https://imbue--swebench-vllm-serve.modal.run/v1/models
"""

import os
import subprocess

import modal

# --- Configurable via env vars at deploy time ---
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
MODEL_REVISION = os.environ.get("VLLM_MODEL_REVISION", "main")
SERVED_NAME = os.environ.get("VLLM_SERVED_NAME", MODEL)
APP_NAME = os.environ.get("VLLM_APP_NAME", "swebench-vllm")
IS_LOCAL_MODEL = MODEL.startswith("/")

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("vllm==0.17.1", "huggingface-hub")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
sera_models_vol = modal.Volume.from_name("sera-models")

volumes = {
    "/root/.cache/vllm": vllm_cache_vol,
}
if IS_LOCAL_MODEL:
    volumes["/models"] = sera_models_vol
else:
    volumes["/root/.cache/huggingface"] = hf_cache_vol


@app.function(
    image=vllm_image,
    gpu="H100:1",
    volumes=volumes,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=900,  # 15 minutes
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", SERVED_NAME,
        "--max-model-len", "131072",
        "--tensor-parallel-size", "1",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--enforce-eager",
    ]
    if not IS_LOCAL_MODEL:
        cmd.extend(["--revision", MODEL_REVISION])
    subprocess.Popen(cmd)
