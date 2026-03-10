import glob
import os
import runpod
import subprocess
import requests
import time
import sys

SERVER_URL = "http://127.0.0.1:8080"
CTX_SIZE = 65536   # 64k; 128k OOMs on 4090 with this model
N_GPU_LAYERS = 99

server_proc = None


def find_model():
    """Locate the GGUF file regardless of RunPod's cache path structure."""
    patterns = [
        # RunPod HuggingFace cache (when model URL provided in endpoint config)
        "/runpod-volume/huggingface-cache/hub/*Qwen3.5-35B-A3B*/**/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        # Manual network volume path (from deployment plan)
        "/runpod-volume/models/qwen3.5-35b/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        # Fallback: any Q4_K_XL gguf on the volume
        "/runpod-volume/**/*Q4_K_XL.gguf",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            print(f"Found model at: {matches[0]}", flush=True)
            return matches[0]
    contents = os.listdir("/runpod-volume") if os.path.exists("/runpod-volume") else "volume not mounted"
    raise FileNotFoundError(f"Model GGUF not found. /runpod-volume contents: {contents}")


def start_server():
    global server_proc
    model_path = find_model()
    cmd = [
        "/app/llama-server",
        "--model", model_path,
        "--ctx-size", str(CTX_SIZE),
        "--n-gpu-layers", str(N_GPU_LAYERS),
        "--host", "127.0.0.1",
        "--port", "8080",
        "--parallel", "1",
        "--jinja",
    ]
    log_file = open("/tmp/llama-server.log", "w")
    server_proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    for i in range(300):  # 5 min — 20GB model load from network volume takes time
        if i % 15 == 0:
            # Flush llama-server log to stdout every 15s so RunPod captures it
            log_file.flush()
            try:
                with open("/tmp/llama-server.log") as f:
                    tail = f.read()[-2000:]
                print(f"[llama-server log @ {i}s]\n{tail}", flush=True)
            except Exception as e:
                print(f"[log read error: {e}]", flush=True)
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                print("llama-server ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server failed to start in time")


def is_server_alive():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def handler(job):
    job_input = job["input"]

    # Inject enable_thinking: false — Qwen3.5 defaults to thinking-on otherwise
    if "chat_template_kwargs" not in job_input:
        job_input["chat_template_kwargs"] = {"enable_thinking": False}

    if not is_server_alive():
        return {"error": "llama-server is not responding — worker needs restart"}

    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=job_input,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError:
        return {"error": f"llama-server error {resp.status_code}: {resp.text[:500]}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    start_server()
    runpod.serverless.start({"handler": handler})
