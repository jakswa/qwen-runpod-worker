import runpod
import subprocess
import requests
import time
import sys

MODEL_PATH = "/runpod-volume/models/qwen3.5-35b/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
SERVER_URL = "http://127.0.0.1:8080"
CTX_SIZE = 32768   # drop to 16384 if KV cache OOM on 4090
N_GPU_LAYERS = 99

server_proc = None


def start_server():
    global server_proc
    cmd = [
        "/app/llama-server",          # correct path in the official image
        "--model", MODEL_PATH,
        "--ctx-size", str(CTX_SIZE),
        "--n-gpu-layers", str(N_GPU_LAYERS),
        "--host", "127.0.0.1",
        "--port", "8080",
        "--parallel", "1",
        "--flash-attn",               # enable for throughput; auto in latest builds but explicit is fine
        "--jinja",                    # required for tool calling
        # --cont-batching removed: now always-on default in recent llama.cpp
    ]
    server_proc = subprocess.Popen(cmd)
    for _ in range(120):
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
    # LiteLLM sends a standard OpenAI chat completions body as job["input"]
    job_input = job["input"]

    # Inject enable_thinking: false — Qwen3.5 defaults to thinking-on otherwise
    if "chat_template_kwargs" not in job_input:
        job_input["chat_template_kwargs"] = {"enable_thinking": False}

    # Check server is still alive (catches post-startup crashes/OOMs)
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
    except requests.exceptions.HTTPError as e:
        return {"error": f"llama-server error {resp.status_code}: {resp.text[:500]}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    start_server()
    runpod.serverless.start({"handler": handler})
