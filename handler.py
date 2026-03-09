import runpod
import subprocess
import requests
import time

MODEL_PATH = "/runpod-volume/models/qwen3.5-35b/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
SERVER_URL = "http://127.0.0.1:8080"
CTX_SIZE = 32768   # 4090 has enough headroom; reduce to 16384 if KV cache OOM
N_GPU_LAYERS = 99

server_proc = None

def start_server():
    global server_proc
    cmd = [
        "/llama.cpp/build/bin/llama-server",
        "--model", MODEL_PATH,
        "--ctx-size", str(CTX_SIZE),
        "--n-gpu-layers", str(N_GPU_LAYERS),
        "--host", "127.0.0.1",
        "--port", "8080",
        "--parallel", "1",
        "--cont-batching",
        "--jinja",  # required for tool calling
    ]
    server_proc = subprocess.Popen(cmd)
    for _ in range(120):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                print("llama-server ready")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server failed to start in time")

def handler(job):
    # LiteLLM sends a standard OpenAI chat completions body as job["input"]
    # Pass it straight through to llama-server — no translation needed
    job_input = job["input"]

    # Inject enable_thinking: false — required for Qwen3.5 via llama-server
    if "chat_template_kwargs" not in job_input:
        job_input["chat_template_kwargs"] = {"enable_thinking": False}

    resp = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=job_input,
        timeout=300,
    )
    return resp.json()

if __name__ == "__main__":
    start_server()
    runpod.serverless.start({"handler": handler})
