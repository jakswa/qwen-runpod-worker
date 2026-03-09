FROM ghcr.io/ggml-org/llama.cpp:server-cuda13

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages runpod requests

COPY handler.py /handler.py

# Base image sets ENTRYPOINT to /app/llama-server — must override or CMD becomes llama-server args
ENTRYPOINT ["python3", "/handler.py"]
