FROM ghcr.io/ggml-org/llama.cpp:server-cuda12

RUN apt-get update && apt-get install -y python3 python3-venv && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install runpod requests

COPY handler.py /handler.py

ENTRYPOINT ["/opt/venv/bin/python3", "/handler.py"]
