FROM ghcr.io/ggml-org/llama.cpp:server-cuda13

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip3 install runpod requests

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
