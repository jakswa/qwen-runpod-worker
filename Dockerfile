FROM ghcr.io/ggerganov/llama.cpp:server-cuda

RUN pip3 install runpod requests

COPY handler.py /handler.py

CMD ["python3", "/handler.py"]
