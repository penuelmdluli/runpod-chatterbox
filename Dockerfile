# RunPod serverless worker — Chatterbox TTS (Resemble AI). Branded/cloned voice.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir chatterbox-tts soundfile "runpod==1.7.7" requests

# Bake the model weights at build.
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu'); print('chatterbox baked OK')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
