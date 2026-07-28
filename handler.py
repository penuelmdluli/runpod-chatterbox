"""RunPod Serverless handler — Chatterbox TTS (voice + optional cloning).

Input (event["input"]):
  text          (str, required) text to speak
  ref_audio     (str, optional) base64 / data-URI / URL of a reference voice
                (~10s) to CLONE. Omit for the default voice.
  exaggeration  (float) 0-1 emotion intensity (default 0.5)
  cfg_weight    (float) pacing/guidance (default 0.5)

Output: { audio_base64 (wav), sample_rate, seconds } | { error, trace }
"""
import base64
import io
import os
import tempfile
import time
import traceback

import runpod

_M = {}


def _log(m):
    print(f"[chatterbox] {m}", flush=True)


def _model():
    if "m" in _M:
        return _M["m"]
    from chatterbox.tts import ChatterboxTTS
    t = time.time()
    _M["m"] = ChatterboxTTS.from_pretrained(device="cuda")
    _log(f"loaded in {time.time()-t:.1f}s")
    return _M["m"]


def _save_ref(field):
    field = str(field).strip()
    if field.startswith(("http://", "https://")):
        import requests
        data = requests.get(field, timeout=120).content
    else:
        if field.startswith("data:") and "," in field:
            field = field.split(",", 1)[1]
        data = base64.b64decode(field)
    p = os.path.join(tempfile.gettempdir(), f"ref_{int(time.time()*1000)}.wav")
    with open(p, "wb") as f:
        f.write(data)
    return p


def handler(event):
    t0 = time.time()
    try:
        import soundfile as sf
        inp = (event or {}).get("input", {}) or {}
        text = inp.get("text")
        if not text or not str(text).strip():
            return {"error": "missing required field 'text'"}
        kwargs = {}
        if inp.get("ref_audio"):
            kwargs["audio_prompt_path"] = _save_ref(inp["ref_audio"])
        if inp.get("exaggeration") is not None:
            kwargs["exaggeration"] = float(inp["exaggeration"])
        if inp.get("cfg_weight") is not None:
            kwargs["cfg_weight"] = float(inp["cfg_weight"])

        model = _model()
        _log(f"generating {len(text)} chars")
        wav = model.generate(text, **kwargs)
        # wav is a torch tensor [1, N] at model.sr
        import torch
        audio = wav.squeeze(0).detach().cpu().numpy() if hasattr(wav, "squeeze") else wav
        buf = io.BytesIO()
        sf.write(buf, audio, model.sr, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"audio_base64": b64, "sample_rate": model.sr,
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        tb = traceback.format_exc()
        _log("ERROR:\n" + tb)
        return {"error": str(e), "trace": tb[-1500:]}


runpod.serverless.start({"handler": handler})
