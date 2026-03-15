# qwen3-tts-api

REST wrapper for **Qwen3-TTS** with Docker + Kubernetes manifests.

- Default port: `8888`
- Default model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Default response format: `wav`

## Endpoints

### `POST /api/tts`
Synthesize speech.

Request JSON:
```json
{
  "text": "Hello from Qwen3-TTS",
  "language": "English",
  "audio_format": "wav",
  "mode": "design",
  "model_size": "quality",
  "voice_preset": "sentia_calm",
  "voice_description": "Optional explicit design description",
  "reference_audio": "myvoice.wav"
}
```

Fields:
- `text` (required)
- `language` (default `Auto`)
- `audio_format` (`wav` or `ogg`)
- `mode` (`clone` or `design`)
- `model_size` (`fast`, `quality`, or `auto`; design uses `quality`)
  - `auto` selects `quality` when free CUDA VRAM is >= `QWEN_AUTO_QUALITY_MIN_VRAM_MB` (default `3000`), otherwise `fast`
- `reference_audio` (required in `clone` mode)
- `voice_description` (required in `design` mode unless `voice_preset` is provided)
- `voice_preset` (optional preset name from `/voice-presets`; overrides `voice_description` and forces `design` mode)

### `GET /healthz`
Process health.

### `GET /readyz`
Model readiness.

### `GET /speakers`
Supported speakers (if model supports it).

### `GET /info`
Returns default model/speaker/language, supported language list, and speakers.

### Reference-audio management
- `GET /reference-audio` → list uploaded files
- `POST /reference-audio/upload` → multipart upload (`file` form field)
- `DELETE /reference-audio/{filename}` → remove file

### Voice design presets
- `GET /voice-presets` → list presets (`default` + `presets[]`)
- `POST /voice-presets` → create/update preset
- `DELETE /voice-presets/{name}` → delete preset (default preset `sentia_calm` is protected)

Seeded presets on startup:
- `sentia_calm` (default)
- `sentia_alert`

## Voice cloning behavior

When `reference_audio` is set in `/api/tts`, the API validates `/reference-audio/<filename>` and calls:
- `model.generate_voice_clone(..., ref_audio=<path>, x_vector_only_mode=True)`

This follows the Qwen3-TTS voice clone API (`ref_audio` parameter), allowing single-shot clone usage without `ref_text`.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

export QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8888
```

## Kubernetes

Manifests in `k8s/`:
- `configmap.yaml`
- `deployment.yaml`
- `service.yaml`
- `pvc-reference-audio.yaml`

Apply:
```bash
kubectl apply -f k8s/pvc-reference-audio.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Environment variables

- `QWEN_MODEL_ID`
- `QWEN_DEVICE`
- `QWEN_DTYPE`
- `QWEN_ATTN_IMPL`
- `QWEN_DEFAULT_LANGUAGE`
- `QWEN_DEFAULT_SPEAKER`
- `QWEN_DEFAULT_INSTRUCT`
- `QWEN_AUTO_QUALITY_MIN_VRAM_MB` (used when `model_size=auto`, default `3000`)

## Notes

- `POST /reference-audio/upload` requires `python-multipart`.
- If FlashAttention2 fails, set `QWEN_ATTN_IMPL=eager`.

## References

- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- HF collection: https://huggingface.co/collections/Qwen/qwen3-tts
