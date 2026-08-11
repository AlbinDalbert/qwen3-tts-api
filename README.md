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
  "cpu_mode": false,
  "voice_preset": "sentia_calm",
  "voice_description": "Optional explicit design description",
  "reference_audio": "myvoice.wav"
}
```

### `POST /api/tts/audio-stream`
Stream synthesized audio directly as `audio/wav` using HTTP chunked transfer.

This endpoint synthesizes one normalized text chunk at a time, yields that audio immediately, then releases the chunk before generating the next one. This is the preferred endpoint for long text because it avoids building one huge waveform in memory. Current `qwen-tts==0.1.1` does not expose token-level audio frames, so streaming granularity is text chunks.

Same request JSON/fields as `/api/tts`; `audio_format` must be `wav`.

Example:
```bash
curl -N -X POST http://localhost:8888/api/tts/audio-stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from streamed Qwen3-TTS","mode":"design","voice_preset":"sentia_calm","audio_format":"wav"}' \
  --output stream.wav
```

### `POST /api/tts/stream`
Server-sent events with progress and a final base64 WAV payload. Useful for UI progress, but not memory-efficient for long text; prefer `/api/tts/audio-stream` for long text.

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
- `cpu_mode` (default `false`; when `true`, runs the complete TTS pipeline on CPU with no GPU fallback)

### Errors

If CUDA runs out of VRAM while loading or running a requested model, the API returns HTTP `503` with a retry hint (or an equivalent error event if streaming headers were already sent):

```json
{"detail":"GPU busy"}
```

CPU mode is explicit only: the API never falls back from GPU to CPU automatically.

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

## Build and push

```bash
./scripts/build-push.sh
```

Equivalent manual commands:

```bash
docker build -t ghcr.io/albindalbert/qwen3-tts-api:latest .
docker push ghcr.io/albindalbert/qwen3-tts-api:latest
```

Override the image if needed:

```bash
IMAGE=ghcr.io/albindalbert/qwen3-tts-api:dev ./scripts/build-push.sh
```

## Kubernetes

Manifests in `k8s/`:
- `configmap.yaml`
- `deployment.yaml`
- `service.yaml`
- `pvc-reference-audio.yaml`

Apply small-GPU profile (default; CPU-offloads auxiliary modules for ~3 GiB VRAM):
```bash
kubectl apply -f k8s/pvc-reference-audio.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Apply larger-GPU profile (~8 GiB VRAM; keeps auxiliary modules on CUDA for lower latency):
```bash
kubectl apply -f k8s/configmap-8gb-gpu.yaml
kubectl rollout restart deployment/qwen3-tts-api-caladan -n sietch-sentia
```

## Environment variables

- `QWEN_MODEL_ID`
- `QWEN_DEVICE`
- `QWEN_DTYPE`
- `QWEN_ATTN_IMPL` (default `sdpa`; use `eager` if SDPA causes trouble)
- `QWEN_CPU_DTYPE` (default `float32`; used when request `cpu_mode=true`)
- `QWEN_CPU_ATTN_IMPL` (default `sdpa`; used when request `cpu_mode=true`)
- `QWEN_DEFAULT_LANGUAGE`
- `QWEN_DEFAULT_SPEAKER`
- `QWEN_DEFAULT_INSTRUCT`
- `QWEN_AUTO_QUALITY_MIN_VRAM_MB` (used when `model_size=auto`, default `3000`)
- `QWEN_MAX_CHUNK_CHARS` (text chunk size for long text/streaming, default `160`)
- `QWEN_STREAM_PREBUFFER_CHUNKS` (default `1`; increase to `2` or `3` to trade startup delay for fewer playback gaps)
- `QWEN_TRIM_CHUNK_SILENCE` (default `1`; trims generated leading/trailing silence at chunk boundaries)
- `QWEN_TRIM_SILENCE_KEEP_MS` (default `60`)
- `QWEN_CHUNK_FADE_MS` (default `8`)
- `QWEN_MAX_NEW_TOKENS` (hard generation cap per chunk, default `192`)
- `QWEN_MAX_NEW_TOKENS_PER_CHAR` (dynamic output-token estimate, default `1.0`)
- `QWEN_MAX_NEW_TOKENS_BUFFER` (extra output-token headroom per chunk, default `48`)
- `QWEN_NON_STREAMING_MODE` (`0` by default; qwen's name is inverted, so `0` enables its lower-latency streaming-text path)
- `QWEN_SPEECH_TOKENIZER_DEVICE` (default `cpu`; keep vocoder/codec decoder off small GPUs)
- `QWEN_SPEECH_TOKENIZER_CPU_DTYPE` (default `float32`)
- `QWEN_SPEAKER_ENCODER_DEVICE` (default `cpu`; keep clone embedding model off small GPUs)
- `QWEN_SPEAKER_ENCODER_CPU_DTYPE` (default `float32`)
- `PYTORCH_CUDA_ALLOC_CONF` (default image/config: `expandable_segments:True`)

### GPU profiles

Small GPU / smoother buffered profile:
```bash
QWEN_SPEECH_TOKENIZER_DEVICE=cpu
QWEN_SPEAKER_ENCODER_DEVICE=cpu
QWEN_MAX_CHUNK_CHARS=220
QWEN_STREAM_PREBUFFER_CHUNKS=2
QWEN_TRIM_CHUNK_SILENCE=1
QWEN_MAX_NEW_TOKENS=192
QWEN_MAX_NEW_TOKENS_PER_CHAR=1.0
QWEN_MAX_NEW_TOKENS_BUFFER=48
```

Larger GPU / lower-latency profile:
```bash
QWEN_SPEECH_TOKENIZER_DEVICE=cuda:0
QWEN_SPEAKER_ENCODER_DEVICE=cuda:0
QWEN_STREAM_PREBUFFER_CHUNKS=1
QWEN_TRIM_CHUNK_SILENCE=1
QWEN_MAX_NEW_TOKENS=512
QWEN_MAX_NEW_TOKENS_PER_CHAR=2.0
QWEN_MAX_NEW_TOKENS_BUFFER=64
```

`QWEN_DEVICE` controls the main Qwen generator. `QWEN_SPEECH_TOKENIZER_DEVICE` controls the codec/vocoder decoder, and `QWEN_SPEAKER_ENCODER_DEVICE` controls clone speaker embedding extraction. Use `cpu`, `cuda:0`, or `keep`. A request with `cpu_mode=true` overrides these device settings for that request and keeps the complete pipeline on CPU.

## Notes

- TTS models are loaded on demand and explicitly unloaded after every completed TTS request, including streamed requests and failed generations. This releases their CUDA allocations; the next request incurs model load time.
- `POST /reference-audio/upload` requires `python-multipart`.
- If FlashAttention2 fails, set `QWEN_ATTN_IMPL=eager`.

## References

- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- HF collection: https://huggingface.co/collections/Qwen/qwen3-tts
