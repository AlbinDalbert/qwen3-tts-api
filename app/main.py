import base64
import io
import json
import os
import re
import threading
import gc
import time
import queue
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from transformers import LogitsProcessor, LogitsProcessorList

SUPPORTED_LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "French", "German", "Spanish", "Italian", "Portuguese", "Arabic", "Russian",
]
REFERENCE_AUDIO_DIR = Path("/reference-audio")
DEFAULT_VOICE_PRESET_NAME = "sentia_calm"
SEEDED_VOICE_PRESETS = [
    {
        "name": "sentia_calm",
        "voice_description": "Female, mid-to-low pitch, slight rasp — lived-in not gravelly. Warm but precise. Direct. Like someone who has seen things and finds it quietly amusing.",
        "description": "Sentia default calm voice",
    },
    {
        "name": "sentia_alert",
        "voice_description": "Female, mid-to-low pitch, slight rasp — lived-in not gravelly. Warm but precise. Alert and switched on — like someone who processes fast and finds it quietly amusing. Direct. Slight European neutral accent.",
        "description": "Sentia alert/switched-on voice for technical or energetic messages",
    },
]
MODEL_IDS = {
    "clone": {
        "fast": os.getenv("QWEN_CLONE_MODEL_ID_06", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        "quality": os.getenv("QWEN_CLONE_MODEL_ID_17", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    },
    "design": {
        "quality": os.getenv("QWEN_DESIGN_MODEL_ID_17", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    },
}
model_status = {"state": "idle", "model_id": None}  # idle | loading | ready


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="Auto", description="Language or Auto")
    audio_format: str = Field(default="wav", description="wav | ogg")
    mode: str = Field(default="clone", description="clone | design")
    model_size: str = Field(default="fast", description="fast | quality | auto")
    reference_audio: Optional[str] = Field(default=None, description="Filename from /reference-audio when mode=clone")
    voice_description: Optional[str] = Field(default=None, description="Plain-text voice description when mode=design")
    voice_preset: Optional[str] = Field(default=None, description="Named voice preset to resolve into voice_description")
    user: Optional[str] = Field(default="default", description="User namespace for reference audio")


class SaveDesignRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)
    audio_b64: str = Field(..., min_length=1)


class VoicePresetRequest(BaseModel):
    name: str = Field(..., min_length=1)
    voice_description: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None)


class ResetRequest(BaseModel):
    restart: bool = Field(default=False, description="Restart process after reset")


class QwenService:
    def __init__(self) -> None:
        self.device = os.getenv("QWEN_DEVICE", "cuda:0")
        self.dtype = os.getenv("QWEN_DTYPE", "bfloat16")
        self.attn_impl = os.getenv("QWEN_ATTN_IMPL", "flash_attention_2")
        # Force local-only model resolution by default so runtime never depends on
        # live Hugging Face API checks (works with pre-populated HF cache).
        self.local_files_only = os.getenv("QWEN_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"}
        self.auto_quality_min_vram_mb = int(os.getenv("QWEN_AUTO_QUALITY_MIN_VRAM_MB", "3000"))
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self._model = None
        self._model_id: Optional[str] = None

    def _torch_dtype(self):
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16

    def _unload_model_locked(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    def get_model(self, model_id: str):
        if self._model is not None and self._model_id == model_id:
            return self._model

        if not self.load_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Model load already in progress")

        try:
            with self.lock:
                if self._model is not None and self._model_id == model_id:
                    return self._model

                model_status.update({"state": "loading", "model_id": model_id})
                self._unload_model_locked()

                self._model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=self._torch_dtype(),
                    attn_implementation=self.attn_impl,
                    local_files_only=self.local_files_only,
                )
                self._model_id = model_id
                model_status.update({"state": "ready", "model_id": model_id})
                return self._model
        except Exception:
            model_status.update({"state": "idle", "model_id": None})
            raise
        finally:
            self.load_lock.release()

    def hard_reset(self):
        with self.lock:
            self._unload_model_locked()
            model_status.update({"state": "idle", "model_id": None})

    def resolve_reference_audio(self, filename: str, user_dir: Path) -> Path:
        candidate = (user_dir / filename).resolve()
        if not str(candidate).startswith(str(user_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Reference audio not found: {filename}")
        return candidate

    def _resolve_model_size(self, mode: str, requested_size: str) -> str:
        size = (requested_size or "quality").strip().lower()
        if size not in {"fast", "quality", "auto"}:
            size = "quality"

        if size == "auto":
            threshold_bytes = self.auto_quality_min_vram_mb * 1024 * 1024
            free_vram_bytes = 0
            total_vram_bytes = 0

            if torch.cuda.is_available():
                try:
                    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
                except Exception:
                    free_vram_bytes = 0
                    total_vram_bytes = 0

            selected_size = "quality" if free_vram_bytes >= threshold_bytes else "fast"
            print(
                json.dumps(
                    {
                        "event": "tts_model_size_auto",
                        "mode": mode,
                        "requested": "auto",
                        "selected": selected_size,
                        "free_vram_mb": int(free_vram_bytes / (1024 * 1024)),
                        "total_vram_mb": int(total_vram_bytes / (1024 * 1024)),
                        "threshold_mb": self.auto_quality_min_vram_mb,
                    }
                )
            )
            size = selected_size

        if mode == "design" and size == "fast":
            print(json.dumps({"event": "tts_warn", "detail": "design mode does not support fast; using quality"}))
            size = "quality"

        return size

    def synthesize(self, req: TTSRequest, user_dir: Path, extra_kwargs: Optional[dict] = None):
        extra_kwargs = extra_kwargs or {}
        mode = (req.mode or "clone").strip().lower()
        if mode not in MODEL_IDS:
            raise HTTPException(status_code=400, detail="mode must be clone or design")

        size = self._resolve_model_size(mode, req.model_size or "quality")

        if size not in MODEL_IDS[mode]:
            size = "quality"

        model_id = MODEL_IDS[mode][size]

        if mode == "clone":
            if not req.reference_audio:
                raise HTTPException(status_code=400, detail="reference_audio required for clone mode")
            model = self.get_model(model_id)
            ref_path = self.resolve_reference_audio(req.reference_audio, user_dir)
            wavs, sr = model.generate_voice_clone(
                text=req.text,
                language=req.language or "Auto",
                ref_audio=str(ref_path),
                x_vector_only_mode=True,
                **extra_kwargs,
            )
        else:
            if not req.voice_description:
                raise HTTPException(status_code=400, detail="voice_description required for design mode")
            model = self.get_model(model_id)
            wavs, sr = model.generate_voice_design(
                text=req.text,
                language=req.language or "Auto",
                instruct=req.voice_description,
                **extra_kwargs,
            )

        return wavs[0], sr, model_id, size


app = FastAPI(title="qwen3-tts-api", version="0.5.0")
svc = QwenService()
start_time = time.time()

# ---------------------------------------------------------------------------
# Text normalization + chunking
# ---------------------------------------------------------------------------
MAX_CHUNK_CHARS = int(os.getenv("QWEN_MAX_CHUNK_CHARS", "800"))


def normalize_text(text: str) -> str:
    """Clean up messy whitespace before synthesis."""
    # Collapse 3+ newlines → paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs (but not newlines) → single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _group_parts(parts: list[str], max_chars: int, sep: str = " ") -> list[str]:
    """Greedily join parts into chunks up to max_chars."""
    result: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + len(sep) + len(part) <= max_chars:
            current += sep + part
        else:
            result.append(current)
            current = part
    if current:
        result.append(current)
    return result


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into contextually-aware chunks.

    Strategy (in order):
      1. Normalize whitespace.
      2. If the whole text fits, return as-is.
      3. Split on paragraph breaks (double newline).
      4. If a paragraph is still too long, split on single newlines first
         (preferred — natural line breaks mid-paragraph).
      5. If a line-group is still too long, fall back to sentence boundaries.
    """
    text = normalize_text(text)

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        # Prefer splitting on single newlines (explicit line breaks)
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        line_groups = _group_parts(lines, max_chars)

        for group in line_groups:
            if len(group) <= max_chars:
                chunks.append(group)
            else:
                # Fall back to sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", group)
                chunks.extend(_group_parts(sentences, max_chars))

    return chunks if chunks else [text]


def meta_path_for(directory: Path, filename: str) -> Path:
    return directory / f"{filename}.meta.json"


def write_meta(directory: Path, filename: str, source: str, description: Optional[str] = None):
    payload = {"source": source}
    if description is not None:
        payload["description"] = description
    meta_path_for(directory, filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read_meta(directory: Path, filename: str):
    path = meta_path_for(directory, filename)
    if not path.is_file():
        return {"source": "upload", "description": None}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"source": "upload", "description": None}

    source = payload.get("source") if isinstance(payload, dict) else "upload"
    description = payload.get("description") if isinstance(payload, dict) else None
    if source not in {"upload", "design"}:
        source = "upload"
    if description is not None and not isinstance(description, str):
        description = str(description)
    return {"source": source, "description": description}




def user_audio_dir(user: str) -> Path:
    candidate = (user or "default").strip()
    if not candidate or len(candidate) > 32 or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise HTTPException(status_code=400, detail="Invalid user. Use up to 32 chars: letters, numbers, underscore, hyphen")
    directory = REFERENCE_AUDIO_DIR / candidate
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def user_presets_path(user: str) -> Path:
    return user_audio_dir(user) / "voice_presets.json"


voice_presets_lock = threading.Lock()


def _normalize_preset_name(name: str) -> str:
    normalized = (name or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Preset name is required")
    return normalized


def _normalize_voice_preset_payload(payload: dict) -> dict:
    name = _normalize_preset_name(str(payload.get("name", "")))
    voice_description = str(payload.get("voice_description", "")).strip()
    if not voice_description:
        raise HTTPException(status_code=400, detail="voice_description is required")
    description = payload.get("description")
    if description is not None:
        description = str(description).strip() or None
    return {
        "name": name,
        "voice_description": voice_description,
        "description": description,
    }


def load_voice_presets(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    presets: dict[str, dict] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                try:
                    normalized = _normalize_voice_preset_payload(item)
                    presets[normalized["name"]] = normalized
                except HTTPException:
                    continue
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                candidate = {"name": key, **value}
            else:
                candidate = {"name": key, "voice_description": value}
            try:
                normalized = _normalize_voice_preset_payload(candidate)
                presets[normalized["name"]] = normalized
            except HTTPException:
                continue

    return presets


def save_voice_presets(path: Path, presets: dict[str, dict]):
    ordered = [presets[name] for name in sorted(presets)]
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_voice_presets_if_needed(user: str) -> dict[str, dict]:
    path = user_presets_path(user)
    presets = load_voice_presets(path)
    changed = False
    if user == "sentia":
        for seed in SEEDED_VOICE_PRESETS:
            if seed["name"] not in presets:
                presets[seed["name"]] = seed.copy()
                changed = True
    if changed or (user == "sentia" and not path.is_file()):
        save_voice_presets(path, presets)
    return presets


def apply_voice_preset(req: TTSRequest) -> TTSRequest:
    if not req.voice_preset:
        return req

    preset_name = _normalize_preset_name(req.voice_preset)
    with voice_presets_lock:
        presets = seed_voice_presets_if_needed(req.user or "default")
    preset = presets.get(preset_name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"voice preset not found: {preset_name}")

    return req.model_copy(
        update={
            "mode": "design",
            "voice_description": preset["voice_description"],
            "voice_preset": preset_name,
        }
    )


@app.on_event("startup")
def ensure_reference_audio_dir():
    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    legacy_audio = [
        p for p in REFERENCE_AUDIO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".ogg", ".mp3"}
    ]
    if legacy_audio:
        dna_dir = REFERENCE_AUDIO_DIR / "dna"
        dna_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(legacy_audio):
            dst = dna_dir / src.name
            src.replace(dst)
            print(json.dumps({"event": "reference_audio_migrated", "from": str(src), "to": str(dst)}))

            src_sidecar = src.with_name(f"{src.name}.meta.json")
            if src_sidecar.is_file():
                dst_sidecar = dna_dir / src_sidecar.name
                src_sidecar.replace(dst_sidecar)
                print(json.dumps({"event": "reference_audio_migrated", "from": str(src_sidecar), "to": str(dst_sidecar)}))

    legacy_presets_path = REFERENCE_AUDIO_DIR / "voice_presets.json"
    sentia_presets_path = user_presets_path("sentia")
    if legacy_presets_path.is_file() and not sentia_presets_path.exists():
        legacy_presets_path.replace(sentia_presets_path)
        print(json.dumps({"event": "voice_presets_migrated", "from": str(legacy_presets_path), "to": str(sentia_presets_path)}))

    with voice_presets_lock:
        seed_voice_presets_if_needed("sentia")


@app.get("/healthz")
def healthz():
    return {"ok": True, "uptime_s": int(time.time() - start_time)}


@app.get("/status")
def status():
    return {
        "model_state": model_status["state"],
        "model_id": model_status["model_id"],
        "uptime_s": int(time.time() - start_time),
    }


@app.get("/readyz")
def readyz():
    if model_status["state"] == "ready":
        return {"ready": True, "model_id": model_status["model_id"]}
    return JSONResponse(
        status_code=503,
        content={"ready": False, "reason": model_status["state"]},
    )


@app.get("/info")
def info():
    return {
        "models": MODEL_IDS,
        "modes": ["clone", "design"],
        "sizes": ["fast", "quality", "auto"],
        "current_model": svc._model_id,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    svc.hard_reset()

    response = {"ok": True, "state": "idle", "model_id": None}
    if req and req.restart:
        exit_code = int(os.getenv("QWEN_RESET_EXIT_CODE", "1"))

        def _delayed_exit():
            time.sleep(0.2)
            os._exit(exit_code)

        threading.Thread(target=_delayed_exit, daemon=True).start()
        response["restart_scheduled"] = True
        response["exit_code"] = exit_code

    return response


@app.get("/reference-audio")
def list_reference_audio(user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    files = []
    for p in sorted(dir_path.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name.endswith(".meta.json"):
            continue
        meta = read_meta(dir_path, p.name)
        files.append(
            {
                "filename": p.name,
                "source": meta.get("source", "upload"),
                "description": meta.get("description"),
            }
        )

    return {"files": files}


@app.post("/reference-audio/upload")
async def upload_reference_audio(file: UploadFile = File(...), user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    target = (dir_path / filename).resolve()
    if not str(target).startswith(str(dir_path.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    target.write_bytes(data)
    write_meta(dir_path, filename, source="upload")
    return {"ok": True, "filename": filename, "size": len(data)}


@app.post("/reference-audio/save-design")
def save_design_reference_audio(req: SaveDesignRequest, user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    filename = Path(req.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    target = (dir_path / filename).resolve()
    if not str(target).startswith(str(dir_path.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")

    audio_b64 = req.audio_b64.strip()
    if "," in audio_b64 and audio_b64.lower().startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]

    try:
        data = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio_b64: {e}")

    if not data:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")

    target.write_bytes(data)
    write_meta(dir_path, filename, source="design", description=(req.description or "").strip() or None)
    return {"ok": True, "filename": filename}


@app.delete("/reference-audio/{filename}")
def delete_reference_audio(filename: str, user: str = Query(default="default")):
    dir_path = user_audio_dir(user)
    target = svc.resolve_reference_audio(filename, dir_path)
    target.unlink()
    sidecar = meta_path_for(dir_path, target.name)
    if sidecar.exists():
        sidecar.unlink()
    return {"ok": True, "deleted": target.name}


@app.get("/voice-presets")
def list_voice_presets(user: str = Query(default="default")):
    with voice_presets_lock:
        presets = seed_voice_presets_if_needed(user)
        items = [presets[name] for name in sorted(presets)]
    return {
        "default": DEFAULT_VOICE_PRESET_NAME,
        "presets": items,
    }


@app.post("/voice-presets")
def upsert_voice_preset(req: VoicePresetRequest, user: str = Query(default="default")):
    normalized = _normalize_voice_preset_payload(req.model_dump())
    with voice_presets_lock:
        path = user_presets_path(user)
        presets = seed_voice_presets_if_needed(user)
        presets[normalized["name"]] = normalized
        save_voice_presets(path, presets)
    return {"ok": True, "preset": normalized}


@app.delete("/voice-presets/{name}")
def delete_voice_preset(name: str, user: str = Query(default="default")):
    preset_name = _normalize_preset_name(name)
    if preset_name == DEFAULT_VOICE_PRESET_NAME:
        raise HTTPException(status_code=400, detail=f"cannot delete default preset: {DEFAULT_VOICE_PRESET_NAME}")

    with voice_presets_lock:
        path = user_presets_path(user)
        presets = seed_voice_presets_if_needed(user)
        if preset_name not in presets:
            raise HTTPException(status_code=404, detail=f"voice preset not found: {preset_name}")
        del presets[preset_name]
        save_voice_presets(path, presets)

    return {"ok": True, "deleted": preset_name}


@app.post("/api/tts/stream")
def tts_stream(req: TTSRequest):
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    audio_format = req.audio_format.lower().strip()
    if audio_format != "wav":
        raise HTTPException(status_code=400, detail="audio_format must be wav for stream endpoint")

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)

    q: queue.Queue = queue.Queue()

    def run_generation():
        try:
            wav_arrays: list[np.ndarray] = []
            sr_final = 24000
            total_tokens = 0

            for chunk_idx, chunk in enumerate(chunks):
                chunk_estimated = max(1, int((len(chunk) / 5) * 2.5 * 12))
                chunk_token_count = [0]

                q.put(("chunk_start", {
                    "chunk": chunk_idx + 1,
                    "of": n_chunks,
                    "chars": len(chunk),
                }))

                class ProgressProcessor(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        chunk_token_count[0] += 1
                        # Progress within this chunk (0–95%), scaled to its share of total
                        chunk_pct = min(95, int((chunk_token_count[0] / chunk_estimated) * 100))
                        # Overall progress across all chunks
                        overall_pct = int(((chunk_idx + chunk_pct / 100) / n_chunks) * 100)
                        q.put(("progress", {
                            "chunk": chunk_idx + 1,
                            "of": n_chunks,
                            "chunk_pct": chunk_pct,
                            "overall_pct": min(95, overall_pct),
                            "tokens": chunk_token_count[0],
                        }))
                        return scores

                chunk_req = req.model_copy(update={"text": chunk})
                wav, sr, _used_model_id, _used_size = svc.synthesize(
                    chunk_req,
                    user_dir,
                    extra_kwargs={
                        "logits_processor": LogitsProcessorList([ProgressProcessor()]),
                    },
                )
                wav_arrays.append(np.asarray(wav, dtype=np.float32))
                sr_final = sr
                total_tokens += chunk_token_count[0]

                q.put(("chunk_done", {
                    "chunk": chunk_idx + 1,
                    "of": n_chunks,
                    "tokens": chunk_token_count[0],
                }))

            data = np.concatenate(wav_arrays) if len(wav_arrays) > 1 else wav_arrays[0]
            buf = io.BytesIO()
            sf.write(buf, data, sr_final, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode()
            q.put(("done", {
                "audio_b64": audio_b64,
                "chunks": n_chunks,
                "tokens_generated": total_tokens,
            }))
        except Exception as e:
            q.put(("error", {"detail": str(e)}))

    threading.Thread(target=run_generation, daemon=True).start()

    def event_stream():
        while True:
            event_type, payload = q.get()
            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()
            if event_type in {"done", "error"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/tts")
def tts(req: TTSRequest):
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    t0 = time.time()
    audio_format = req.audio_format.lower().strip()
    if audio_format not in {"wav", "ogg"}:
        raise HTTPException(status_code=400, detail="audio_format must be wav or ogg")

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)

    t1 = time.time()
    wav_arrays: list[np.ndarray] = []
    sr_final: int = 24000
    used_model_id: str = ""
    used_size: str = ""

    for i, chunk in enumerate(chunks):
        chunk_req = req.model_copy(update={"text": chunk})
        wav, sr, used_model_id, used_size = svc.synthesize(chunk_req, user_dir)
        wav_arrays.append(np.asarray(wav, dtype=np.float32))
        sr_final = sr
        if n_chunks > 1:
            print(json.dumps({"event": "tts_chunk", "chunk": i + 1, "of": n_chunks, "chars": len(chunk)}))

    data = np.concatenate(wav_arrays) if len(wav_arrays) > 1 else wav_arrays[0]
    t2 = time.time()

    buf = io.BytesIO()
    if audio_format == "wav":
        sf.write(buf, data, sr_final, format="WAV")
        mime = "audio/wav"
    else:
        sf.write(buf, data, sr_final, format="OGG", subtype="VORBIS")
        mime = "audio/ogg"
    t3 = time.time()

    total = t3 - t0
    t_synth = t2 - t1
    t_encode = t3 - t2

    try:
        print(
            json.dumps(
                {
                    "event": "tts_timing",
                    "mode": req.mode,
                    "model_size": used_size,
                    "chars": len(req.text or ""),
                    "chunks": n_chunks,
                    "model_id": used_model_id,
                    "t_total_ms": int(total * 1000),
                    "t_synth_ms": int(t_synth * 1000),
                    "t_encode_ms": int(t_encode * 1000),
                }
            )
        )
    except Exception:
        pass

    return Response(content=buf.getvalue(), media_type=mime)
