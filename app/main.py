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
model_status = {"state": "idle", "model_id": None, "device": None}  # idle | loading | ready
QWEN_MEMORY_PATCH_ENABLED = False


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """Recognize CUDA allocation failures, including wrapped PyTorch errors."""
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
    )
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        if any(marker in str(current).lower() for marker in markers):
            return True
        current = current.__cause__ or current.__context__
    return False


def gpu_busy_error() -> HTTPException:
    return HTTPException(status_code=503, detail="GPU busy", headers={"Retry-After": "5"})


def patch_qwen_tts_memory() -> None:
    """Reduce upstream qwen-tts generation memory use.

    qwen-tts 0.1.1 stores every transformer layer hidden state for every
    generated audio token because it uses the HF `hidden_states` channel to
    smuggle codec ids back out of `generate()`. The public API only needs the
    codec ids. Keep that channel, but replace the huge all-layer payload with
    the one last-token hidden vector the upstream wrapper expects.
    """
    global QWEN_MEMORY_PATCH_ENABLED
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
    except Exception as e:
        print(json.dumps({"event": "qwen_memory_patch_skipped", "detail": str(e)}), flush=True)
        return

    if getattr(Qwen3TTSTalkerForConditionalGeneration, "_qwen3_tts_api_memory_patch", False):
        QWEN_MEMORY_PATCH_ENABLED = True
        return

    import functools

    original_forward = Qwen3TTSTalkerForConditionalGeneration.forward

    @functools.wraps(original_forward)
    def memory_light_forward(self, *args, **kwargs):
        # GenerationMixin still needs output_hidden_states=True at the outer
        # level so it collects our custom `(last_hidden, codec_ids)` payload.
        # But the inner transformer does not need to materialize all layer
        # hidden states; that is the expensive part.
        kwargs["output_hidden_states"] = False
        output = original_forward(self, *args, **kwargs)
        try:
            hidden_payload = getattr(output, "hidden_states", None)
            codec_ids = hidden_payload[-1] if isinstance(hidden_payload, (tuple, list)) and hidden_payload else None
            compact_last_hidden = getattr(output, "past_hidden", None)
            if compact_last_hidden is not None or codec_ids is not None:
                output.hidden_states = ((compact_last_hidden,), codec_ids)
        except Exception as e:
            print(json.dumps({"event": "qwen_memory_patch_payload_error", "detail": str(e)}), flush=True)
        return output

    Qwen3TTSTalkerForConditionalGeneration.forward = memory_light_forward
    Qwen3TTSTalkerForConditionalGeneration._qwen3_tts_api_memory_patch = True
    QWEN_MEMORY_PATCH_ENABLED = True
    print(json.dumps({"event": "qwen_memory_patch_enabled"}), flush=True)


patch_qwen_tts_memory()


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
    cpu_mode: bool = Field(default=False, description="Run this request on CPU; no automatic fallback")


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
        self.cpu_dtype = os.getenv("QWEN_CPU_DTYPE", "float32")
        self.cpu_attn_impl = os.getenv("QWEN_CPU_ATTN_IMPL", "sdpa")
        # Force local-only model resolution by default so runtime never depends on
        # live Hugging Face API checks (works with pre-populated HF cache).
        self.local_files_only = os.getenv("QWEN_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"}
        self.auto_quality_min_vram_mb = int(os.getenv("QWEN_AUTO_QUALITY_MIN_VRAM_MB", "3000"))
        self.max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "192"))
        self.max_new_tokens_per_char = float(os.getenv("QWEN_MAX_NEW_TOKENS_PER_CHAR", "1.0"))
        self.max_new_tokens_buffer = int(os.getenv("QWEN_MAX_NEW_TOKENS_BUFFER", "48"))
        # On ~3 GiB GPUs the 0.6B talker can fit, but the auxiliary vocoder /
        # speech tokenizer and speaker encoder leave almost no generation
        # headroom. Keep those aux modules on CPU by default; only the talker
        # stays on CUDA for generation.
        self.speech_tokenizer_device = os.getenv("QWEN_SPEECH_TOKENIZER_DEVICE", "cpu").strip().lower()
        self.speech_tokenizer_cpu_dtype = os.getenv("QWEN_SPEECH_TOKENIZER_CPU_DTYPE", "float32").strip().lower()
        self.speaker_encoder_device = os.getenv("QWEN_SPEAKER_ENCODER_DEVICE", "cpu").strip().lower()
        self.speaker_encoder_cpu_dtype = os.getenv("QWEN_SPEAKER_ENCODER_CPU_DTYPE", "float32").strip().lower()
        # qwen's parameter is inverted: non_streaming_mode=False enables its
        # lower-latency streaming-text path. Keep false by default.
        self.non_streaming_mode = os.getenv("QWEN_NON_STREAMING_MODE", "0").lower() in {"1", "true", "yes"}
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self._model = None
        self._model_id: Optional[str] = None
        self._model_device: Optional[str] = None

    def _torch_dtype(self, device: str):
        dtype_name = self.cpu_dtype if device == "cpu" else self.dtype
        dtype = self._dtype_from_name(dtype_name)
        if dtype is None:
            raise ValueError(f"QWEN_{'CPU_' if device == 'cpu' else ''}DTYPE must specify a dtype")
        return dtype

    def _dtype_from_name(self, name: str):
        normalized = (name or "").strip().lower()
        if normalized in {"", "none", "keep"}:
            return None
        if normalized in {"float32", "fp32"}:
            return torch.float32
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        raise ValueError(f"unsupported dtype: {name}")

    def _move_module(self, module, device_name: str, cpu_dtype_name: str, label: str) -> None:
        device_name = (device_name or "").strip().lower()
        if device_name in {"", "none", "keep"} or module is None:
            return

        dtype = self._dtype_from_name(cpu_dtype_name) if device_name == "cpu" else None
        try:
            if dtype is None:
                module.to(device_name)
            else:
                module.to(device=device_name, dtype=dtype)
            print(json.dumps({
                "event": "tts_aux_module_moved",
                "module": label,
                "device": device_name,
                "dtype": str(dtype) if dtype is not None else "keep",
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "event": "tts_aux_module_move_failed",
                "module": label,
                "device": device_name,
                "detail": str(e),
            }), flush=True)
            raise

    def _apply_post_load_device_policy(self, wrapped_model, model_device: str) -> None:
        qwen_model = getattr(wrapped_model, "model", None)
        speech_tokenizer = getattr(qwen_model, "speech_tokenizer", None)
        speech_tokenizer_model = getattr(speech_tokenizer, "model", None)
        # cpu_mode applies to the complete pipeline. It must never move an
        # auxiliary module back to CUDA through the normal GPU profile.
        speech_tokenizer_device = "cpu" if model_device == "cpu" else self.speech_tokenizer_device
        speaker_encoder_device = "cpu" if model_device == "cpu" else self.speaker_encoder_device
        self._move_module(
            speech_tokenizer_model,
            speech_tokenizer_device,
            self.speech_tokenizer_cpu_dtype,
            "speech_tokenizer",
        )
        if speech_tokenizer is not None and speech_tokenizer_device not in {"", "none", "keep"}:
            speech_tokenizer.device = torch.device(speech_tokenizer_device)

        self._move_module(
            getattr(qwen_model, "speaker_encoder", None),
            speaker_encoder_device,
            self.speaker_encoder_cpu_dtype,
            "speaker_encoder",
        )
        cleanup_generation_memory()

    def _unload_model_locked(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._model_id = None
        self._model_device = None
        cleanup_generation_memory()

    def get_model(self, model_id: str, cpu_mode: bool = False):
        model_device = "cpu" if cpu_mode else self.device.strip().lower()
        if self._model is not None and self._model_id == model_id and self._model_device == model_device:
            return self._model

        if not self.load_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Model load already in progress")

        try:
            with self.lock:
                if self._model is not None and self._model_id == model_id and self._model_device == model_device:
                    return self._model

                model_status.update({"state": "loading", "model_id": model_id, "device": model_device})
                self._unload_model_locked()

                try:
                    self._model = Qwen3TTSModel.from_pretrained(
                        model_id,
                        device_map=model_device,
                        dtype=self._torch_dtype(model_device),
                        attn_implementation=self.cpu_attn_impl if model_device == "cpu" else self.attn_impl,
                        local_files_only=self.local_files_only,
                    )
                    self._apply_post_load_device_policy(self._model, model_device)
                except Exception as e:
                    # A failed load can leave a partially constructed model and
                    # CUDA cache allocations behind. Release both before replying.
                    self._unload_model_locked()
                    if model_device.startswith("cuda") and is_cuda_out_of_memory(e):
                        print(json.dumps({
                            "event": "tts_model_load_gpu_busy",
                            "model_id": model_id,
                            "device": model_device,
                            "detail": str(e),
                            "cuda": cuda_memory_snapshot(),
                        }), flush=True)
                        raise gpu_busy_error() from None
                    raise

                self._model_id = model_id
                self._model_device = model_device
                model_status.update({"state": "ready", "model_id": model_id, "device": model_device})
                return self._model
        except Exception:
            model_status.update({"state": "idle", "model_id": None, "device": None})
            raise
        finally:
            self.load_lock.release()

    def unload_model(self) -> None:
        """Drop the loaded model and release its CUDA allocations.

        Wait for an in-flight chunk to finish before removing the service's
        final model reference. TTS endpoints call this once their complete
        request (including all chunks) has finished.
        """
        with self.generation_lock:
            with self.lock:
                unloaded_model_id = self._model_id
                had_model = self._model is not None
                self._unload_model_locked()
                model_status.update({"state": "idle", "model_id": None, "device": None})

        if had_model:
            print(json.dumps({
                "event": "tts_model_unloaded",
                "model_id": unloaded_model_id,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)

    def hard_reset(self):
        self.unload_model()

    def resolve_reference_audio(self, filename: str, user_dir: Path) -> Path:
        candidate = (user_dir / filename).resolve()
        if not str(candidate).startswith(str(user_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Reference audio not found: {filename}")
        return candidate

    def _resolve_model_size(self, mode: str, requested_size: str, cpu_mode: bool = False) -> str:
        size = (requested_size or "quality").strip().lower()
        if size not in {"fast", "quality", "auto"}:
            size = "quality"

        if size == "auto":
            threshold_bytes = self.auto_quality_min_vram_mb * 1024 * 1024
            free_vram_bytes = 0
            total_vram_bytes = 0

            if not cpu_mode and torch.cuda.is_available():
                try:
                    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
                except Exception:
                    free_vram_bytes = 0
                    total_vram_bytes = 0

            # CUDA availability is irrelevant to a CPU-only request. Prefer
            # the smaller model there (design is normalized to quality below).
            selected_size = "fast" if cpu_mode else ("quality" if free_vram_bytes >= threshold_bytes else "fast")
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

    def _max_new_tokens_for_text(self, text: str) -> int:
        estimated = int(len(text or "") * self.max_new_tokens_per_char) + self.max_new_tokens_buffer
        return max(32, min(self.max_new_tokens, estimated))

    def _raise_gpu_busy_if_oom(self, exc: BaseException) -> None:
        if (self._model_device or "").startswith("cuda") and is_cuda_out_of_memory(exc):
            cleanup_generation_memory()
            raise gpu_busy_error() from None

    def create_xvector_voice_clone_prompt(self, req: TTSRequest, user_dir: Path):
        """Create a lightweight voice-clone prompt once per request.

        Upstream qwen-tts currently runs the speech tokenizer on the reference
        audio even when x_vector_only_mode=True, then discards those ref codes.
        For our clone mode we only use the speaker embedding, so skip that
        extra encode path.
        """
        mode = (req.mode or "clone").strip().lower()
        if mode != "clone":
            return None
        if not req.reference_audio:
            raise HTTPException(status_code=400, detail="reference_audio required for clone mode")

        size = self._resolve_model_size(mode, req.model_size or "quality", req.cpu_mode)
        if size not in MODEL_IDS[mode]:
            size = "quality"
        model = self.get_model(MODEL_IDS[mode][size], req.cpu_mode)
        ref_path = self.resolve_reference_audio(req.reference_audio, user_dir)

        with self.generation_lock:
            try:
                import librosa

                normalized = model._normalize_audio_inputs([str(ref_path)])
                wav, sr = normalized[0]
                wav_resample = wav
                target_sr = model.model.speaker_encoder_sample_rate
                if sr != target_sr:
                    wav_resample = librosa.resample(
                        y=wav_resample.astype(np.float32),
                        orig_sr=int(sr),
                        target_sr=int(target_sr),
                    )
                from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

                speaker_encoder = model.model.speaker_encoder
                if speaker_encoder is None:
                    raise RuntimeError("loaded model has no speaker_encoder")
                try:
                    spk_param = next(speaker_encoder.parameters())
                    spk_device = spk_param.device
                    spk_dtype = spk_param.dtype
                except StopIteration:
                    spk_device = torch.device("cpu")
                    spk_dtype = torch.float32

                mels = mel_spectrogram(
                    torch.from_numpy(wav_resample.astype(np.float32)).unsqueeze(0),
                    n_fft=1024,
                    num_mels=128,
                    sampling_rate=24000,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=12000,
                ).transpose(1, 2)
                with torch.inference_mode():
                    spk_emb = speaker_encoder(mels.to(spk_device).to(spk_dtype))[0].detach().cpu()
                cleanup_generation_memory()
                return {
                    "ref_code": [None],
                    "ref_spk_embedding": [spk_emb],
                    "x_vector_only_mode": [True],
                    "icl_mode": [False],
                }
            except Exception as e:
                cleanup_generation_memory()
                self._raise_gpu_busy_if_oom(e)
                raise HTTPException(status_code=500, detail=f"failed to create voice clone prompt: {e}")

    def prepare_request_extra_kwargs(self, req: TTSRequest, user_dir: Path) -> dict:
        """Validate and load before an endpoint commits streaming headers."""
        mode = (req.mode or "clone").strip().lower()
        if mode == "clone":
            prompt = self.create_xvector_voice_clone_prompt(req, user_dir)
            return {"voice_clone_prompt": prompt}
        if mode != "design":
            raise HTTPException(status_code=400, detail="mode must be clone or design")
        if not req.voice_description:
            raise HTTPException(status_code=400, detail="voice_description required for design mode")

        size = self._resolve_model_size(mode, req.model_size or "quality", req.cpu_mode)
        if size not in MODEL_IDS[mode]:
            size = "quality"
        self.get_model(MODEL_IDS[mode][size], req.cpu_mode)
        return {}

    def synthesize(self, req: TTSRequest, user_dir: Path, extra_kwargs: Optional[dict] = None):
        extra_kwargs = extra_kwargs.copy() if extra_kwargs else {}
        extra_kwargs.setdefault("max_new_tokens", self._max_new_tokens_for_text(req.text))
        extra_kwargs.setdefault("non_streaming_mode", self.non_streaming_mode)
        print(json.dumps({
            "event": "tts_generate_start",
            "chars": len(req.text or ""),
            "max_new_tokens": extra_kwargs.get("max_new_tokens"),
            "non_streaming_mode": extra_kwargs.get("non_streaming_mode"),
            "cuda": cuda_memory_snapshot(),
        }), flush=True)
        mode = (req.mode or "clone").strip().lower()
        if mode not in MODEL_IDS:
            raise HTTPException(status_code=400, detail="mode must be clone or design")

        size = self._resolve_model_size(mode, req.model_size or "quality", req.cpu_mode)

        if size not in MODEL_IDS[mode]:
            size = "quality"

        model_id = MODEL_IDS[mode][size]

        if mode == "clone":
            if not req.reference_audio:
                raise HTTPException(status_code=400, detail="reference_audio required for clone mode")
            model = self.get_model(model_id, req.cpu_mode)
            ref_path = self.resolve_reference_audio(req.reference_audio, user_dir)
            try:
                with self.generation_lock:
                    wavs, sr = model.generate_voice_clone(
                        text=req.text,
                        language=req.language or "Auto",
                        ref_audio=str(ref_path),
                        x_vector_only_mode=True,
                        **extra_kwargs,
                    )
            except Exception as e:
                self._raise_gpu_busy_if_oom(e)
                raise
        else:
            if not req.voice_description:
                raise HTTPException(status_code=400, detail="voice_description required for design mode")
            model = self.get_model(model_id, req.cpu_mode)
            try:
                with self.generation_lock:
                    wavs, sr = model.generate_voice_design(
                        text=req.text,
                        language=req.language or "Auto",
                        instruct=req.voice_description,
                        **extra_kwargs,
                    )
            except Exception as e:
                self._raise_gpu_busy_if_oom(e)
                raise

        return wavs[0], sr, model_id, size


app = FastAPI(title="qwen3-tts-api", version="0.6.0")
svc = QwenService()
start_time = time.time()


def cuda_memory_snapshot():
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_mb": int(free_bytes / (1024 * 1024)),
            "total_mb": int(total_bytes / (1024 * 1024)),
            "allocated_mb": int(torch.cuda.memory_allocated() / (1024 * 1024)),
            "reserved_mb": int(torch.cuda.memory_reserved() / (1024 * 1024)),
            "max_allocated_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        }
    except Exception as e:
        return {"error": str(e)}


def cleanup_generation_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Text normalization + chunking
# ---------------------------------------------------------------------------
MAX_CHUNK_CHARS = int(os.getenv("QWEN_MAX_CHUNK_CHARS", "160"))
STREAM_PREBUFFER_CHUNKS = max(1, int(os.getenv("QWEN_STREAM_PREBUFFER_CHUNKS", "1")))
TRIM_CHUNK_SILENCE = os.getenv("QWEN_TRIM_CHUNK_SILENCE", "1").lower() not in {"0", "false", "no"}
TRIM_SILENCE_THRESHOLD = float(os.getenv("QWEN_TRIM_SILENCE_THRESHOLD", "0.004"))
TRIM_SILENCE_RELATIVE_THRESHOLD = float(os.getenv("QWEN_TRIM_SILENCE_RELATIVE_THRESHOLD", "0.02"))
TRIM_SILENCE_KEEP_MS = int(os.getenv("QWEN_TRIM_SILENCE_KEEP_MS", "60"))
CHUNK_FADE_MS = int(os.getenv("QWEN_CHUNK_FADE_MS", "8"))
STREAM_WAV_CHANNELS = 1
STREAM_WAV_BITS_PER_SAMPLE = 16


def wav_stream_header(sample_rate: int) -> bytes:
    """Return a WAV header suitable for HTTP chunked streaming.

    The final data length is unknown, so sizes are set to 0xffffffff. Most
    browsers/players accept this for live-ish WAV streams.
    """
    byte_rate = sample_rate * STREAM_WAV_CHANNELS * STREAM_WAV_BITS_PER_SAMPLE // 8
    block_align = STREAM_WAV_CHANNELS * STREAM_WAV_BITS_PER_SAMPLE // 8
    return (
        b"RIFF" + (0xFFFFFFFF).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + STREAM_WAV_CHANNELS.to_bytes(2, "little")
        + int(sample_rate).to_bytes(4, "little")
        + int(byte_rate).to_bytes(4, "little")
        + int(block_align).to_bytes(2, "little")
        + STREAM_WAV_BITS_PER_SAMPLE.to_bytes(2, "little")
        + b"data" + (0xFFFFFFFF).to_bytes(4, "little")
    )


def audio_to_float_mono(wav) -> np.ndarray:
    data = np.asarray(wav, dtype=np.float32)
    if data.ndim > 1:
        data = data.reshape(-1)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def postprocess_chunk_audio(wav, sample_rate: int) -> np.ndarray:
    data = audio_to_float_mono(wav)
    if not TRIM_CHUNK_SILENCE or data.size == 0:
        return data

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return data

    threshold = max(TRIM_SILENCE_THRESHOLD, peak * TRIM_SILENCE_RELATIVE_THRESHOLD)
    voiced = np.flatnonzero(np.abs(data) > threshold)
    if voiced.size == 0:
        return data

    keep = int(sample_rate * TRIM_SILENCE_KEEP_MS / 1000)
    start = max(0, int(voiced[0]) - keep)
    end = min(data.size, int(voiced[-1]) + keep)
    trimmed = data[start:end].copy()

    fade = min(int(sample_rate * CHUNK_FADE_MS / 1000), trimmed.size // 2)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        trimmed[:fade] *= ramp
        trimmed[-fade:] *= ramp[::-1]

    return trimmed


def float_wav_to_pcm16_bytes(wav) -> bytes:
    data = audio_to_float_mono(wav)
    data = np.clip(data, -1.0, 1.0)
    return (data * 32767.0).astype("<i2", copy=False).tobytes()


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
        "device": model_status["device"],
        "uptime_s": int(time.time() - start_time),
        "cuda": cuda_memory_snapshot(),
    }


@app.get("/readyz")
def readyz():
    if model_status["state"] == "ready":
        return {"ready": True, "model_id": model_status["model_id"], "device": model_status["device"]}
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
        "current_device": svc._model_device,
        "supported_languages": SUPPORTED_LANGUAGES,
        "generation": {
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "stream_prebuffer_chunks": STREAM_PREBUFFER_CHUNKS,
            "trim_chunk_silence": TRIM_CHUNK_SILENCE,
            "trim_silence_keep_ms": TRIM_SILENCE_KEEP_MS,
            "chunk_fade_ms": CHUNK_FADE_MS,
            "max_new_tokens_hard_cap": svc.max_new_tokens,
            "max_new_tokens_per_char": svc.max_new_tokens_per_char,
            "max_new_tokens_buffer": svc.max_new_tokens_buffer,
            "non_streaming_mode": svc.non_streaming_mode,
            "default_device": svc.device,
            "cpu_mode_supported": True,
            "cpu_dtype": svc.cpu_dtype,
            "cpu_attn_impl": svc.cpu_attn_impl,
            "speech_tokenizer_device": svc.speech_tokenizer_device,
            "speaker_encoder_device": svc.speaker_encoder_device,
            "qwen_memory_patch": QWEN_MEMORY_PATCH_ENABLED,
        },
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    svc.hard_reset()

    response = {"ok": True, "state": "idle", "model_id": None, "device": None}
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


@app.post("/api/tts/audio-stream")
def tts_audio_stream(req: TTSRequest):
    """Stream audio bytes as they are generated chunk-by-chunk.

    This is real HTTP audio streaming at the API layer: each text chunk is
    synthesized, immediately converted to PCM16, yielded, then released. It
    avoids holding the full utterance in RAM/VRAM like /api/tts and the SSE
    endpoint do. The qwen-tts 0.1.1 library itself does not expose token-level
    audio frames, so chunk boundaries are currently the streaming granularity.
    """
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    audio_format = req.audio_format.lower().strip()
    if audio_format != "wav":
        raise HTTPException(status_code=400, detail="audio_format must be wav for audio-stream endpoint")

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)
    except Exception:
        svc.unload_model()
        raise

    def build_stream_payload(i: int, chunk: str, sr_final: Optional[int]):
        t_chunk = time.time()
        chunk_req = req.model_copy(update={"text": chunk})
        wav, sr, used_model_id, used_size = svc.synthesize(
            chunk_req,
            user_dir,
            extra_kwargs={**base_extra_kwargs, "non_streaming_mode": False},
        )

        raw_samples = int(np.asarray(wav).size)
        wav = postprocess_chunk_audio(wav, int(sr))
        trimmed_samples = int(wav.size)
        audio_ms = int((trimmed_samples / max(int(sr), 1)) * 1000)
        pcm = float_wav_to_pcm16_bytes(wav)
        del wav
        cleanup_generation_memory()

        if sr_final is None:
            sr_final = int(sr)
            payload = wav_stream_header(sr_final) + pcm
        elif int(sr) != sr_final:
            raise RuntimeError(f"sample rate changed during stream: {sr_final} -> {sr}")
        else:
            payload = pcm

        print(json.dumps({
            "event": "tts_audio_stream_chunk",
            "chunk": i + 1,
            "of": n_chunks,
            "chars": len(chunk),
            "bytes": len(payload),
            "audio_ms": audio_ms,
            "raw_samples": raw_samples,
            "trimmed_samples": trimmed_samples,
            "rtf": round((time.time() - t_chunk) / max(audio_ms / 1000, 0.001), 2),
            "t_chunk_ms": int((time.time() - t_chunk) * 1000),
            "cuda": cuda_memory_snapshot(),
        }), flush=True)
        return payload, sr_final, used_model_id, used_size

    # Generate one or more chunks before sending HTTP headers. If this fails,
    # the client gets a proper JSON 500 instead of an apparently successful
    # stream with zero bytes. Setting QWEN_STREAM_PREBUFFER_CHUNKS > 1 trades
    # first-audio latency for fewer playback underruns/gaps on slow hardware.
    prebuffer_count = min(STREAM_PREBUFFER_CHUNKS, n_chunks)
    prebuffered_payloads: list[bytes] = []
    sr_final: Optional[int] = None
    used_model_id = ""
    used_size = ""
    try:
        for i, chunk in enumerate(chunks[:prebuffer_count]):
            payload, sr_final, used_model_id, used_size = build_stream_payload(i, chunk, sr_final)
            prebuffered_payloads.append(payload)
    except HTTPException:
        svc.unload_model()
        raise
    except Exception as e:
        cleanup_generation_memory()
        print(json.dumps({
            "event": "tts_audio_stream_first_chunk_error",
            "detail": str(e),
            "chunks": n_chunks,
            "prebuffer_chunks": prebuffer_count,
            "cuda": cuda_memory_snapshot(),
        }), flush=True)
        svc.unload_model()
        raise HTTPException(status_code=500, detail=f"tts audio stream failed before first bytes: {e}")

    def audio_iter():
        nonlocal sr_final, used_model_id, used_size
        t0 = time.time()
        try:
            for payload in prebuffered_payloads:
                yield payload
            for i, chunk in enumerate(chunks[prebuffer_count:], start=prebuffer_count):
                payload, sr_final, used_model_id, used_size = build_stream_payload(i, chunk, sr_final)
                yield payload
                del payload

            print(json.dumps({
                "event": "tts_audio_stream_done",
                "mode": req.mode,
                "model_size": used_size,
                "chars": len(req.text or ""),
                "chunks": n_chunks,
                "prebuffer_chunks": prebuffer_count,
                "model_id": used_model_id,
                "t_total_ms": int((time.time() - t0) * 1000),
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
        except Exception as e:
            cleanup_generation_memory()
            print(json.dumps({
                "event": "tts_audio_stream_error",
                "detail": str(e),
                "chunks": n_chunks,
                "cuda": cuda_memory_snapshot(),
            }), flush=True)
            raise
        finally:
            # A StreamingResponse may end normally, fail, or be closed when the
            # client disconnects. In every case, release the model from VRAM.
            svc.unload_model()

    return StreamingResponse(
        audio_iter(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Disposition": 'inline; filename="tts-stream.wav"',
        },
    )


@app.post("/api/tts/stream")
def tts_stream(req: TTSRequest):
    req = apply_voice_preset(req)
    user_dir = user_audio_dir(req.user or "default")
    audio_format = req.audio_format.lower().strip()
    if audio_format != "wav":
        raise HTTPException(status_code=400, detail="audio_format must be wav for stream endpoint")

    chunks = chunk_text(req.text)
    n_chunks = len(chunks)
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)
    except Exception:
        svc.unload_model()
        raise

    q: queue.Queue = queue.Queue()

    def run_generation():
        try:
            wav_arrays: list[np.ndarray] = []
            sr_final = 24000
            total_tokens = 0

            for chunk_idx, chunk in enumerate(chunks):
                chunk_estimated = max(1, svc._max_new_tokens_for_text(chunk))
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
                        **base_extra_kwargs,
                        "logits_processor": LogitsProcessorList([ProgressProcessor()]),
                    },
                )
                wav = postprocess_chunk_audio(wav, int(sr))
                wav_arrays.append(np.asarray(wav, dtype=np.float32))
                del wav
                cleanup_generation_memory()
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
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            payload = {"detail": detail}
            if isinstance(e, HTTPException):
                payload["status_code"] = e.status_code
            q.put(("error", payload))
        finally:
            svc.unload_model()

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
    try:
        base_extra_kwargs = svc.prepare_request_extra_kwargs(req, user_dir)

        t1 = time.time()
        wav_arrays: list[np.ndarray] = []
        sr_final: int = 24000
        used_model_id: str = ""
        used_size: str = ""

        for i, chunk in enumerate(chunks):
            chunk_req = req.model_copy(update={"text": chunk})
            wav, sr, used_model_id, used_size = svc.synthesize(
                chunk_req, user_dir, extra_kwargs=base_extra_kwargs
            )
            wav = postprocess_chunk_audio(wav, int(sr))
            wav_arrays.append(np.asarray(wav, dtype=np.float32))
            del wav
            cleanup_generation_memory()
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
    finally:
        svc.unload_model()
