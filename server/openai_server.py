#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Voice clone server:
    python server/openai_server.py clone \
        --ref-audio voice.wav --ref-text "Reference transcription" \
        --language English

    # Multiple named clone voices from a JSON config:
    python server/openai_server.py clone --voices voices.json

    # CustomVoice server:
    python server/openai_server.py custom \
        --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
        --speaker aiden

    # VoiceDesign server:
    python server/openai_server.py design \
        --model Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign \
        --instruct "Warm, confident narrator"

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import struct
import sys
import threading
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
server_mode: str = "clone"
voices: dict = {}
default_voice: Optional[str] = None
default_speaker: Optional[str] = None
available_speakers: list[str] = []
default_language: str = "Auto"
default_instruct: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "default"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied
    instruct: Optional[str] = None  # optional style instruction or design prompt (works with custom or design, experimental with voice cloning)
    xvec_only: bool = False  # use x-vector-only mode instead of ICL mode. This is for clone mode only


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


def resolve_speaker(requested_voice: str) -> str:
    """Resolve a speaker ID for custom voice mode."""
    candidate = requested_voice.strip() if requested_voice else ""
    if not candidate:
        candidate = default_speaker

    if not candidate:
        raise HTTPException(
            status_code=400,
            detail="Custom mode requires a speaker. Set request.voice or start the server with --speaker.",
        )

    if available_speakers and candidate not in available_speakers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Speaker {candidate!r} is not available. "
                f"Available speakers: {available_speakers}"
            ),
        )

    return candidate


def resolve_request_context(req: SpeechRequest) -> tuple[Optional[dict], str, Optional[str], Optional[str]]:
    """Resolve mode-specific request context as (voice_cfg, language, instruct, speaker)."""
    if server_mode == "clone":
        voice_cfg = resolve_voice(req.voice)
        return voice_cfg, voice_cfg.get("language", default_language), req.instruct or default_instruct, None

    if server_mode == "custom":
        speaker = resolve_speaker(req.voice)
        return None, default_language, req.instruct or default_instruct, speaker

    instruct = req.instruct or default_instruct
    if not instruct:
        raise HTTPException(
            status_code=400,
            detail="Design mode requires instruct in the request or at server startup.",
        )
    return None, default_language, instruct, None


def maybe_add_instruct(kwargs: dict, instruct: Optional[str]) -> dict:
    """Add instruct only when it is explicitly provided."""
    if instruct is not None:
        kwargs["instruct"] = instruct
    return kwargs


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(
    text: str,
    language: str,
    instruct: Optional[str],
    voice_cfg: Optional[dict],
    xvec_only: bool,
    speaker: Optional[str],
) -> AsyncGenerator[bytes, None]:
    """Run the mode-specific streaming generator in a background thread (such as 
    generate_voice_clone_streaming) and yield raw PCM bytes for each chunk as they arrive."""
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                if server_mode == "clone":
                    generator_kwargs = maybe_add_instruct({
                        "text": text,
                        "language": language,
                        "ref_audio": voice_cfg["ref_audio"],
                        "ref_text": voice_cfg.get("ref_text", ""),
                        "xvec_only": xvec_only,
                        "chunk_size": voice_cfg.get("chunk_size", 12),
                        "non_streaming_mode": False,
                    }, instruct)
                    generator = tts_model.generate_voice_clone_streaming(**generator_kwargs)
                elif server_mode == "custom":
                    generator_kwargs = maybe_add_instruct({
                        "text": text,
                        "speaker": speaker,
                        "language": language,
                        "chunk_size": 8,
                        "non_streaming_mode": None,
                    }, instruct)
                    generator = tts_model.generate_custom_voice_streaming(**generator_kwargs)
                else:
                    generator_kwargs = maybe_add_instruct({
                        "text": text,
                        "language": language,
                        "chunk_size": 8,
                        "non_streaming_mode": None,
                    }, instruct)
                    generator = tts_model.generate_voice_design_streaming(**generator_kwargs)

                for chunk, _sr, _timing in generator:
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None, "mode": server_mode}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg, language, instruct, speaker = resolve_request_context(req)
    fmt = req.response_format.lower()

    content_types = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in content_types:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = content_types[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                if server_mode == "clone":
                    generate_kwargs = maybe_add_instruct({
                        "text": req.input,
                        "language": language,
                        "ref_audio": voice_cfg["ref_audio"],
                        "ref_text": voice_cfg.get("ref_text", ""),
                        "xvec_only": req.xvec_only,
                    }, instruct)
                    return tts_model.generate_voice_clone(**generate_kwargs)
                if server_mode == "custom":
                    generate_kwargs = maybe_add_instruct({
                        "text": req.input,
                        "speaker": speaker,
                        "language": language,
                    }, instruct)
                    return tts_model.generate_custom_voice(**generate_kwargs)
                generate_kwargs = maybe_add_instruct({
                    "text": req.input,
                    "language": language,
                }, instruct)
                return tts_model.generate_voice_design(**generate_kwargs)

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE) # stream with unknown data length
        async for raw_chunk in _stream_chunks(
            text=req.input,
            language=language,
            instruct=instruct,
            voice_cfg=voice_cfg,
            xvec_only=req.xvec_only,
            speaker=speaker,
        ):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")

    sub = p.add_subparsers(dest="mode", required=True)

    def add_mode_common(sp, default_model: str):
        sp.add_argument(
            "--model",
            default=os.environ.get("QWEN_TTS_MODEL", default_model),
            help=f"HuggingFace model ID or local path (default: {default_model})",
        )
        sp.add_argument(
            "--language",
            default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
            help="Target language (English, French, Auto, …)",
        )

    sp = sub.add_parser("clone", help="Voice cloning server (reference audio)")
    add_mode_common(sp, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    sp.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    sp.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    sp.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    sp.add_argument(
        "--instruct",
        default=os.environ.get("QWEN_TTS_INSTRUCT") or None,
        help="Default optional instruction when the request omits instruct",
    )

    sp = sub.add_parser("custom", help="CustomVoice server (speaker IDs)")
    add_mode_common(sp, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    sp.add_argument(
        "--speaker",
        default=os.environ.get("QWEN_TTS_SPEAKER"),
        help="Default speaker ID when request.voice is not provided",
    )
    sp.add_argument(
        "--instruct",
        default=os.environ.get("QWEN_TTS_INSTRUCT") or None,
        help="Default optional instruction when the request omits instruct",
    )

    sp = sub.add_parser("design", help="VoiceDesign server (instruction-based)")
    add_mode_common(sp, "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    sp.add_argument(
        "--instruct",
        default=os.environ.get("QWEN_TTS_INSTRUCT") or None,
        help="Default instruction when the request omits instruct",
    )

    return p.parse_args()


def main():
    global tts_model, server_mode, voices, default_voice, default_speaker
    global available_speakers, default_language, default_instruct, SAMPLE_RATE

    args = _parse_args()
    server_mode = args.mode
    default_language = args.language
    default_instruct = getattr(args, "instruct", None)

    # Fail fast on startup configuration that does not require loading a model.
    if server_mode == "clone":
        if not args.voices and not args.ref_audio:
            print(
                "ERROR: clone mode requires --ref-audio <file> or --voices <config.json>",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.voices:
            if not os.path.isfile(args.voices):
                print(f"ERROR: voices config file not found: {args.voices}", file=sys.stderr)
                sys.exit(1)
            try:
                with open(args.voices, encoding="utf-8") as f:
                    voices = json.load(f)
            except Exception as exc:
                print(f"ERROR: failed to read voices config {args.voices}: {exc}", file=sys.stderr)
                sys.exit(1)

            if not isinstance(voices, dict) or not voices:
                print("ERROR: voices config must be a non-empty JSON object", file=sys.stderr)
                sys.exit(1)

            for voice_name, voice_cfg in voices.items():
                if not isinstance(voice_cfg, dict):
                    print(f"ERROR: voice entry {voice_name!r} must be a JSON object", file=sys.stderr)
                    sys.exit(1)
                ref_audio = voice_cfg.get("ref_audio")
                if not ref_audio:
                    print(f"ERROR: voice entry {voice_name!r} is missing ref_audio", file=sys.stderr)
                    sys.exit(1)
                if not os.path.isfile(ref_audio):
                    print(
                        f"ERROR: reference audio not found for voice {voice_name!r}: {ref_audio}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            default_voice = next(iter(voices))
            logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
        else:
            if not os.path.isfile(args.ref_audio):
                print(f"ERROR: reference audio file not found: {args.ref_audio}", file=sys.stderr)
                sys.exit(1)
            voices = {
                "default": {
                    "ref_audio": args.ref_audio,
                    "ref_text": args.ref_text,
                    "language": args.language,
                }
            }
            default_voice = "default"
            logger.info("Using single voice from --ref-audio: %s", args.ref_audio)

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s ...", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )

    detected_model_type = getattr(tts_model.model.model, "tts_model_type", None)
    expected_model_type = {
        "clone": "base",
        "custom": "custom_voice",
        "design": "voice_design",
    }[server_mode]
    if detected_model_type != expected_model_type:
        print(
            f"ERROR: mode {server_mode!r} expects model type {expected_model_type!r}, "
            f"but loaded model reports {detected_model_type!r}",
            file=sys.stderr,
        )
        sys.exit(2)

    if server_mode == "custom":
        default_speaker = args.speaker
        speaker_getter = getattr(tts_model.model, "get_supported_speakers", None)
        if callable(speaker_getter):
            available_speakers = speaker_getter() or []
            if not available_speakers:
                print(
                    "ERROR: custom mode model reported no available speakers",
                    file=sys.stderr,
                )
                sys.exit(1)
        if default_speaker and available_speakers and default_speaker not in available_speakers:
            print(
                f"ERROR: --speaker {default_speaker!r} is not in available speakers: {available_speakers}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not default_speaker and available_speakers:
            default_speaker = available_speakers[0]
        if default_speaker:
            logger.info("Default speaker: %s", default_speaker)
        if available_speakers:
            logger.info("Available speakers: %s", ", ".join(available_speakers))

    SAMPLE_RATE = tts_model.sample_rate
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Server mode: %s", server_mode)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
