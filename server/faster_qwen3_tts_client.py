#!/usr/bin/env python3
"""
OpenAI-compatible TTS client for faster-qwen3-tts.

Posts text-to-speech requests to a running openai_server.py instance and saves/plays the audio.

Usage:
    # Basic: write to file
    python server/faster_qwen3_tts_client.py \
        --text "Hello, how are you?" \
        --output output.wav

    # With custom instruct (emotion, style, etc.)
    python server/faster_qwen3_tts_client.py \
        --text "Hello, how are you?" \
        --instruct "gender: female. pitch: low female pitch. speed: deliberate pace, starting slow. age: 32. clarity: medium clarity. accent: American English. texture: slightly gravelly. tone: sad. personality: depressed. Emotion: Spoke with a very sad and tearful voice. Start sad and end very upset and sobbing" \
        --output output_sad.wav

    # Custom voice/speaker
    python server/faster_qwen3_tts_client.py \
        --text "Hello world" \
        --speaker aiden \
        --output output.wav

    # Different response format
    python server/faster_qwen3_tts_client.py \
        --text "Hello" \
        --format pcm \
        --output output.pcm
"""

import argparse
import sys
import requests
from typing import Optional


def main():
    parser = argparse.ArgumentParser(
        prog="faster_qwen3_tts_client",
        description="OpenAI-compatible TTS client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Connection
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )

    # Request parameters
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="Target language (default: Auto)",
    )
    parser.add_argument(
        "--speaker",
        default="default",
        help="Speaker ID or voice name (default: default)",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional style instruction (emotion, accent, etc.)",
    )
    parser.add_argument(
        "--format",
        choices=["wav", "pcm", "mp3"],
        default="wav",
        help="Audio format (default: wav)",
    )
    parser.add_argument(
        "--model",
        default="tts-1",
        help="Model ID (default: tts-1, ignored by server)",
    )

    # Output
    parser.add_argument(
        "--output",
        required=True,
        help="Output audio file path",
    )

    args = parser.parse_args()

    # Build request
    url = f"http://{args.host}:{args.port}/v1/audio/speech"
    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.speaker,
        "response_format": args.format,
    }
    if args.instruct:
        payload["instruct"] = args.instruct

    print(f"Posting to {url}")
    print(f"  Text: {args.text[:60]}{'...' if len(args.text) > 60 else ''}")
    if args.instruct:
        print(f"  Instruct: {args.instruct[:60]}{'...' if len(args.instruct) > 60 else ''}")
    print(f"  Speaker: {args.speaker}")
    print(f"  Language: {args.language}")
    print(f"  Format: {args.format}")

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=300,  # 5 minutes for generation + streaming
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Failed to connect to server at {url}", file=sys.stderr)
        print(f"  Is the server running? (python server/openai_server.py ...)", file=sys.stderr)
        print(f"  Original error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        print(f"ERROR: Request timed out after 5 minutes", file=sys.stderr)
        print(f"  Original error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: Server returned {response.status_code}", file=sys.stderr)
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text
        print(f"  {error_detail}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Write response to file
    try:
        with open(args.output, "wb") as f:
            f.write(response.content)
        print(f"Saved {len(response.content)} bytes to {args.output}")
    except IOError as e:
        print(f"ERROR: Failed to write to {args.output}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
