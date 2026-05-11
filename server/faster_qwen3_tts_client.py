#!/usr/bin/env python3
"""
OpenAI-compatible TTS client for faster-qwen3-tts.

Posts text-to-speech requests to a running openai_server.py instance and saves/plays the audio.

Usage:
    # Basic: write to file
    python server/faster_qwen3_tts_client.py \
        --text "Hello, how are you?" \
        --output output.wav

    # True HTTP streaming + live playback (no file)
    python server/faster_qwen3_tts_client.py \
        --text "Hello, this should stream in real time" \
        --format pcm \
        --stream --play

    # With custom instruct (emotion, style, etc.)
    python server/faster_qwen3_tts_client.py \
        --text "Hello, how are you?" \
        --instruct "gender: female. pitch: low female pitch. speed: deliberate pace, starting slow. age: 32. clarity: medium clarity. accent: American English. texture: slightly gravelly. tone: sad. personality: depressed. Emotion: Spoke with a very sad and tearful voice. Start sad and end very upset and sobbing" \
        --format pcm \
        --stream --play \
        --output output_sad.pcm

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
import time

import numpy as np
import requests

from audio import StreamPlayer


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

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Consume response as a stream instead of buffering the full body",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play streamed audio in real time (wav/pcm only)",
    )

    # Output
    parser.add_argument(
        "--output",
        default=None,
        help="Output audio file path",
    )

    args = parser.parse_args()

    if not args.output and not args.play:
        print("ERROR: set at least one sink: --output and/or --play", file=sys.stderr)
        sys.exit(2)

    if args.play and args.format == "mp3":
        print("ERROR: --play supports only --format wav or pcm", file=sys.stderr)
        sys.exit(2)

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
            timeout=(30, 300),
            stream=args.stream,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Failed to connect to server at {url}", file=sys.stderr)
        print(f"  Is the server running? (python server/openai_server.py ...)", file=sys.stderr)
        print(f"  Original error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        print("ERROR: Request timed out", file=sys.stderr)
        print(f"  Original error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError:
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

    output_file = None
    player = None
    total_bytes = 0
    started_at = time.perf_counter()
    stream_receive_done = None
    success = False

    try:
        if args.output:
            output_file = open(args.output, "wb")

        if args.play:
            player = StreamPlayer()

        if args.stream:
            # Stream chunk-by-chunk from HTTP response.
            wav_header_skip = 44 if args.format == "wav" else 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total_bytes += len(chunk)

                if output_file:
                    output_file.write(chunk)

                if player:
                    payload_chunk = chunk
                    if wav_header_skip > 0:
                        if len(payload_chunk) <= wav_header_skip:
                            wav_header_skip -= len(payload_chunk)
                            continue
                        payload_chunk = payload_chunk[wav_header_skip:]
                        wav_header_skip = 0

                    # Convert little-endian PCM16 to float32 in [-1, 1] for StreamPlayer.
                    pcm16 = np.frombuffer(payload_chunk, dtype=np.int16)
                    if pcm16.size:
                        audio = (pcm16.astype(np.float32) / 32768.0)
                        player(audio, sample_rate=24000)
        else:
            data = response.content
            total_bytes = len(data)

            if output_file:
                output_file.write(data)

            if player:
                payload_chunk = data[44:] if args.format == "wav" else data
                pcm16 = np.frombuffer(payload_chunk, dtype=np.int16)
                if pcm16.size:
                    audio = (pcm16.astype(np.float32) / 32768.0)
                    player(audio, sample_rate=24000)

        stream_receive_done = time.perf_counter()
        elapsed_receive = stream_receive_done - started_at
        if args.output:
            print(f"Saved {total_bytes} bytes to {args.output}")
        print(f"Received stream in {elapsed_receive:.2f}s")
        success = True

    except IOError as e:
        print(f"ERROR: Failed to write to {args.output}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR during stream processing: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if output_file:
            output_file.close()
        if player:
            player.close()
        response.close()
        if success:
            if args.play:
                print("Playback completed")
            total_elapsed = time.perf_counter() - started_at
            if stream_receive_done is not None:
                drain_time = total_elapsed - (stream_receive_done - started_at)
                print(f"Playback drain time: {drain_time:.2f}s")
            print(f"Total elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
