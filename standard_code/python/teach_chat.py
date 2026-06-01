#!/usr/bin/env python3
"""
Offline teaching chat for driving sessions.

Loads a pre-generated lecture (created by teach_prepare.py) and starts an
interactive conversation with an Ollama model. Works fully offline — no
internet connection needed once the lecture file and model are ready.

NOTE: This script is interactive and must be run directly from a terminal.
It cannot be run via run_pipeline.exe because the pipeline runner does not
wire stdin to subprocesses.

Usage:
    python teach_chat.py --lecture ./lectures/sound_waves.json

Controls during chat:
    Press Enter (empty input) — teacher continues to the next part of the lecture
    Type a question        — ask anything about the topic
    Type 'hint'            — get a simpler explanation of the last point
    Type 'quiz'            — ask the teacher to quiz you
    Type 'summary'         — get a summary of what has been covered so far
    Type 'quit' / 'exit'   — end the session
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def chat_ollama(url: str, model: str, messages: list) -> str:
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        f"{url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data["message"]["content"]


def print_banner(topic: str, duration: int, model: str) -> None:
    width = 62
    print("\n" + "=" * width)
    print(f"  TEACHING ASSISTANT — {topic.upper()}")
    print(f"  {duration}-minute lecture  |  Model: {model}")
    print("=" * width)
    print("  Enter  → teacher continues")
    print("  hint   → simpler explanation")
    print("  quiz   → teacher quizzes you")
    print("  summary → recap so far")
    print("  quit   → end session")
    print("=" * width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline driving-mode teaching chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: This script is interactive and must be run directly from a terminal.
It cannot be run via run_pipeline.exe because the pipeline runner does not
wire stdin to subprocesses. See pipeline_configs/teach_chat.yaml for details.

Example:
    python standard_code/python/teach_chat.py --lecture lectures/sound_waves.json
        """,
    )
    parser.add_argument(
        "--lecture",
        required=True,
        help="Path to the lecture JSON file created by teach_prepare.py",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama server URL (overrides the URL stored in the lecture file)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.exists(args.lecture):
        logger.error("Lecture file not found: %s", args.lecture)
        print(f"ERROR: Lecture file not found: {args.lecture}")
        sys.exit(1)

    with open(args.lecture, "r", encoding="utf-8") as fh:
        lecture = json.load(fh)

    topic = lecture.get("topic", "Unknown topic")
    duration = lecture.get("duration_minutes", 60)
    model = lecture.get("model", "llama3.2")
    ollama_url = args.ollama_url or lecture.get("ollama_url", "http://localhost:11434")
    system_prompt = lecture["system_prompt"]

    print_banner(topic, duration, model)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Kick off with the introduction
    messages.append({"role": "user", "content": "Please begin with the introduction."})
    print("Teacher: ", end="", flush=True)
    try:
        reply = chat_ollama(ollama_url, model, messages)
    except urllib.error.URLError as exc:
        logger.error("Could not connect to Ollama at %s: %s", ollama_url, exc)
        print(f"\nERROR: Could not connect to Ollama at {ollama_url}")
        print(f"Make sure Ollama is running and model '{model}' is available.")
        sys.exit(1)

    print(reply)
    messages.append({"role": "assistant", "content": reply})

    # Main chat loop
    while True:
        print()
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended. Drive safe!")
            break

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nSession ended. Drive safe!")
            break

        # Map shortcut commands to natural-language prompts
        shortcuts = {
            "": "Continue to the next part of the lecture.",
            "hint": "Please explain that last point again using a simpler analogy.",
            "quiz": "Quiz me on what we have covered so far with one question.",
            "summary": "Give me a brief summary of everything we have covered so far.",
        }
        effective_input = shortcuts.get(user_input.lower(), user_input)

        messages.append({"role": "user", "content": effective_input})
        print("\nTeacher: ", end="", flush=True)

        try:
            reply = chat_ollama(ollama_url, model, messages)
        except urllib.error.URLError as exc:
            logger.error("Ollama request failed: %s", exc)
            print(f"\nERROR: {exc}")
            messages.pop()  # remove the unsent user message
            continue

        print(reply)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
