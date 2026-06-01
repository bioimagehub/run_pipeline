#!/usr/bin/env python3
"""
Prepare an offline teaching lecture using Ollama.

Run this at home (before driving) to generate a structured lecture on any topic.
The output JSON file is then used by teach_chat.py for offline driving sessions.

Usage:
    python teach_prepare.py --topic "sound waves" --duration 60 --output-folder ./lectures
    python teach_prepare.py --topic "quantum entanglement" --duration 30 --model phi3:mini
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime


def call_ollama_generate(url: str, model: str, prompt: str) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
        return data["response"]


def build_lecture_prompt(topic: str, duration: int) -> str:
    intro_min = max(3, duration // 15)
    core_min = duration - intro_min * 2
    return f"""Create a comprehensive {duration}-minute spoken lecture about "{topic}".

Structure the lecture as follows:

## INTRODUCTION ({intro_min} minutes)
- Start with a surprising or relatable hook about {topic}
- Explain why understanding {topic} is useful or fascinating
- Give a brief roadmap of what will be covered

## CORE CONTENT ({core_min} minutes)
Divide into 4-6 clearly titled sections. For each section provide:
- A short section title
- 3-5 detailed talking points with concrete examples and analogies
- At least one real-world application or surprising fact

## SUMMARY ({intro_min} minutes)
- Recap the 4-6 most important ideas in plain, memorable language
- End with a thought-provoking question or takeaway

## QUIZ (8-10 questions)
List questions and their answers to test understanding of {topic}.

## GLOSSARY (10-15 terms)
List the most important technical terms with clear, one-sentence definitions.

Write in conversational, engaging language suitable for audio — the student will be listening while driving, not reading. Be detailed and thorough: this document is the sole teaching reference during the drive."""


def slugify(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text


def build_system_prompt(topic: str, duration: int, lecture_content: str) -> str:
    return (
        f"You are an expert, engaging teacher conducting a {duration}-minute lecture "
        f'about "{topic}". Here is your complete prepared lecture material:\n\n'
        f"{lecture_content}\n\n"
        "INSTRUCTIONS FOR THIS CHAT SESSION:\n"
        "- Guide the student conversationally through the lecture, section by section\n"
        "- Keep each response to 3-5 sentences — the student is listening while driving\n"
        "- After finishing a section, ask one short check-in question to confirm understanding\n"
        "- If the student answers a check-in question, briefly confirm or clarify, then move on\n"
        "- If asked a question, answer using the lecture material; say if something is outside the lecture scope\n"
        "- Do not rush — let the student set the pace by responding or pressing Enter to continue\n"
        "- Do not invent facts not present in the lecture material above"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an offline Ollama lecture before a drive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--topic", required=True, help="Topic to prepare a lecture about")
    parser.add_argument("--duration", type=int, default=60, help="Lecture duration in minutes")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server base URL")
    parser.add_argument("--output-folder", default="./lectures", help="Folder to save the lecture JSON file")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    print(f"Preparing a {args.duration}-minute lecture about '{args.topic}' using model '{args.model}'...")
    print("This may take 1-3 minutes. Please wait.\n")

    prompt = build_lecture_prompt(args.topic, args.duration)

    try:
        lecture_content = call_ollama_generate(args.ollama_url, args.model, prompt)
    except urllib.error.URLError as exc:
        print(f"ERROR: Could not connect to Ollama at {args.ollama_url}")
        print(f"Make sure Ollama is running and the model '{args.model}' is pulled.")
        print(f"Details: {exc}")
        sys.exit(1)

    system_prompt = build_system_prompt(args.topic, args.duration, lecture_content)

    lecture = {
        "topic": args.topic,
        "duration_minutes": args.duration,
        "model": args.model,
        "ollama_url": args.ollama_url,
        "generated_at": datetime.now().isoformat(),
        "system_prompt": system_prompt,
        "lecture_content": lecture_content,
    }

    slug = slugify(args.topic)
    output_path = os.path.join(args.output_folder, f"{slug}.json")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(lecture, fh, indent=2, ensure_ascii=False)

    print(f"Lecture saved to: {output_path}")
    print(f"\nTo start your driving session, run:")
    print(f"  python teach_chat.py --lecture {output_path}")


if __name__ == "__main__":
    main()
