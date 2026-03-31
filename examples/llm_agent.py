"""
Data Janitor Env — LLM Agent (text mode)
=========================================

Uses the "text" mode gym wrapper to run any LLM as a data cleaning agent.
The LLM receives a natural-language description of the dirty dataset and
outputs JSON cleaning commands.

Compatible with:
  - OpenAI, HuggingFace Inference API, Ollama, LM Studio (any OpenAI-compat API)
  - TRL / GRPO for reinforcement fine-tuning of open-source LLMs

Run:
    export HF_TOKEN=your_token
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    python examples/llm_agent.py

The environment runs fully in-process — no server needed.
"""

from __future__ import annotations

import json
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_env import DataJanitorGymEnv

# ── LLM configuration ──────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY  = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL    = os.getenv("MODEL_NAME", "")

SYSTEM_PROMPT = """\
You are a data cleaning agent. You receive a description of a dirty dataset
and must issue cleaning commands to transform it into the target schema.

Respond with exactly one JSON object per turn:
{"command": "...", "column": "..." or null, "params": {...}}

Available commands: inspect, drop_duplicates, fill_missing, drop_nulls,
convert_type, normalize_text, standardize_date, standardize_phone,
rename_column, map_values, filter_rows, split_column, merge_columns,
join, add_column, submit

Output ONLY valid JSON. No explanations."""


def call_llm(messages: list, client) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""


def parse_json_action(text: str) -> str:
    """Extract the first JSON object from LLM output."""
    text = text.strip()
    try:
        start = text.index("{")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    parsed = json.loads(candidate)
                    if "command" in parsed:
                        return json.dumps({
                            "command": parsed["command"],
                            "column": parsed.get("column"),
                            "params": parsed.get("params", {}),
                        })
    except (ValueError, json.JSONDecodeError):
        pass
    # Fallback: extract just the command name
    m = re.search(r'"command"\s*:\s*"(\w+)"', text)
    if m:
        return json.dumps({"command": m.group(1), "column": None, "params": {}})
    return json.dumps({"command": "submit", "column": None, "params": {}})


def run_task(task_id: str, client) -> float:
    env = DataJanitorGymEnv(task_id=task_id, mode="text")
    obs, info = env.reset()

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")
    print(f"Initial quality: {info['quality_score']:.2%} | Issues: {len(info['issues'])}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(info["max_steps"]):
        messages.append({"role": "user", "content": obs})
        llm_output = call_llm(messages, client)
        messages.append({"role": "assistant", "content": llm_output})

        action = parse_json_action(llm_output)
        cmd = json.loads(action)["command"]
        print(f"  Step {step+1:2d}: {cmd}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"          quality={info['quality_score']:.2%}  Δ={reward:+.4f}")
        if info.get("message"):
            print(f"          {info['message'][:80]}")

        if terminated or truncated:
            break

        # Keep context window manageable
        if len(messages) > 14:
            messages = messages[:2] + messages[-8:]

    final = info["quality_score"]
    print(f"  Final: {final:.4f}")
    env.close()
    return final


def main():
    if not MODEL:
        print("ERROR: set MODEL_NAME env var (e.g. Qwen/Qwen2.5-72B-Instruct)")
        return
    if not API_KEY:
        print("ERROR: set HF_TOKEN env var")
        return

    from openai import OpenAI
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    task_ids = ["fix_basics", "normalize_chaos", "pipeline_merge"]
    scores = {}
    for tid in task_ids:
        try:
            scores[tid] = run_task(tid, client)
        except Exception as e:
            print(f"  ERROR: {e}")
            scores[tid] = 0.0

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    for tid, s in scores.items():
        bar = "#" * int(s * 40)
        print(f"  {tid:20s} [{bar:<40}] {s:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'Average':20s}                                          {avg:.4f}")


if __name__ == "__main__":
    main()
