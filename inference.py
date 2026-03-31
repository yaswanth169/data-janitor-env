"""
Baseline inference script for data-janitor-env.

Required environment variables:
    API_BASE_URL   LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier for inference
    HF_TOKEN       Hugging Face / API key
    ENV_BASE_URL   Environment server URL (default: http://localhost:7860)
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, Optional

from openai import OpenAI

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

TASK_IDS = ["fix_basics", "normalize_chaos", "pipeline_merge"]
TEMPERATURE = 0.1
MAX_TOKENS = 512


SYSTEM_PROMPT = textwrap.dedent("""\
    You are a data cleaning agent. You receive a description of a dirty dataset
    and must issue cleaning commands to transform it into the target schema.

    Respond with exactly one JSON object per turn:
    {"command": "...", "column": "..." or null, "params": {...}}

    Available commands:
    - inspect: Look at a column or dataset. params: {}
    - drop_duplicates: Remove duplicate rows. params: {"subset": ["col1","col2"]} (optional)
    - fill_missing: Fill nulls. params: {"strategy": "mean"|"median"|"mode"|"constant", "value": ...}
    - drop_nulls: Drop rows where column is null. params: {}
    - convert_type: Cast column. params: {"target_type": "int"|"float"|"str"}
    - normalize_text: Text ops. params: {"operation": "trim"|"lower"|"upper"|"title"|"regex_replace", "pattern": "...", "replacement": "..."}
    - standardize_date: Parse dates to ISO. params: {"format": "%Y-%m-%d"}
    - standardize_phone: Normalize phones to (XXX) XXX-XXXX. params: {}
    - rename_column: Rename. params: {"new_name": "..."}
    - map_values: Remap values. params: {"mapping": {"old": "new", ...}}
    - filter_rows: Remove rows matching condition. params: {"operator": "=="|"!="|">"|"<"|">="|"<=", "value": ...}
    - split_column: Split by delimiter. params: {"delimiter": ",", "new_columns": ["a","b"]}
    - merge_columns: Combine columns. params: {"columns": ["a","b"], "new_column": "c", "separator": " "}
    - join: Merge secondary dataset. params: {"on": "col", "how": "inner"|"left"}
    - add_column: Compute column. params: {"expression": "col_a * col_b"} or {"value": constant}
    - submit: Finalize and get score. params: {}

    Strategy:
    1. Read the task description and issues carefully
    2. Fix the most impactful issues first (duplicates, type errors)
    3. Work through remaining issues systematically
    4. Submit when quality is high or steps are running low

    Output ONLY valid JSON. No explanations.
""")


def build_user_prompt(obs: Dict[str, Any]) -> str:
    parts = [
        f"Task: {obs.get('task_description', '')}",
        f"Rows: {obs.get('row_count', 0)} | Quality: {obs.get('quality_score', 0):.2%}",
        f"Step: {obs.get('steps_taken', 0)}/{obs.get('max_steps', 20)}",
        "",
        "Target schema:",
        json.dumps(obs.get("target_schema", {}), indent=2),
        "",
        "Current issues:",
    ]
    for issue in obs.get("issues", []):
        parts.append(f"  - {issue}")

    parts.append("")
    parts.append("Schema:")
    for col_info in obs.get("schema_info", []):
        if isinstance(col_info, dict):
            parts.append(
                f"  {col_info['name']}: {col_info['dtype']} "
                f"({col_info['null_count']} nulls, {col_info['unique_count']} unique) "
                f"sample={col_info.get('sample_values', [])[:3]}"
            )

    parts.append("")
    parts.append("Sample rows:")
    for row in obs.get("sample_rows", [])[:3]:
        parts.append(f"  {json.dumps(row, default=str)}")

    sec = obs.get("secondary_data_info")
    if sec:
        parts.append("")
        parts.append(f"Secondary dataset: {sec.get('row_count', 0)} rows")
        parts.append(f"  Columns: {sec.get('columns', [])}")
        for row in sec.get("sample_rows", [])[:2]:
            parts.append(f"  {json.dumps(row, default=str)}")

    if obs.get("message"):
        parts.append("")
        parts.append(f"Last result: {obs['message']}")

    return "\n".join(parts)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "command" in parsed:
                return {
                    "command": parsed["command"],
                    "column": parsed.get("column"),
                    "params": parsed.get("params", {}),
                }
        except json.JSONDecodeError:
            pass

    match = re.search(r'"command"\s*:\s*"(\w+)"', text)
    if match:
        return {"command": match.group(1), "column": None, "params": {}}

    return None


async def run_task_ws(task_id: str, client: OpenAI) -> float:
    """Run a single task over WebSocket (stateful session)."""
    import websockets

    ws_url = ENV_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws"

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    async with websockets.connect(ws_url) as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
        raw = json.loads(await ws.recv())
        payload = raw.get("data", raw)
        obs = payload.get("observation", payload)
        done = payload.get("done", False)

        print(f"  Rows: {obs.get('row_count')} | Quality: {obs.get('quality_score', 0):.2%}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        max_steps = obs.get("max_steps", 20)

        for step in range(max_steps):
            if done:
                break

            user_prompt = build_user_prompt(obs)
            messages.append({"role": "user", "content": user_prompt})

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            assistant_text = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})

            action = parse_action(assistant_text)
            if not action:
                action = {"command": "submit", "column": None, "params": {}}
                print(f"  Step {step + 1}: Failed to parse, submitting")
            else:
                print(f"  Step {step + 1}: {action['command']}({action.get('column', '')})")

            await ws.send(json.dumps({"type": "step", "data": action}))
            raw = json.loads(await ws.recv())
            payload = raw.get("data", raw)
            obs = payload.get("observation", payload)
            done = payload.get("done", False)
            reward = payload.get("reward")

            print(f"    Quality: {obs.get('quality_score', 0):.2%} | Reward: {reward}")

            if len(messages) > 12:
                messages = messages[:2] + messages[-8:]

    final_score = obs.get("quality_score", 0.0)
    if reward is not None and done:
        final_score = reward

    print(f"  Final score: {final_score:.4f}")
    return final_score


async def run_all():
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is required.")
        return
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is required.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores: Dict[str, float] = {}

    for task_id in TASK_IDS:
        try:
            score = await run_task_ws(task_id, client)
            scores[task_id] = score
        except Exception as e:
            print(f"  ERROR on {task_id}: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        bar = "#" * int(score * 40)
        print(f"  {task_id:20s} [{bar:<40}] {score:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  {'Average':20s}                                          {avg:.4f}")


def main():
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
