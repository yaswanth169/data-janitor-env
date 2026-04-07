"""
Baseline inference script for data-janitor-env.

Required environment variables:
    API_BASE_URL    LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME      Model identifier for inference
    OPENAI_API_KEY  OpenAI-compatible API key (or HF_TOKEN / API_KEY)
    HF_TOKEN        Hugging Face / API key (alternative to OPENAI_API_KEY)
    ENV_BASE_URL    Environment server URL (default: deployed HF Space)
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, Optional

from openai import OpenAI

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://yaswanth169-data-janitor-env.hf.space")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

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
    # Extract outermost JSON object (handles nested braces in params)
    try:
        start = text.index("{")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(text[start : i + 1])
                    if "command" in parsed:
                        # Support both {"command":"x","column":"y"} and
                        # {"command":"x","params":{"column":"y",...}} formats
                        params = parsed.get("params", {})
                        column = parsed.get("column") or params.pop("column", None)
                        return {
                            "command": parsed["command"],
                            "column": column,
                            "params": params,
                        }
                    break
    except (ValueError, json.JSONDecodeError):
        pass

    match = re.search(r'"command"\s*:\s*"(\w+)"', text)
    if match:
        return {"command": match.group(1), "column": None, "params": {}}

    return None


async def _wake_space(http_url: str) -> None:
    """Ping /health until the Space responds (handles cold starts)."""
    import httpx
    for _ in range(12):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{http_url}/health")
                if r.status_code == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(5)


async def run_task_ws(task_id: str, client: OpenAI) -> float:
    """Run a single task over WebSocket with keepalive pings."""
    import websockets

    await _wake_space(ENV_BASE_URL)

    ws_url = ENV_BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws"

    print(f"[START] task={task_id} env=data-janitor model={MODEL_NAME}", flush=True)

    final_score = 0.0
    rewards: list = []
    steps_taken = 0
    success = False

    try:
        # ping_interval keeps connection alive during slow LLM calls
        async with websockets.connect(ws_url, ping_interval=None) as ws:
            # Reset
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            raw = json.loads(await ws.recv())
            payload = raw.get("data", raw)
            obs = payload.get("observation", payload)
            done = payload.get("done", False)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            max_steps = obs.get("max_steps", 20)
            reward = 0.0

            for step in range(max_steps):
                if done:
                    break

                user_prompt = build_user_prompt(obs)
                messages.append({"role": "user", "content": user_prompt})

                # Run LLM call in thread so WebSocket pings fire concurrently
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    ),
                )
                assistant_text = completion.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": assistant_text})

                action = parse_action(assistant_text)
                if not action:
                    action = {"command": "submit", "column": None, "params": {}}

                col = action.get("column") or ""
                action_str = f"{action['command']}({col})"

                await ws.send(json.dumps({"type": "step", "data": action}))
                raw = json.loads(await ws.recv())
                payload = raw.get("data", raw)
                obs = payload.get("observation", payload)
                done = payload.get("done", False)
                reward = payload.get("reward", 0.0) or 0.0

                steps_taken = step + 1
                rewards.append(reward)

                print(
                    f"[STEP] step={steps_taken} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null",
                    flush=True,
                )

                if len(messages) > 12:
                    messages = messages[:2] + messages[-8:]

            final_score = obs.get("quality_score", 0.0)
            if reward is not None and done:
                final_score = float(reward) if float(reward) > 0 else final_score
            # Clamp to (0, 1) exclusive — required by evaluator
            final_score = max(0.01, min(0.99, final_score))
            success = final_score >= 0.95

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(
            f"[STEP] step={steps_taken + 1} action=error reward=0.00 done=true error={error_msg}",
            flush=True,
        )

    clamped = [max(0.01, min(0.99, r)) if r != 0.0 else r for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in clamped) if clamped else "0.01"
    print(
        f"[END] success={str(success).lower()} steps={steps_taken} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return final_score


async def run_all():
    if not API_KEY:
        # Emit a failed [END] for each task so the evaluator sees structured output
        for task_id in TASK_IDS:
            print(f"[START] task={task_id} env=data-janitor model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores: Dict[str, float] = {}

    for task_id in TASK_IDS:
        score = await run_task_ws(task_id, client)
        scores[task_id] = score

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"# Average score: {avg:.4f}", flush=True)


def main():
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
