"""
Full test suite for data-janitor-env.

Run from project root:
    python tests/test_suite.py

Server must be running on localhost:7860 before running this.
Start server:
    set PYTHONPATH=C:\\Users\\yaswa\\Documents\\meta\\data-janitor-env
    cd server
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

import asyncio
import json
import sys
import time
import traceback
from typing import Any, Dict, Tuple

import requests
import websockets

BASE_HTTP = "http://localhost:7860"
BASE_WS = "ws://localhost:7860/ws"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []


def record(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    print(f"  {tag} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, passed))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Server health
# ─────────────────────────────────────────────────────────────────────────────

def test_http_health():
    print("\n[Section 1] Server health & HTTP endpoints")
    try:
        r = requests.get(f"{BASE_HTTP}/health", timeout=5)
        record("GET /health returns 200", r.status_code == 200)
    except Exception as e:
        record("GET /health returns 200", False, str(e))

    try:
        r = requests.get(f"{BASE_HTTP}/metadata", timeout=5)
        record("GET /metadata returns 200", r.status_code == 200)
    except Exception as e:
        record("GET /metadata returns 200", False, str(e))

    try:
        r = requests.get(f"{BASE_HTTP}/schema", timeout=5)
        record("GET /schema returns 200", r.status_code == 200)
    except Exception as e:
        record("GET /schema returns 200", False, str(e))

    try:
        r = requests.get(f"{BASE_HTTP}/docs", timeout=5)
        record("GET /docs (Swagger UI) returns 200", r.status_code == 200)
    except Exception as e:
        record("GET /docs (Swagger UI) returns 200", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: WebSocket reset
# ─────────────────────────────────────────────────────────────────────────────

async def ws_reset(task_id: str) -> Dict[str, Any]:
    async with websockets.connect(BASE_WS) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
        raw = json.loads(await ws.recv())
        return raw["data"]


async def ws_step(ws, action: Dict) -> Tuple[Dict, Any, bool]:
    await ws.send(json.dumps({"type": "step", "data": action}))
    raw = json.loads(await ws.recv())
    p = raw["data"]
    return p["observation"], p.get("reward"), p.get("done", False)


async def test_reset_all_tasks():
    print("\n[Section 2] Reset — all 3 tasks load correctly")
    for task_id, expected_rows in [
        ("fix_basics", 40),
        ("normalize_chaos", 100),
        ("pipeline_merge", 80),
    ]:
        try:
            payload = await ws_reset(task_id)
            obs = payload["observation"]
            row_count = obs["row_count"]
            quality = obs["quality_score"]
            has_issues = len(obs["issues"]) > 0
            has_schema = len(obs["schema_info"]) > 0
            has_samples = len(obs["sample_rows"]) > 0

            record(f"{task_id}: {expected_rows} rows loaded", row_count == expected_rows, f"got {row_count}")
            record(f"{task_id}: initial quality 0–1", 0.0 <= quality <= 1.0, f"{quality:.2%}")
            record(f"{task_id}: issues detected", has_issues)
            record(f"{task_id}: schema info present", has_schema)
            record(f"{task_id}: sample rows present", has_samples)
        except Exception as e:
            record(f"{task_id}: reset", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Dirty baseline scores
# ─────────────────────────────────────────────────────────────────────────────

async def test_dirty_baselines():
    print("\n[Section 3] Dirty baseline scores (no cleaning)")
    expected = {
        "fix_basics": (0.95, 1.0),
        "normalize_chaos": (0.65, 0.85),
        "pipeline_merge": (0.45, 0.65),
    }
    for task_id, (lo, hi) in expected.items():
        try:
            payload = await ws_reset(task_id)
            quality = payload["observation"]["quality_score"]
            in_range = lo <= quality <= hi
            record(
                f"{task_id}: dirty score in expected range [{lo:.0%}–{hi:.0%}]",
                in_range,
                f"got {quality:.2%}",
            )
        except Exception as e:
            record(f"{task_id}: dirty score", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Step commands
# ─────────────────────────────────────────────────────────────────────────────

async def test_step_commands():
    print("\n[Section 4] Step commands — all 16 execute without errors")
    commands_to_test = [
        {"command": "inspect"},
        {"command": "inspect", "column": "employee_id"},
        {"command": "drop_duplicates", "params": {"subset": ["employee_id"]}},
        {"command": "convert_type", "column": "age", "params": {"target_type": "int"}},
        {"command": "normalize_text", "column": "email", "params": {"operation": "trim"}},
        {"command": "normalize_text", "column": "email", "params": {"operation": "lower"}},
        {"command": "normalize_text", "column": "department", "params": {"operation": "lower"}},
        {"command": "normalize_text", "column": "salary", "params": {"operation": "regex_replace", "pattern": r"[\$,]", "replacement": ""}},
        {"command": "convert_type", "column": "salary", "params": {"target_type": "float"}},
        {"command": "fill_missing", "column": "name", "params": {"strategy": "constant", "value": "Unknown"}},
        {"command": "map_values", "column": "department", "params": {"mapping": {"engineering": "Engineering", "hr": "HR", "marketing": "Marketing", "operations": "Operations", "sales": "Sales", "finance": "Finance"}}},
    ]

    async with websockets.connect(BASE_WS) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "fix_basics"}}))
        await ws.recv()

        for action in commands_to_test:
            try:
                obs, reward, done = await ws_step(ws, action)
                msg = obs.get("message", "")
                hard_error = "unknown command" in msg.lower() or "parameter" in msg.lower() and "required" in msg.lower()
                record(
                    f"command '{action['command']}' executes",
                    not hard_error and obs.get("row_count", 0) >= 0,
                    msg[:60] if hard_error else "",
                )
            except Exception as e:
                record(f"command '{action['command']}'", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Optimal playthrough — all tasks reach 1.0
# ─────────────────────────────────────────────────────────────────────────────

async def optimal_task1(ws) -> float:
    await ws.send(json.dumps({"type": "reset", "data": {"task_id": "fix_basics"}}))
    await ws.recv()
    steps = [
        {"command": "drop_duplicates", "params": {"subset": ["employee_id"]}},
        {"command": "normalize_text", "column": "salary", "params": {"operation": "regex_replace", "pattern": r"[\$,]", "replacement": ""}},
        {"command": "convert_type", "column": "salary", "params": {"target_type": "float"}},
        {"command": "convert_type", "column": "age", "params": {"target_type": "int"}},
        {"command": "normalize_text", "column": "email", "params": {"operation": "trim"}},
        {"command": "normalize_text", "column": "email", "params": {"operation": "lower"}},
        {"command": "normalize_text", "column": "department", "params": {"operation": "lower"}},
        {"command": "map_values", "column": "department", "params": {"mapping": {
            "engg": "Engineering", "engineering": "Engineering",
            "mktg": "Marketing", "marketing": "Marketing",
            "ops": "Operations", "operations": "Operations",
            "sales": "Sales", "hr": "HR", "finance": "Finance",
            "human resources": "HR",
        }}},
        {"command": "submit"},
    ]
    obs, reward, done = None, None, False
    for action in steps:
        obs, reward, done = await ws_step(ws, action)
    return reward


async def optimal_task2(ws) -> float:
    sys.path.insert(0, "server")
    from task_data import STATES

    await ws.send(json.dumps({"type": "reset", "data": {"task_id": "normalize_chaos"}}))
    await ws.recv()

    lower_mapping = {}
    for full_name, code in STATES.items():
        lower_mapping[full_name.lower()] = code
        lower_mapping[code.lower()] = code

    steps = [
        {"command": "drop_duplicates", "params": {"subset": ["id"]}},
        {"command": "standardize_date", "column": "signup_date"},
        {"command": "standardize_phone", "column": "phone"},
        {"command": "normalize_text", "column": "first_name", "params": {"operation": "title"}},
        {"command": "normalize_text", "column": "last_name", "params": {"operation": "title"}},
        {"command": "normalize_text", "column": "state", "params": {"operation": "lower"}},
        {"command": "map_values", "column": "state", "params": {"mapping": lower_mapping}},
        {"command": "convert_type", "column": "zip_code", "params": {"target_type": "str"}},
        {"command": "submit"},
    ]
    obs, reward, done = None, None, False
    for action in steps:
        obs, reward, done = await ws_step(ws, action)
    return reward


async def optimal_task3(ws) -> float:
    await ws.send(json.dumps({"type": "reset", "data": {"task_id": "pipeline_merge"}}))
    await ws.recv()

    steps = [
        {"command": "normalize_text", "column": "product_id", "params": {"operation": "regex_replace", "pattern": r"-", "replacement": ""}},
        {"command": "normalize_text", "column": "product_id", "params": {"operation": "upper"}},
        {"command": "normalize_text", "column": "product_id", "params": {"operation": "regex_replace", "pattern": r"([A-Z]+)(\d+)", "replacement": r"\1-\2"}},
        {"command": "normalize_text", "column": "customer_name", "params": {"operation": "title"}},
        {"command": "standardize_date", "column": "order_date"},
        {"command": "filter_rows", "column": "quantity", "params": {"operator": "<=", "value": 0}},
        {"command": "normalize_text", "column": "unit_price", "params": {"operation": "regex_replace", "pattern": r"[\$,]", "replacement": ""}},
        {"command": "convert_type", "column": "unit_price", "params": {"target_type": "float"}},
        {"command": "join", "params": {"on": "product_id", "how": "inner"}},
        {"command": "add_column", "column": "total", "params": {"expression": "quantity * unit_price"}},
        {"command": "submit"},
    ]
    obs, reward, done = None, None, False
    for action in steps:
        obs, reward, done = await ws_step(ws, action)
    return reward


async def test_optimal_playthrough():
    print("\n[Section 5] Optimal playthrough — all tasks must reach score 1.0")
    task_runners = [
        ("fix_basics", optimal_task1),
        ("normalize_chaos", optimal_task2),
        ("pipeline_merge", optimal_task3),
    ]
    for task_id, runner in task_runners:
        try:
            async with websockets.connect(BASE_WS) as ws:
                score = await runner(ws)
                record(
                    f"{task_id}: optimal play reaches 1.0000",
                    score == 1.0,
                    f"got {score}",
                )
        except Exception as e:
            record(f"{task_id}: optimal play", False, traceback.format_exc()[-200:])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Reward signal quality
# ─────────────────────────────────────────────────────────────────────────────

async def test_reward_signal():
    print("\n[Section 6] Reward signal — per-step delta and final score")
    async with websockets.connect(BASE_WS) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "fix_basics"}}))
        await ws.recv()

        # Step that improves quality
        obs, reward, done = await ws_step(ws, {
            "command": "normalize_text", "column": "department",
            "params": {"operation": "lower"}
        })
        record("Per-step reward is a float", isinstance(reward, float), f"{reward}")
        record("Done is False mid-episode", not done)

        # Submit
        obs, reward, done = await ws_step(ws, {"command": "submit"})
        record("Submit sets done=True", done)
        record("Final reward is a float 0–1", isinstance(reward, float) and 0.0 <= reward <= 1.0, f"{reward}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Determinism
# ─────────────────────────────────────────────────────────────────────────────

async def test_determinism():
    print("\n[Section 7] Determinism — same task always produces same initial quality")
    scores = []
    for _ in range(3):
        payload = await ws_reset("fix_basics")
        scores.append(payload["observation"]["quality_score"])
    record(
        "fix_basics initial quality is identical across 3 resets",
        len(set(scores)) == 1,
        f"scores: {scores}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Edge cases
# ─────────────────────────────────────────────────────────────────────────────

async def test_edge_cases():
    print("\n[Section 8] Edge cases")

    # Unknown command
    async with websockets.connect(BASE_WS) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "fix_basics"}}))
        await ws.recv()
        obs, reward, done = await ws_step(ws, {"command": "nonexistent_command"})
        msg = obs.get("message", "")
        record("Unknown command returns error message", "unknown" in msg.lower() or "not" in msg.lower(), msg[:80])

    # Unknown task_id falls back to fix_basics
    async with websockets.connect(BASE_WS) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "bad_task"}}))
        raw = json.loads(await ws.recv())
        obs = raw["data"]["observation"]
        record("Unknown task_id falls back gracefully", obs.get("row_count", 0) > 0)

    # Session isolation — A modifies, B should be unaffected
    async with websockets.connect(BASE_WS) as ws_a:
        await ws_a.send(json.dumps({"type": "reset", "data": {"task_id": "fix_basics"}}))
        await ws_a.recv()
        await ws_a.send(json.dumps({"type": "step", "data": {"command": "drop_duplicates", "params": {"subset": ["employee_id"]}}}))
        rows_a = json.loads(await ws_a.recv())["data"]["observation"]["row_count"]

    async with websockets.connect(BASE_WS) as ws_b:
        await ws_b.send(json.dumps({"type": "reset", "data": {"task_id": "fix_basics"}}))
        rows_b = json.loads(await ws_b.recv())["data"]["observation"]["row_count"]

    record(
        "Sessions are isolated (A=35 after dedup, B=40 fresh reset)",
        rows_a == 35 and rows_b == 40,
        f"A={rows_a}, B={rows_b}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def run_async_tests():
    await test_reset_all_tasks()
    await test_dirty_baselines()
    await test_step_commands()
    await test_optimal_playthrough()
    await test_reward_signal()
    await test_determinism()
    await test_edge_cases()


def main():
    print("=" * 60)
    print("  Data Janitor Env — Full Test Suite")
    print("  Server: " + BASE_HTTP)
    print("=" * 60)

    test_http_health()
    asyncio.run(run_async_tests())

    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED:")
        for name, ok in results:
            if not ok:
                print(f"    FAIL: {name}")
    else:
        print("  — ALL PASS")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
