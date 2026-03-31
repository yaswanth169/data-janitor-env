"""
Gymnasium wrapper for Data Janitor Env.

Two usage modes:
    mode="text"  — for LLM-based RL (TRL / GRPO / REINFORCE).
                   obs = natural-language prompt string
                   action = JSON command string
    mode="dict"  — for classical RL (PPO / DQN / Q-learning).
                   obs = 8-dim float32 numpy array
                   action = Discrete index into a task-specific action table

Quick start:
    from gym_env import DataJanitorGymEnv

    env = DataJanitorGymEnv(task_id="fix_basics", mode="text")
    obs, info = env.reset()
    action = '{"command": "drop_duplicates", "column": null, "params": {}}'
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Make server modules importable in-process (no server/network needed) ─────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = os.path.join(_HERE, "server")
for _p in (_SERVER, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
from gymnasium import spaces

# Lazy-imported server modules (only when env is instantiated)
_env_mod = None
_models_mod = None
_task_data_mod = None


def _load_server():
    global _env_mod, _models_mod, _task_data_mod
    if _env_mod is None:
        import importlib
        _env_mod = importlib.import_module("environment")
        _models_mod = importlib.import_module("models")
        _task_data_mod = importlib.import_module("task_data")


# ── Pre-defined action tables for discrete mode ───────────────────────────────
# Each entry: (command, column, params)
_ACTIONS: Dict[str, List[Tuple[str, Optional[str], Dict]]] = {
    "fix_basics": [
        ("inspect",         None,         {}),
        ("drop_duplicates", None,         {"subset": ["employee_id"]}),
        ("convert_type",    "age",        {"target_type": "int"}),
        ("normalize_text",  "department", {"operation": "lower"}),
        ("normalize_text",  "email",      {"operation": "trim"}),
        ("normalize_text",  "email",      {"operation": "lower"}),
        ("normalize_text",  "salary",     {"operation": "regex_replace", "pattern": r"[$,]", "replacement": ""}),
        ("convert_type",    "salary",     {"target_type": "float"}),
        ("fill_missing",    "age",        {"strategy": "mean"}),
        ("drop_duplicates", None,         {}),
        ("submit",          None,         {}),
    ],
    "normalize_chaos": [
        ("inspect",            None,          {}),
        ("drop_duplicates",    None,          {"subset": ["contact_id"]}),
        ("standardize_date",   "signup_date", {"format": "%Y-%m-%d"}),
        ("standardize_phone",  "phone",       {}),
        ("normalize_text",     "name",        {"operation": "title"}),
        ("normalize_text",     "state",       {"operation": "upper"}),
        ("map_values",         "state",       {"mapping": {
            "California": "CA", "Texas": "TX", "New York": "NY",
            "Florida": "FL", "Illinois": "IL", "Pennsylvania": "PA",
            "Ohio": "OH", "Georgia": "GA", "Michigan": "MI",
            "Washington": "WA", "Colorado": "CO", "Arizona": "AZ",
        }}),
        ("normalize_text",     "name",        {"operation": "trim"}),
        ("drop_duplicates",    None,          {}),
        ("submit",             None,          {}),
    ],
    "pipeline_merge": [
        ("inspect",           None,            {}),
        ("normalize_text",    "product_id",    {"operation": "regex_replace", "pattern": "-", "replacement": ""}),
        ("normalize_text",    "product_id",    {"operation": "upper"}),
        ("normalize_text",    "product_id",    {"operation": "regex_replace", "pattern": r"([A-Z]+)(\d+)", "replacement": r"\1-\2"}),
        ("join",              None,            {"on": "product_id", "how": "left"}),
        ("normalize_text",    "unit_price",    {"operation": "regex_replace", "pattern": r"[$,]", "replacement": ""}),
        ("convert_type",      "unit_price",    {"target_type": "float"}),
        ("filter_rows",       "quantity",      {"operator": "<=", "value": 0}),
        ("standardize_date",  "order_date",    {"format": "%Y-%m-%d"}),
        ("normalize_text",    "customer_name", {"operation": "title"}),
        ("add_column",        "total_amount",  {"expression": "quantity * unit_price"}),
        ("drop_duplicates",   None,            {}),
        ("submit",            None,            {}),
    ],
}

_TASK_MAX_STEPS = {"fix_basics": 15, "normalize_chaos": 20, "pipeline_merge": 30}
_TASK_IDS = list(_ACTIONS.keys())


class DataJanitorGymEnv(gym.Env):
    """
    Gymnasium environment for Data Janitor — in-process (no server needed).

    Parameters
    ----------
    task_id : str
        One of "fix_basics", "normalize_chaos", "pipeline_merge".
        Pass "random" to sample a random task at each reset.
    mode : str
        "text"  — LLM-friendly; obs is a prompt string, action is JSON.
        "dict"  — classical RL; obs is a float32 array, action is Discrete.
    render_mode : str or None
        "human" prints the observation to stdout; "ansi" returns it as a string.
    """

    metadata = {"render_modes": ["human", "ansi"]}
    TASK_IDS = _TASK_IDS

    def __init__(
        self,
        task_id: str = "fix_basics",
        mode: str = "text",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert task_id in (*_TASK_IDS, "random"), (
            f"task_id must be one of {_TASK_IDS} or 'random'"
        )
        assert mode in ("text", "dict"), "mode must be 'text' or 'dict'"

        self.task_id = task_id
        self.mode = mode
        self.render_mode = render_mode

        self._inner = None          # DataJanitorEnvironment instance
        self._last_obs_obj = None   # DataJanitorObservation
        self._active_task = None    # resolved task id
        self._initial_row_count = 1
        self._np_rng = None

        if mode == "text":
            # Observation: natural-language prompt (unbounded text)
            self.observation_space = spaces.Text(min_length=10, max_length=8192)
            # Action: JSON command string
            self.action_space = spaces.Text(min_length=2, max_length=1024)
        else:
            # Observation: 8-dim float32 feature vector (all normalised to [0,1])
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            )
            # Action: index into the task's predefined action table
            max_actions = max(len(v) for v in _ACTIONS.values())
            self.action_space = spaces.Discrete(max_actions)

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        _load_server()

        if self.task_id == "random":
            self._active_task = self.np_random.choice(_TASK_IDS)
        else:
            self._active_task = self.task_id

        self._inner = _env_mod.DataJanitorEnvironment()
        obs_obj = self._inner.reset(task_id=self._active_task)
        self._last_obs_obj = obs_obj
        self._initial_row_count = max(obs_obj.row_count, 1)

        info = self._build_info(obs_obj)
        return self._encode_obs(obs_obj), info

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        assert self._inner is not None, "Call reset() before step()"

        action_obj = self._decode_action(action)
        obs_obj = self._inner.step(action_obj)
        self._last_obs_obj = obs_obj

        reward = float(obs_obj.reward) if obs_obj.reward is not None else 0.0
        terminated = bool(obs_obj.done)
        truncated = False
        info = self._build_info(obs_obj)

        return self._encode_obs(obs_obj), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        if self._last_obs_obj is None:
            return None
        text = self._obs_to_text(self._last_obs_obj)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        self._inner = None
        self._last_obs_obj = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_obs(self, obs_obj) -> Any:
        if self.mode == "text":
            return self._obs_to_text(obs_obj)
        else:
            return self._obs_to_vector(obs_obj)

    def _obs_to_text(self, obs) -> str:
        """Build the natural-language prompt (same format as inference.py)."""
        import json as _json
        parts = [
            f"Task: {obs.task_description}",
            f"Rows: {obs.row_count} | Quality: {obs.quality_score:.2%}",
            f"Step: {obs.steps_taken}/{obs.max_steps}",
            "",
            "Target schema:",
            _json.dumps(obs.target_schema, indent=2),
            "",
            "Current issues:",
        ]
        for issue in obs.issues:
            parts.append(f"  - {issue}")
        parts.append("")
        parts.append("Schema:")
        for col_info in obs.schema_info:
            if hasattr(col_info, "name"):
                parts.append(
                    f"  {col_info.name}: {col_info.dtype} "
                    f"({col_info.null_count} nulls, {col_info.unique_count} unique) "
                    f"sample={col_info.sample_values[:3]}"
                )
        parts.append("")
        parts.append("Sample rows:")
        for row in obs.sample_rows[:3]:
            parts.append(f"  {_json.dumps(row, default=str)}")
        if obs.message:
            parts.append("")
            parts.append(f"Last result: {obs.message}")
        return "\n".join(parts)

    def _obs_to_vector(self, obs) -> np.ndarray:
        """Convert observation to 8-dim float32 vector for classical RL."""
        issues_text = " ".join(obs.issues)
        has_dup = float("duplicate" in issues_text.lower())
        has_type = float("mixed types" in issues_text.lower() or "type" in issues_text.lower())
        has_casing = float("casing" in issues_text.lower())

        total_cells = max(obs.row_count * max(len(obs.schema_info), 1), 1)
        total_nulls = sum(
            c.null_count for c in obs.schema_info if hasattr(c, "null_count")
        )
        null_frac = min(total_nulls / total_cells, 1.0)
        issues_frac = min(len(obs.issues) / 20.0, 1.0)
        steps_frac = obs.steps_taken / max(obs.max_steps, 1)
        row_frac = obs.row_count / max(self._initial_row_count, 1)

        return np.array([
            obs.quality_score,
            steps_frac,
            row_frac,
            issues_frac,
            null_frac,
            has_dup,
            has_type,
            has_casing,
        ], dtype=np.float32)

    def _decode_action(self, action: Any):
        """Parse action (text JSON or discrete int) into a DataJanitorAction."""
        Action = _models_mod.DataJanitorAction
        if self.mode == "text":
            if isinstance(action, str):
                try:
                    d = json.loads(action)
                    return Action(
                        command=d.get("command", "inspect"),
                        column=d.get("column"),
                        params=d.get("params", {}),
                    )
                except (json.JSONDecodeError, Exception):
                    # Fall back to inspect if JSON is malformed
                    return Action(command="inspect")
            elif isinstance(action, dict):
                return Action(**action)
            else:
                return Action(command="inspect")
        else:
            # Discrete: index into action table
            table = _ACTIONS.get(self._active_task, _ACTIONS["fix_basics"])
            idx = int(action) % len(table)
            cmd, col, params = table[idx]
            return Action(command=cmd, column=col, params=params)

    def _build_info(self, obs) -> Dict[str, Any]:
        return {
            "task_id": self._active_task,
            "quality_score": obs.quality_score,
            "steps_taken": obs.steps_taken,
            "max_steps": obs.max_steps,
            "row_count": obs.row_count,
            "issues": obs.issues,
            "message": obs.message,
            "done": obs.done,
        }

    # ── Convenience methods ────────────────────────────────────────────────────

    def action_from_dict(
        self,
        command: str,
        column: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> str:
        """Build a valid text-mode action string from components."""
        return json.dumps({
            "command": command,
            "column": column,
            "params": params or {},
        })

    @property
    def action_table(self) -> List[Tuple[str, Optional[str], Dict]]:
        """Pre-defined action table for the current task (dict mode only)."""
        return _ACTIONS.get(self._active_task or self.task_id, [])

    def get_action_count(self) -> int:
        """Number of valid discrete actions for the current task."""
        return len(self.action_table)
