import uuid
from typing import Any, Dict, List, Optional

from models import (
    ColumnInfo,
    DataJanitorAction,
    DataJanitorObservation,
    DataJanitorState,
)
from engine import DataEngine
from graders import detect_issues, grade
from task_data import get_task, TASK_IDS

try:
    from openenv.core.env_server import Environment
except ImportError:
    from abc import ABC

    class Environment(ABC):  # type: ignore[no-redef]
        SUPPORTS_CONCURRENT_SESSIONS = True


class DataJanitorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._engine: Optional[DataEngine] = None
        self._ground_truth: List[Dict[str, Any]] = []
        self._config: Dict[str, Any] = {}
        self._state = DataJanitorState()
        self._done = False
        self._initial_quality = 0.0
        self._current_quality = 0.0
        self._join_done = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataJanitorObservation:
        task_id = kwargs.get("task_id", "fix_basics")
        if task_id not in TASK_IDS:
            task_id = "fix_basics"

        dirty_data, clean_data, config = get_task(task_id)
        secondary = config.get("secondary_data")

        self._engine = DataEngine(dirty_data, secondary_data=secondary)
        self._ground_truth = clean_data
        self._config = config
        self._done = False
        self._join_done = False

        self._initial_quality = grade(
            self._engine.data, self._ground_truth, config["primary_key"]
        )
        self._current_quality = self._initial_quality

        self._state = DataJanitorState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=config["difficulty"],
            initial_quality=self._initial_quality,
            current_quality=self._current_quality,
        )

        return self._build_observation(
            message=(
                f"Task '{config['name']}' loaded. "
                f"{len(dirty_data)} rows, {len(self._engine.columns)} columns. "
                f"Initial quality: {self._initial_quality:.2%}. "
                f"Clean the data and submit when ready."
            )
        )

    def step(
        self,
        action: DataJanitorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataJanitorObservation:
        if self._done:
            return self._build_observation(message="Episode already finished.")

        if self._engine is None:
            return self._build_observation(
                message="No active episode. Call reset() first."
            )

        self._state.step_count += 1
        max_steps = self._config.get("max_steps", 20)

        if action.command == "join" and self._engine.secondary_data is not None:
            self._join_done = True

        result = self._engine.execute(action.command, action.column, action.params)

        prev_quality = self._current_quality
        self._current_quality = grade(
            self._engine.data,
            self._ground_truth,
            self._config["primary_key"],
        )
        self._state.current_quality = self._current_quality

        if result == "submitted" or self._state.step_count >= max_steps:
            self._done = True
            final_score = self._current_quality
            if result != "submitted":
                result = f"Max steps reached. Auto-submitting. Final score: {final_score:.4f}"
            else:
                result = f"Submitted. Final score: {final_score:.4f}"
            return self._build_observation(
                message=result,
                reward=final_score,
                done=True,
            )

        delta = self._current_quality - prev_quality
        step_reward = round(delta, 4)

        return self._build_observation(
            message=result,
            reward=step_reward,
        )

    @property
    def state(self) -> DataJanitorState:
        return self._state

    def _build_observation(
        self,
        message: str = "",
        reward: Optional[float] = None,
        done: Optional[bool] = None,
    ) -> DataJanitorObservation:
        if self._engine is None or not self._engine.data:
            return DataJanitorObservation(
                done=done if done is not None else self._done,
                reward=reward,
                message=message,
                available_commands=DataEngine.COMMANDS,
            )

        schema_info = self._compute_schema()
        sample_rows = self._engine.data[:5]
        issues = detect_issues(
            self._engine.data,
            self._config.get("target_schema"),
        )

        secondary_info = None
        if self._engine.secondary_data is not None:
            sec = self._engine.secondary_data
            secondary_info = {
                "name": "secondary_dataset",
                "row_count": len(sec),
                "columns": list(sec[0].keys()) if sec else [],
                "sample_rows": sec[:3],
            }

        task_desc = self._config.get("description", "")
        if self._join_done:
            task_desc += " [JOIN COMPLETE — secondary data already merged. Do NOT call join again.]"

        return DataJanitorObservation(
            done=done if done is not None else self._done,
            reward=reward,
            schema_info=schema_info,
            sample_rows=sample_rows,
            row_count=len(self._engine.data),
            quality_score=self._current_quality,
            issues=issues,
            task_description=task_desc,
            target_schema=self._config.get("target_schema", {}),
            steps_taken=self._state.step_count,
            max_steps=self._config.get("max_steps", 20),
            available_commands=DataEngine.COMMANDS,
            message=message,
            secondary_data_info=secondary_info,
        )

    def _compute_schema(self) -> List[ColumnInfo]:
        if not self._engine or not self._engine.data:
            return []

        info: List[ColumnInfo] = []
        for col in self._engine.columns:
            values = [row.get(col) for row in self._engine.data]
            non_null = [v for v in values if v is not None and str(v).strip() != ""]
            types = set(type(v).__name__ for v in non_null)
            dtype = ", ".join(sorted(types)) if types else "unknown"
            sample = [v for v in non_null[:5]]
            info.append(ColumnInfo(
                name=col,
                dtype=dtype,
                null_count=len(values) - len(non_null),
                total_count=len(values),
                unique_count=len(set(str(v) for v in non_null)),
                sample_values=sample,
            ))
        return info
