from typing import Dict, Any

from .models import (
    ColumnInfo,
    DataJanitorAction,
    DataJanitorObservation,
    DataJanitorState,
)

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult

    class DataJanitorEnv(
        EnvClient[DataJanitorAction, DataJanitorObservation, DataJanitorState]
    ):
        def _step_payload(self, action: DataJanitorAction) -> dict:
            return {
                "command": action.command,
                "column": action.column,
                "params": action.params,
            }

        def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
            obs_data = payload.get("observation", payload)
            schema_raw = obs_data.get("schema_info", [])
            schema_info = [
                ColumnInfo(**s) if isinstance(s, dict) else s for s in schema_raw
            ]

            obs = DataJanitorObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                schema_info=schema_info,
                sample_rows=obs_data.get("sample_rows", []),
                row_count=obs_data.get("row_count", 0),
                quality_score=obs_data.get("quality_score", 0.0),
                issues=obs_data.get("issues", []),
                task_description=obs_data.get("task_description", ""),
                target_schema=obs_data.get("target_schema", {}),
                steps_taken=obs_data.get("steps_taken", 0),
                max_steps=obs_data.get("max_steps", 20),
                available_commands=obs_data.get("available_commands", []),
                message=obs_data.get("message", ""),
                secondary_data_info=obs_data.get("secondary_data_info"),
            )

            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict[str, Any]) -> DataJanitorState:
            return DataJanitorState(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
                task_id=payload.get("task_id", ""),
                difficulty=payload.get("difficulty", ""),
                initial_quality=payload.get("initial_quality", 0.0),
                current_quality=payload.get("current_quality", 0.0),
            )

except ImportError:
    pass
