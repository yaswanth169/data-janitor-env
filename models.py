from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None

    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    total_count: int
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list)


class DataJanitorAction(Action):
    command: str
    column: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class DataJanitorObservation(Observation):
    schema_info: List[ColumnInfo] = Field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    quality_score: float = 0.0
    issues: List[str] = Field(default_factory=list)
    task_description: str = ""
    target_schema: Dict[str, str] = Field(default_factory=dict)
    steps_taken: int = 0
    max_steps: int = 20
    available_commands: List[str] = Field(default_factory=list)
    message: str = ""
    secondary_data_info: Optional[Dict[str, Any]] = None


class DataJanitorState(State):
    task_id: str = ""
    difficulty: str = ""
    initial_quality: float = 0.0
    current_quality: float = 0.0
