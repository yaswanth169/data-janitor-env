---
title: Data Janitor Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

**Team Name:** Byte Me
**Team Members:** 
- Ireddi Rakshitha
- Devavarapu Yashwanth

# Data Janitor Env

An OpenEnv environment for training AI agents to clean messy real-world data.

Data scientists spend 80% of their time on data wrangling: deduplication, type conversion, format standardization, and dataset reconciliation. This environment turns that problem into a structured RL benchmark with measurable per-step progress, deterministic grading, and realistic scenarios drawn from production data pipelines.

## Why This Matters

| Gap | How Data Janitor fills it |
|-----|--------------------------|
| No RL benchmark for data cleaning | 4 tasks across 4 difficulty levels with deterministic graders |
| LLM agents cannot practice data ops | 16 transformation commands with rich per-step feedback |
| Existing envs are games or toys | Modeled on real employee records, CRM exports, sales pipelines, and student registries |
| Sparse reward is uninformative | Multi-dimensional grader gives signal at every meaningful step |

## Action Space

```python
class DataJanitorAction(Action):
    command: str                          # Transformation command
    column: Optional[str] = None          # Target column
    params: Dict[str, Any] = {}           # Command-specific parameters
```

### Available Commands

| Command | Description | Key Params |
|---------|-------------|------------|
| `inspect` | View column or dataset stats | none |
| `drop_duplicates` | Remove duplicate rows | `subset`: list of columns |
| `fill_missing` | Fill null values | `strategy`: mean/median/mode/constant, `value` |
| `drop_nulls` | Drop rows with nulls in column | none |
| `convert_type` | Cast column to target type | `target_type`: int/float/str |
| `normalize_text` | Text operations on column | `operation`: trim/lower/upper/title/regex_replace |
| `standardize_date` | Parse dates to ISO format | `format`: strftime pattern |
| `standardize_phone` | Normalize phone to (XXX) XXX-XXXX | none |
| `rename_column` | Rename a column | `new_name` |
| `map_values` | Remap specific values | `mapping`: {old: new} |
| `filter_rows` | Remove rows by condition | `operator`, `value` |
| `split_column` | Split column by delimiter | `delimiter`, `new_columns` |
| `merge_columns` | Combine multiple columns | `columns`, `new_column`, `separator` |
| `join` | Merge secondary dataset | `on`, `how`: inner/left |
| `add_column` | Compute a new column | `expression`: "col_a * col_b" |
| `submit` | Finalize and score the episode | none |

## Observation Space

```python
class DataJanitorObservation(Observation):
    schema_info: List[ColumnInfo]         # Column names, types, null counts, samples
    sample_rows: List[Dict]               # First 5 rows of current data
    row_count: int                        # Current row count
    quality_score: float                  # 0.0 to 1.0 composite grader score
    issues: List[str]                     # Auto-detected problems
    task_description: str                 # What needs to be cleaned
    target_schema: Dict[str, str]         # Expected output column types
    steps_taken: int                      # Steps used so far
    max_steps: int                        # Step budget
    available_commands: List[str]         # Valid commands
    message: str                          # Feedback from last action
    secondary_data_info: Optional[Dict]   # Secondary dataset preview (Task 3 only)
```

## Tasks

### Task 1: Fix the Basics (Easy)

**Dataset:** 40 employee records (35 unique + 5 duplicates)

**Issues:**
- Duplicate rows
- Ages stored as strings instead of integers
- Department names with inconsistent casing and abbreviations (OPERATIONS, Ops, engineering)
- Emails with extra whitespace and uppercase characters
- Salaries formatted as strings with dollar signs and commas

**Expected steps:** 4 to 6 | **Max steps:** 15

### Task 2: Normalize the Chaos (Medium)

**Dataset:** 100 customer contacts (90 unique + 10 duplicates)

**Issues:**
- Signup dates in 5 different formats
- Phone numbers in 6 different formats
- US states stored as full names instead of 2-letter codes
- First and last names with inconsistent casing
- Zip codes stored as integers with stripped leading zeros

**Expected steps:** 7 to 10 | **Max steps:** 20

### Task 3: Pipeline Merge (Hard)

**Dataset:** 80 orders + 30 products requiring a join

**Issues:**
- Product ID casing mismatch between orders and products tables
- Incorrect totals that do not match quantity multiplied by unit price
- Non-positive quantities that must be filtered
- Mixed date formats across order records
- Currency symbols in unit price column
- Customer name casing inconsistencies

**Expected steps:** 8 to 12 | **Max steps:** 30

### Task 4: Student Records Cleanup (Transfer Task, Medium-Hard)

**Dataset:** 60 university student records (50 unique + 10 duplicates)

**Issues:**
- GPA stored as strings with extra whitespace
- Major names with inconsistent casing (COMPUTER SCIENCE, computer science)
- Enrollment dates in 4 different formats
- Graduation year stored as strings instead of integers
- Status values with synonyms (enrolled/active/current map to Active, graduated/complete map to Graduated, withdrawn/dropped map to Inactive)
- Full name casing inconsistencies

This task uses a completely different domain (university administration vs business operations) to test whether agents generalise cleaning skills beyond the training distribution.

**Expected steps:** 7 to 10 | **Max steps:** 20

## Reward Design

The environment uses a multi-dimensional grader that produces signal at every meaningful cleaning step, not just at submission.

**Per-step reward:** Change in quality score between consecutive steps. Positive when the data improves, negative when a transformation regresses quality.

**Final reward (on submit):** The composite quality score comparing the cleaned dataset against ground truth.

### Multi-Dimensional Quality Score

The quality score combines four dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Accuracy | 70% | Cell-by-cell value correctness vs ground truth (case-sensitive, exact match for strings; numeric tolerance +/- 0.02 for numbers) |
| Completeness | 15% | Fraction of expected rows still present after cleaning |
| Integrity | 10% | Absence of duplicate primary keys |
| Type correctness | 5% | Column Python types matching the target schema |

```
quality_score = 0.70 * accuracy + 0.15 * completeness + 0.10 * integrity + 0.05 * type_score
```

This design gives agents feedback at every step: removing duplicates immediately improves integrity, fixing casing immediately improves accuracy, converting types immediately improves type correctness. Agents do not have to wait until submission to know whether they are making progress.

String comparisons are case-sensitive and exact (after whitespace stripping), meaning agents must actually apply normalization commands to receive credit. Numeric comparisons use a tolerance of +/- 0.02 to handle floating-point formatting differences.

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via the included `inference.py` script against the live HF Space:

| Task | Difficulty | Score |
|------|-----------|-------|
| fix_basics | Easy | **0.97** |
| normalize_chaos | Medium | **0.99** |
| pipeline_merge | Hard | **0.92** |
| student_records | Transfer | **0.83** |
| **Average** | | **0.9274** |

## Architecture

```
LLM Agent (any OpenAI-compatible model)
    observe, reason, act loop over WebSocket
            |
            | WebSocket  {"type": "step", "data": {...}}
            |
    FastAPI Server (openenv-core)
        /ws  stateful session
        /health  /docs  / (dashboard UI)
            |
    DataJanitorEnvironment
        reset(task_id)  produces dirty dataset and initial observation
        step(action)    applies transformation, grades result, returns reward
            |                           |
    DataEngine                      Multi-Dimensional Grader
        16 transformation commands      accuracy     (70%)
        drop_duplicates                 completeness (15%)
        fill_missing                    integrity    (10%)
        convert_type                    type_score   ( 5%)
        normalize_text                  numeric tolerance +/-0.02
        standardize_date/phone          case-sensitive string match
        join / merge / add_column       score clamped to (0, 1)
            |
    TaskData (seeded, deterministic)
        Task 1  fix_basics       40 employee records      15 steps  easy
        Task 2  normalize_chaos  100 customer contacts    20 steps  medium
        Task 3  pipeline_merge   80 orders + 30 products  30 steps  hard
        Task 4  student_records  60 student records       20 steps  transfer
```

## Quick Start

### 1. Run the Baseline Inference Script

```bash
git clone https://huggingface.co/spaces/yaswanth169/data-janitor-env
cd data-janitor-env
pip install openai websockets httpx

export HF_TOKEN="your-hf-token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

Expected output:

```
[START] task=fix_basics env=data-janitor model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=drop_duplicates(employee_id) reward=0.01 done=false error=null
[STEP] step=2 action=normalize_text(department) reward=0.02 done=false error=null
...
[END] success=true steps=10 score=0.97 rewards=0.01,0.02,...
```

### 2. WebSocket API

```python
import asyncio, json, websockets

async def run():
    url = "wss://yaswanth169-data-janitor-env.hf.space/ws"
    async with websockets.connect(url, ping_interval=None) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "fix_basics"}}))
        raw = json.loads(await ws.recv())
        obs = raw["data"]["observation"]
        print(f"Quality: {obs['quality_score']:.2%}")

        action = {"command": "drop_duplicates", "column": None, "params": {}}
        await ws.send(json.dumps({"type": "step", "data": action}))
        raw = json.loads(await ws.recv())
        print(f"Reward: {raw['data']['reward']:.4f}")

asyncio.run(run())
```

### 3. Interactive Dashboard

Visit the live HF Space at https://huggingface.co/spaces/yaswanth169/data-janitor-env

The dashboard lets you select a task, issue cleaning commands, watch quality improve step by step, and run the built-in baseline agent with one click.

### 4. Server Mode

```bash
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860

export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### 5. Docker

```bash
docker build -t data-janitor-env .
docker run -p 7860:7860 data-janitor-env
```

Dashboard: http://localhost:7860
API docs:  http://localhost:7860/docs
WebSocket: ws://localhost:7860/ws

## Real-World Use Case

This environment trains AI agents to automate data wrangling:

1. Train an agent on the 4 tasks using the continuous reward signal
2. Fine-tune an LLM using the per-step quality improvements as the training signal
3. Deploy the trained model to an ETL pipeline
4. The model auto-detects issues and applies the correct transformations in order
5. Only edge cases escalate to human review, saving hours per dataset

The transfer task (student_records) exists specifically to test whether trained agents generalise beyond the original three domains.

## Project Structure

```
data-janitor-env/
├── inference.py           # Baseline LLM agent (entry point for evaluation)
├── models.py              # Pydantic models: Action, Observation, State
├── client.py              # WebSocket client
├── gym_env.py             # Gymnasium wrapper for RL training
├── openenv.yaml           # OpenEnv environment manifest
├── Dockerfile             # Container definition
├── requirements.txt       # Server dependencies
├── pyproject.toml         # Package configuration
├── examples/
│   ├── quickstart.py      # 5-minute getting started script
│   ├── train_rl_agent.py  # PPO and Q-table RL training examples
│   └── llm_agent.py       # LLM agent via OpenAI-compatible API
└── server/
    ├── app.py             # FastAPI application
    ├── environment.py     # OpenEnv Environment implementation
    ├── engine.py          # Data transformation engine (16 commands)
    ├── graders.py         # Multi-dimensional grading system
    └── task_data.py       # Seeded deterministic dataset generation (4 tasks)
```

## Setup for Development

```bash
git clone https://huggingface.co/spaces/yaswanth169/data-janitor-env
cd data-janitor-env
pip install -e ".[server,inference,gym,dev]"

PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860 &
python tests/test_suite.py

openenv push --repo-id yaswanth169/data-janitor-env
```

## License

MIT
