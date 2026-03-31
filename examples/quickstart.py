"""
Data Janitor Env — 5-Minute Quickstart
======================================

Run from the repo root:
    python examples/quickstart.py

No server, no Docker, no API key needed. Pure in-process execution.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_env import DataJanitorGymEnv

# ── 1. Create environment (in-process, no server needed) ────────────────────
env = DataJanitorGymEnv(task_id="fix_basics", mode="text")
obs, info = env.reset()

print("=" * 60)
print("TASK: Fix the Basics")
print("=" * 60)
print(f"Initial quality score : {info['quality_score']:.2%}")
print(f"Issues detected       : {len(info['issues'])}")
print()

# ── 2. Manual cleaning sequence ─────────────────────────────────────────────
cleaning_steps = [
    env.action_from_dict("drop_duplicates", params={"subset": ["employee_id"]}),
    env.action_from_dict("convert_type", column="age",       params={"target_type": "int"}),
    env.action_from_dict("normalize_text", column="department", params={"operation": "lower"}),
    env.action_from_dict("normalize_text", column="email",    params={"operation": "lower"}),
    env.action_from_dict("normalize_text", column="salary",   params={"operation": "regex_replace",
                                                                        "pattern": r"[$,]",
                                                                        "replacement": ""}),
    env.action_from_dict("convert_type", column="salary",    params={"target_type": "float"}),
    env.action_from_dict("submit"),
]

for i, action in enumerate(cleaning_steps, 1):
    import json
    cmd = json.loads(action)["command"]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i:2d}: {cmd:20s}  quality={info['quality_score']:.2%}  d={reward:+.4f}")
    if terminated:
        print(f"\n✓ Final score: {info['quality_score']:.4f}")
        break

env.close()

# ── 3. Dict (numerical) mode — for classical RL ──────────────────────────────
print("\n" + "=" * 60)
print("NUMERICAL MODE DEMO (for classical RL)")
print("=" * 60)

env2 = DataJanitorGymEnv(task_id="fix_basics", mode="dict")
vec_obs, info = env2.reset()
print(f"Observation vector shape : {vec_obs.shape}")
print(f"Observation values       : {vec_obs}")
print(f"Action space             : {env2.action_space}")
print(f"Action table size        : {env2.get_action_count()} pre-defined actions")
print()

# Run through the optimal action sequence (action indices 1..10)
for action_idx in range(1, env2.get_action_count()):
    vec_obs, reward, terminated, truncated, info = env2.step(action_idx)
    cmd, col, _ = env2.action_table[action_idx]
    print(f"  Action {action_idx:2d}: {cmd}({col or ''})  -> quality={info['quality_score']:.2%}  d={reward:+.4f}")
    if terminated:
        print(f"\n✓ Final score: {info['quality_score']:.4f}")
        break

env2.close()
print("\nDone! See examples/train_rl_agent.py to train a policy.")
