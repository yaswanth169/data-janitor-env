"""
Data Janitor Env — Train a Classical RL Agent
=============================================

Trains a PPO agent using stable-baselines3 (or falls back to a simple
Q-table if SB3 is not installed).

Install dependencies:
    pip install stable-baselines3

Run:
    python examples/train_rl_agent.py

The trained agent learns WHICH cleaning operations to apply and in WHAT
ORDER based on the numerical quality signal — without hard-coding any rules.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from gym_env import DataJanitorGymEnv

# ── Configuration ──────────────────────────────────────────────────────────
TASK_ID    = "fix_basics"   # easy task to demonstrate learning
N_EPISODES = 200            # increase for better convergence
SEED       = 42


def evaluate(env: DataJanitorGymEnv, policy, n_episodes: int = 20) -> float:
    """Run n_episodes and return mean final quality score."""
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(obs)
            obs, _, terminated, truncated, info = env.step(action)
        scores.append(info["quality_score"])
    return float(np.mean(scores))


# ── Option A: PPO via stable-baselines3 ────────────────────────────────────
def train_with_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        return None

    print("\n── Training with Stable-Baselines3 PPO ──")
    env = DataJanitorGymEnv(task_id=TASK_ID, mode="dict")
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        seed=SEED,
        tensorboard_log="./tb_logs/",
    )
    from gym_env import _TASK_MAX_STEPS
    model.learn(total_timesteps=N_EPISODES * _TASK_MAX_STEPS.get(TASK_ID, 15))
    model.save(f"ppo_{TASK_ID}")
    print(f"\nModel saved to ppo_{TASK_ID}.zip")

    # Evaluate
    eval_env = DataJanitorGymEnv(task_id=TASK_ID, mode="dict")
    mean_score = evaluate(eval_env, lambda obs: model.predict(obs, deterministic=True)[0])
    print(f"Mean final quality (20 episodes): {mean_score:.4f}")
    env.close()
    eval_env.close()
    return mean_score


# ── Option B: Simple Q-table (no extra dependencies) ───────────────────────
def train_with_qtable():
    """
    Discretise the 8-dim obs into bins and run Q-learning.
    Works without any extra dependencies.
    """
    print("\n── Training with Q-table (no extra deps) ──")

    env = DataJanitorGymEnv(task_id=TASK_ID, mode="dict")
    n_actions = env.get_action_count()

    # Discretise obs into (quality_bucket, steps_bucket) = 5 x 5 = 25 states
    N_BINS = 5
    q_table = np.zeros((N_BINS, N_BINS, n_actions))

    alpha   = 0.3   # learning rate
    gamma   = 0.95  # discount
    epsilon = 1.0   # exploration
    eps_min = 0.05
    eps_decay = 0.98

    def discretise(obs):
        q_bin = int(obs[0] * (N_BINS - 1))          # quality score
        s_bin = int(obs[1] * (N_BINS - 1))          # steps fraction
        return min(q_bin, N_BINS-1), min(s_bin, N_BINS-1)

    episode_scores = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        state = discretise(obs)
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretise(next_obs)

            # Q-update
            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )
            state = next_state
            total_reward += reward

        episode_scores.append(info["quality_score"])
        epsilon = max(eps_min, epsilon * eps_decay)

        if (ep + 1) % 50 == 0:
            mean = np.mean(episode_scores[-50:])
            print(f"  Episode {ep+1:4d} | Mean quality (last 50): {mean:.4f} | ε={epsilon:.3f}")

    env.close()

    # Evaluate greedy policy
    eval_env = DataJanitorGymEnv(task_id=TASK_ID, mode="dict")
    greedy = lambda obs: int(np.argmax(q_table[discretise(obs)]))
    mean_score = evaluate(eval_env, greedy)
    print(f"\nGreedy policy — mean final quality (20 eps): {mean_score:.4f}")
    eval_env.close()
    return mean_score


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Data Janitor Env — RL Training Demo")
    print(f"Task: {TASK_ID} | Episodes: {N_EPISODES}")

    score = train_with_sb3()
    if score is None:
        print("(stable-baselines3 not found — using Q-table fallback)")
        score = train_with_qtable()

    print(f"\nFinal mean score: {score:.4f}")
    print("\nNext steps:")
    print("  - Increase N_EPISODES for better convergence")
    print("  - Try task_id='normalize_chaos' or 'pipeline_merge'")
    print("  - Use mode='text' + an LLM policy for language-grounded RL")
    print("  - See examples/llm_agent.py for LLM-based training")
