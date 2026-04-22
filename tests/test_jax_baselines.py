import jax
import numpy as np

from modules.jax_baselines import evaluate_baseline_policies
from modules.jax_environment import JaxDecisionTreeEnv


def test_baseline_policy_evaluation_runs():
    env = JaxDecisionTreeEnv(
        num_nodes=3,
        beta_move=4.0,
        eps_move=0.02,
        learning_rate=1.0,
        wm_decay=1.0,
        t_max=10,
        cost=0.01,
        scale_factor=1.0,
        shuffle_nodes=False,
        point_set=np.array([1.0], dtype=np.float32),
    )

    reset_keys = jax.random.split(jax.random.PRNGKey(0), 12)

    stats, optimal_scaled, optimal_raw = evaluate_baseline_policies(
        env=env,
        policy_names=[
            "depth1_then_terminate",
            "visit_all_then_bestg_then_parent_chain",
        ],
        reset_keys=reset_keys,
    )

    assert len(stats) == 2
    assert np.isfinite(optimal_scaled)
    assert np.isfinite(optimal_raw)

    for item in stats:
        assert np.isfinite(item.mean_episode_reward)
        assert np.isfinite(item.mean_no_cost_reward_scaled)
        assert np.isfinite(item.mean_no_cost_reward_raw)
        assert item.mean_episode_length > 0
