import numpy as np
import pytest

from memory_guided_vos.simulation import cartesian as cs
from memory_guided_vos.simulation import reward_analysis as ra
from memory_guided_vos.simulation.target_update import update_goal


def test_goal_update_accepts_confident_valid_target():
    result = update_goal(None, np.array([1.0, 1.0]), 0.55, True)
    assert result.accepted_new
    assert result.goal.tolist() == [1.0, 1.0]


def test_goal_update_retains_previous_goal():
    previous = np.array([0.2, 0.2])
    assert update_goal(previous, np.array([1.0, 1.0]), 0.54, True).goal.tolist() == [0.2, 0.2]
    assert update_goal(previous, np.array([1.0, 1.0]), 0.99, False).goal.tolist() == [0.2, 0.2]


def test_goal_update_pauses_without_goal():
    assert update_goal(None, None, None, False).goal is None


def test_episode_reaches_target():
    observations = [cs.Observation(np.array([1.0, 0.5]), 0.9, True)]
    result = cs.run_episode(
        np.array([0.5, 0.5]),
        np.array([1.0, 0.5]),
        observations,
        max_speed_m_s=0.5,
        closure_radius_m=0.01,
    )
    assert result.success
    assert result.closed
    assert result.label == "reconstruction"
    assert result.path_length_m == pytest.approx(0.5, abs=0.02)


def test_episode_times_out_without_confident_observation():
    observations = [cs.Observation(np.array([1.0, 0.5]), 0.3, True)]
    result = cs.run_episode(
        np.array([0.5, 0.5]),
        np.array([1.0, 0.5]),
        observations,
        max_speed_m_s=0.5,
        closure_radius_m=0.01,
    )
    assert not result.success
    assert not result.closed
    assert result.n_paused_steps == 200
    assert result.duration_s == pytest.approx(10.0)


def test_episode_success_uses_true_target():
    observations = [cs.Observation(np.array([1.05, 0.5]), 0.9, True)]
    result = cs.run_episode(
        np.array([0.5, 0.5]),
        np.array([1.0, 0.5]),
        observations,
        max_speed_m_s=1.0,
        closure_radius_m=0.005,
    )
    assert result.closed
    assert not result.success


def test_episode_requires_motion_parameters():
    with pytest.raises(TypeError):
        cs.run_episode(np.zeros(2), np.ones(2), [cs.Observation(None, None, False)])


def test_cartesian_summary_excludes_selected_episodes():
    rows = [
        {
            "episode": "1",
            "success": "1",
            "confidence": "0.8",
            "episode_s": "2",
            "path_m": "0.5",
            "grasp_s": "",
        },
        {
            "episode": "2",
            "success": "1",
            "confidence": "0.9",
            "episode_s": "3",
            "path_m": "0.4",
            "grasp_s": "2.5",
        },
    ]
    summary = ra.cartesian_summary(rows, (1,))
    assert summary["grasp_s"]["n"] == 1
    assert summary["grasp_s"]["mean"] == 2.5
    assert summary["success_rate"] == 1.0


def test_episode_overlap():
    result = ra.episode_overlap(["1", "2"], ["1", "3"])
    assert result == {
        "in_both": ["1"],
        "reward_only": ["3"],
        "cartesian_only": ["2"],
    }
