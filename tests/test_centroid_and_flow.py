import numpy as np
import pytest

from memory_guided_vos import centroid_metrics as cm
from memory_guided_vos import optical_flow as of
from memory_guided_vos import statistics as stats


def test_path_length_straight_line():
    centroids = np.array([[0, 0], [3, 4], [6, 8]], dtype=float)
    assert cm.path_length(centroids) == pytest.approx(10.0)


def test_path_length_skips_empty_frames():
    centroids = np.array(
        [[0, 0], [np.nan, np.nan], [3, 4], [6, 8]], dtype=float
    )
    assert cm.path_length(centroids) == pytest.approx(5.0)


def test_allan_variance_constant_is_zero():
    centroids = np.tile([[5.0, 7.0]], (20, 1))
    assert cm.allan_variance_2d(centroids, 2) == 0.0


def test_allan_variance_hand_computed():
    centroids = np.array(
        [[0, 0], [0, 0], [2, 0], [2, 0], [4, 0], [4, 0]], dtype=float
    )
    assert cm.allan_variance_2d(centroids, 2) == pytest.approx(2.0)


def test_allan_requires_two_windows():
    assert np.isnan(cm.allan_variance_2d(np.zeros((3, 2)), 2))
    curve = cm.allan_curve(np.zeros((10, 2)), range(1, 11))
    assert [row["tau"] for row in curve] == [1, 2, 3, 4, 5]


def test_trajectory_from_masks(square_mask):
    masks = [square_mask(10, 10, 0, 0, 2), np.zeros((10, 10), dtype=bool)]
    trajectory = cm.trajectory_from_masks(masks)
    assert trajectory[0].tolist() == [0.5, 0.5]
    assert np.isnan(trajectory[1]).all()


def test_fb_residual_zero_for_consistent_translation():
    height, width = 20, 20
    forward = np.zeros((height, width, 2))
    forward[..., 0] = 2.0
    backward = np.zeros((height, width, 2))
    backward[..., 0] = -2.0

    residual = of.forward_backward_residual(forward, backward)
    assert np.nanmax(residual) == pytest.approx(0.0)
    assert np.isnan(residual[:, -1]).all()
    assert of.consistency_violation_rate(residual) == 0.0


def test_fb_residual_detects_inconsistency():
    forward = np.zeros((10, 10, 2))
    forward[..., 0] = 1.0
    backward = np.zeros((10, 10, 2))

    residual = of.forward_backward_residual(forward, backward)
    assert of.consistency_violation_rate(residual, 0.5) == 1.0


def test_flow_magnitude_and_summary():
    flow = np.zeros((4, 4, 2))
    flow[..., 0] = 3
    flow[..., 1] = 4

    summary = of.frame_pair_summary(flow)
    assert summary["mean_magnitude_px"] == 5.0
    assert summary["n_pixels"] == 16


def test_huber_location_is_robust():
    values = np.r_[np.ones(50), 1000.0]
    assert stats.huber_location(values, 1.0) < 2.0 < values.mean()
    assert stats.huber_loss(np.array([0.5, 2.0])).tolist() == [0.125, 1.5]
