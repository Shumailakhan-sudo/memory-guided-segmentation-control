import numpy as np
import pytest

from memory_guided_vos import visibility as vis


def test_area_ratio_uses_first_positive_area():
    r, ref = vis.area_ratio(np.array([0, 0, 100, 50, 2]))
    assert ref == 2
    assert np.isnan(r[0]) and r[2] == 1.0 and r[3] == 0.5 and r[4] == 0.02


def test_low_visibility_threshold_strict():
    r = np.array([1.0, 0.05, 0.0499, np.nan])
    assert vis.low_visibility(r).tolist() == [False, False, True, False]


def test_confirmation_indicator_eq9():
    b = np.array([0, 1, 1, 1, 1, 0, 1, 1, 0], bool)
    o = vis.confirmation_indicator(b, beta=3)
    assert np.flatnonzero(o).tolist() == [3]        # confirmed once, at t = start + beta - 1


def test_confirmation_from_frame_zero():
    b = np.array([1, 1, 1, 0], bool)
    assert np.flatnonzero(vis.confirmation_indicator(b, 3)).tolist() == [2]


def test_run_classification_and_duration():
    b = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], bool)
    runs = vis.extract_runs(b, beta=3)
    assert [(r.start, r.end, r.category) for r in runs] == [
        (1, 2, "short"), (4, 7, "confirmed"), (9, 11, "censored")]
    assert runs[1].duration == 4                    # G = t_end - t_start + 1
    assert runs[1].restoration_frame == 8
    assert runs[2].restoration_frame is None        # right-censored


def test_short_run_at_end_is_unclassifiable():
    runs = vis.extract_runs(np.array([0, 0, 1], bool), beta=3)
    assert runs[0].category == "short_censored"


def test_frames_to_ms():
    assert vis.frames_to_ms(2.10) == pytest.approx(87.5)
    assert vis.frames_to_ms(11) == pytest.approx(458.333, abs=1e-3)


def test_cantelli_matches_formula():
    # P(D >= 6) with mean 2.10, sd 1.02: 1.02^2 / (1.02^2 + 3.9^2)
    assert vis.cantelli_upper_bound(2.10, 1.02, 6) == pytest.approx(1.0404 / (1.0404 + 15.21))
    assert vis.cantelli_upper_bound(2.0, 1.0, 1.0) == 1.0


def test_reacquisition_delay_needs_threshold():
    j = np.array([0.9, 0.0, 0.0, 0.2, 0.6, 0.8])
    assert vis.reacquisition_delay(j, t_vis=3, eta_rec=0.5) == 1
    assert vis.reacquisition_delay(j, t_vis=3, eta_rec=0.95) is None


def test_kaplan_meier_simple():
    t, s = vis.survival_curve(np.array([3, 4, 5]), np.array([1, 1, 0]))
    assert t.tolist() == [3, 4]
    assert s.tolist() == pytest.approx([2 / 3, 1 / 3])
