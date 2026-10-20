import numpy as np
import pytest

from memory_guided_vos import calibration as cal
from memory_guided_vos import confidence_gating as cg
from memory_guided_vos import statistics as stats


def test_bin_boundaries():
    confidence = np.array([0.0, 0.0999, 0.1, 0.65, 0.9999, 1.0])
    assert cal.bin_index(confidence).tolist() == [0, 0, 1, 6, 9, 9]

    with pytest.raises(ValueError):
        cal.bin_index(np.array([1.2]))


def test_ece_known_value():
    confidence = np.array([0.95, 0.95, 0.55, 0.55])
    labels = np.array([1, 1, 1, 0])
    assert cal.expected_calibration_error(confidence, labels) == pytest.approx(0.05)


def test_ece_perfect_calibration_is_zero():
    confidence = np.ones(10)
    labels = np.ones(10, dtype=int)
    assert cal.expected_calibration_error(confidence, labels) == 0.0


def test_reliability_table_counts():
    confidence = np.array([0.05, 0.85, 0.86, 0.91])
    labels = np.array([0, 1, 0, 1])
    bins = {row["bin"]: row for row in cal.reliability_table(confidence, labels)}
    assert bins[8]["n"] == 2
    assert bins[8]["n_incorrect"] == 1
    assert bins[3]["n"] == 0


def test_gate_requires_threshold_and_nonempty_mask():
    confidence = np.array([0.7, 0.7, 0.5])
    nonempty = np.array([True, False, True])
    assert cg.gate(confidence, nonempty, 0.65).tolist() == [True, False, False]


def test_gate_metrics_from_counts_matches_definitions():
    metrics = cg.gate_metrics_from_counts(ac=6, aw=2, dc=3, dw=4)
    assert metrics["coverage"] == pytest.approx(8 / 15)
    assert metrics["error_rejection"] == pytest.approx(4 / 6)
    assert metrics["false_acceptance"] == pytest.approx(2 / 6)
    assert metrics["accepted_error"] == pytest.approx(2 / 8)


def test_gate_metrics_undefined_denominators():
    metrics = cg.gate_metrics_from_counts(ac=0, aw=0, dc=5, dw=0)
    assert metrics["error_rejection"] is None
    assert metrics["accepted_error"] is None


def test_gate_metrics_end_to_end():
    labels = np.array([1, 1, 0, 0])
    confidence = np.array([0.9, 0.3, 0.8, 0.2])
    metrics = cg.gate_metrics(labels, confidence, np.ones(4, dtype=bool), 0.5)
    assert (
        metrics["accepted_correct"],
        metrics["accepted_wrong"],
        metrics["deferred_correct"],
        metrics["deferred_wrong"],
    ) == (1, 1, 1, 1)


def test_cluster_bootstrap_resamples_whole_clusters():
    clusters = np.repeat(["a", "b", "c"], [2, 3, 5])
    sample_sizes = []

    def statistic(indices):
        sample_sizes.append(indices.size)
        return float(indices.size)

    result = stats.cluster_bootstrap(clusters, statistic, n_resamples=200, seed=1)
    possible_sizes = {
        a + b + c
        for a in (2, 3, 5)
        for b in (2, 3, 5)
        for c in (2, 3, 5)
    }
    assert set(sample_sizes[1:]) <= possible_sizes
    assert result["n_clusters"] == 3
    assert result["n_resamples"] == 200


def test_cluster_bootstrap_counts_undefined():
    result = stats.cluster_bootstrap(["a", "b"], lambda indices: None, n_resamples=10)
    assert result["n_undefined_resamples"] == 10
    assert np.isnan(result["ci_low"])


def test_cluster_bootstrap_deterministic():
    data = np.arange(10.0)
    clusters = np.repeat(np.arange(5), 2)

    def statistic(indices):
        return float(data[indices].mean())

    first = stats.cluster_bootstrap(clusters, statistic, 100, seed=7)
    second = stats.cluster_bootstrap(clusters, statistic, 100, seed=7)
    assert first == second
