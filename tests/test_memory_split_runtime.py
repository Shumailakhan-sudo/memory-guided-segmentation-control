import itertools

import numpy as np
import pytest

from memory_guided_vos import dataset_split as ds
from memory_guided_vos import memory_schedule as ms
from memory_guided_vos import records
from memory_guided_vos import runtime as rt


@pytest.mark.parametrize("u", ms.INTERVALS)
def test_update_frames(u):
    f = ms.update_frames(12, u)
    assert f[0] == 0 and np.all(np.diff(f) == u)
    assert all(ms.is_update_frame(int(t), u) for t in f)


def test_nominal_update_hz():
    assert [ms.nominal_update_hz(u) for u in ms.INTERVALS] == [24.0, 12.0, 8.0, 4.8]


def test_memory_mechanisms():
    assert ms.hydra_overrides("memory_temporal_stride", 3) == ["++model.memory_temporal_stride_for_eval=3"]
    with pytest.raises(NotImplementedError):
        ms.hydra_overrides("external_schedule", 3)


def test_complexity_weights_and_terciles():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 50))
    cs = ds.complexity_score(a, b, c)
    assert cs == pytest.approx(0.4 * ds.zscore(a) + 0.4 * ds.zscore(b) + 0.2 * ds.zscore(c))
    t = ds.complexity_terciles(cs)
    assert set(np.unique(t)) == {0, 1, 2}


def test_split_sizes_disjoint_and_deterministic():
    names = [f"seq{i:02d}" for i in range(50)]
    strata = np.repeat([0, 1, 2], [17, 17, 16])
    s1 = ds.stratified_split(names, strata, seed=42)
    s2 = ds.stratified_split(names, strata, seed=42)
    assert s1 == s2
    counts = {p: list(s1.values()).count(p) for p in ("development", "validation", "test")}
    assert counts == {"development": 30, "validation": 10, "test": 10}
    for p in counts:  # every partition sees every stratum
        assert {int(strata[names.index(n)]) for n, q in s1.items() if q == p} == {0, 1, 2}


def test_partition_leakage_detected():
    recs = [{"sequence": "bear", "partition": "development"}, {"sequence": "bear", "partition": "test"},
            {"sequence": "boat", "partition": "test"}]
    assert ds.check_partition_integrity({}, recs) == ["bear: ['development', 'test']"]


def test_stage_timer_with_fake_clock():
    ticks = itertools.count(0.0, 0.001)      # each clock read advances 1 ms
    synced = []
    t = rt.StageTimer(sync=lambda: synced.append(1), warmup_frames=1, clock=lambda: next(ticks))
    for _ in range(3):
        for s in rt.STAGES:
            with t.stage(s):
                pass
        t.end_frame()
    recs = t.records()
    assert len(recs) == 2 and all(r["total"] == pytest.approx(4.0) for r in recs)
    assert t.metadata()["synchronised"] and len(synced) == 24


def test_stage_timer_rejects_incomplete_frame():
    t = rt.StageTimer()
    with t.stage("inference"):
        pass
    with pytest.raises(RuntimeError):
        t.end_frame()


def test_latency_summary_deadline():
    s = rt.latency_summary(np.array([10.0, 10.0, 40.0, 10.0]), deadline_ms=33)
    assert s["nominal_fps"] == pytest.approx(1000 / 17.5)
    assert s["deadline_compliance"] == 0.75 and s["n_deadline_misses"] == 1


def test_record_schema_validation():
    assert records.validate_header("calibration_pairs.csv",
                                   ["sequence", "frame_index", "confidence", "jaccard", "nonempty_pred"]) == []
    assert records.validate_header("calibration_pairs.csv", ["sequence"]) != []
