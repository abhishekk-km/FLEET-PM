import numpy as np, pandas as pd
from fleetpm.models.metrics import nasa_score, cut_rows, last_rows
from fleetpm.models.anomaly import persistent
from fleetpm.features.rolling import build_features
from fleetpm.agent.triage import TriageAgent

CFG = {"triage": dict(critical_rul=15, high_rul=30, medium_rul=60, medium_hi=1, high_hi=2, critical_hi=4)}


def test_nasa():
    e = np.exp(1) - 1
    assert nasa_score([10], [10]) == 0
    assert abs(nasa_score([10], [20]) - e) < 1e-9      # late: d=+10
    assert abs(nasa_score([20], [7]) - e) < 1e-9       # early: d=-13


def test_cut_rows():
    eng = np.repeat([1, 2, 3], [40, 35, 50]); r = cut_rows(eng, 30, np.random.default_rng(0))
    assert all(eng[x] == e for x, e in zip(r, [1, 2, 3]))
    assert (np.array([x - s for x, s in zip(r, [0, 40, 75])]) + 1 >= 30).all()
    assert list(last_rows(eng)) == [39, 74, 124]


def test_persistent():
    f = np.array([1, 1, 0, 1, 1, 1, 1], bool); e = np.ones(7, int)
    assert persistent(f, e, 3).tolist() == [False, False, False, False, False, True, True]


def test_slope_exact_and_causal():
    t = np.arange(60.0); Z = pd.DataFrame({"engine_id": 1, "cycle": t + 1, "s": 0.5 * t + 3})
    F = build_features(Z, ["s"], 30, ["mean", "slope"])
    assert np.allclose(F["s__slope"].iloc[29:], 0.5)
    F2 = build_features(Z.iloc[:40], ["s"], 30, ["mean", "slope"])
    assert np.allclose(F.iloc[:40][["s__mean", "s__slope"]].to_numpy(), F2[["s__mean", "s__slope"]].to_numpy())


def test_triage():
    a = TriageAgent(CFG, "/tmp/wo_test")
    assert a.classify(10, 0, False)[0] == "CRITICAL"
    assert a.classify(100, 5, True)[0] == "HIGH"       # anomaly alone is capped at HIGH
    assert a.classify(25, 5, True)[0] == "CRITICAL"    # corroborated by RUL
    assert a.classify(100, 5, False)[0] == "LOW"      # anomaly alone, non-persistent, never escalates
    assert a.classify(25, 0, False)[0] == "HIGH"
    assert a.classify(50, 0, False)[0] == "MEDIUM"

import pytest


@pytest.mark.parametrize("rul,hi", [(float("nan"), 0.0), (50.0, float("nan")), (-1.0, 0.0), (float("inf"), 0.0)])
def test_invalid_never_low(tmp_path, rul, hi):
    assert TriageAgent(CFG, tmp_path).classify(rul, hi, False)[0] == "REVIEW"


def test_short_history_is_review(tmp_path):
    assert TriageAgent(CFG, tmp_path).classify(10, 0, False, False)[0] == "REVIEW"


def _wo(a, rul=10.0, cycle=100):
    return a.work_order(7, cycle, rul, 1.0, rul, 0.5, False, ["sensor_11"], "v-test")


def test_work_order_idempotent_and_restart_safe(tmp_path):
    a = TriageAgent(CFG, tmp_path, subset="FDX")
    w = _wo(a)
    assert w is not None and w["simulated"] is True and w["work_order_id"] == "WO-FDX-ENG007-CRITICAL"
    assert _wo(a) is None
    assert _wo(TriageAgent(CFG, tmp_path, subset="FDX")) is None      # restart
    assert len(list(tmp_path.glob("WO-*.json"))) == 1


def test_escalation_issues_new_order(tmp_path):
    a = TriageAgent(CFG, tmp_path, subset="FDX")
    assert _wo(a, rul=25.0)["severity"] == "HIGH"
    assert _wo(a, rul=10.0)["severity"] == "CRITICAL"
    assert len(list(tmp_path.glob("WO-*.json"))) == 2


def test_invalid_prediction_creates_review_order(tmp_path):
    nan = float("nan")
    wo = TriageAgent(CFG, tmp_path, subset="FDX").work_order(1, 50, nan, nan, nan, 0.0, False, [], "v")
    assert wo["severity"] == "REVIEW" and wo["rul_conservative"] is None
