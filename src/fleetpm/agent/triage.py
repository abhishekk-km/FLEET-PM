import json
import math
import os
from pathlib import Path

ORDER = {"LOW": 0, "REVIEW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
REC = {"CRITICAL": "Remove from service immediately. Full inspection required.",
       "HIGH": "Schedule maintenance within 2 operational cycles.",   # window from plan
       "MEDIUM": "Schedule inspection within 10 cycles.",             # window from plan
       "REVIEW": "No risk estimate issued: prediction invalid or history too short. Manual data review required.",
       "LOW": None}


def _num(x, nd):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


class TriageAgent:
    """Deterministic rules. Inputs: conservative RUL, health index, persistence flag.
    Invalid input -> REVIEW, never LOW. Work orders are SIMULATED, deterministic by
    incident key, and de-duplicated across restarts by scanning the output directory."""

    def __init__(self, cfg, out_dir, subset=""):
        self.t = cfg["triage"]
        self.subset = subset
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.last = self._load_existing()

    def _load_existing(self):
        last = {}
        for p in sorted(self.out.glob("WO-*.json")):
            try:
                wo = json.loads(p.read_text())
                eid, sev = int(wo["engine_id"]), wo["severity"]
                rank = ORDER[sev]
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise RuntimeError(f"Unreadable work order {p.name}: {e!r}") from e
            if rank > ORDER[last.get(eid, "LOW")]:
                last[eid] = sev
        return last

    def classify(self, rul, hi, pers, history_ok=True):
        t = self.t
        try:
            rul, hi = float(rul), float(hi)
        except (TypeError, ValueError):
            return "REVIEW", "R0a:invalid_prediction"
        if not (math.isfinite(rul) and math.isfinite(hi)) or rul < 0:
            return "REVIEW", "R0a:invalid_prediction"
        if not history_ok:
            return "REVIEW", "R0b:insufficient_history"
        if rul <= t["critical_rul"]:
            return "CRITICAL", "R1a:rul<=critical_rul"
        if pers and hi >= t["critical_hi"] and rul <= t["high_rul"]:
            return "CRITICAL", "R1b:persistent&hi>=critical_hi&rul<=high_rul"
        if rul <= t["high_rul"]:
            return "HIGH", "R2a:rul<=high_rul"
        if pers and hi >= t["high_hi"]:
            return "HIGH", "R2b:persistent&hi>=high_hi"
        if rul <= t["medium_rul"]:
            return "MEDIUM", "R3a:rul<=medium_rul"
        if pers and hi >= t["medium_hi"]:
            return "MEDIUM", "R3b:persistent&hi>=medium_hi"
        return "LOW", "R4:none"

    def work_order(self, engine_id, cycle, rul_mean, rul_std, rul_cons, hi, pers,
                   top_sensors, model_version, history_ok=True):
        sev, rule = self.classify(rul_cons, hi, pers, history_ok)
        eid = int(engine_id)
        if sev == "LOW" or ORDER[sev] <= ORDER[self.last.get(eid, "LOW")]:
            return None  # none for healthy; none unless severity escalates (survives restarts)
        key = f"{self.subset or 'NA'}-ENG{eid:03d}-{sev}"
        wo = {"work_order_id": f"WO-{key}", "incident_key": key, "status": "OPEN",
              "simulated": True,
              "timestamp_note": "replay cycle index of a benchmark trajectory, not calendar time",
              "engine_id": eid, "current_cycle": int(cycle),
              "rul_mean": _num(rul_mean, 2), "rul_std": _num(rul_std, 2),
              "rul_conservative": _num(rul_cons, 2),
              "health_index": _num(hi, 3), "persistent_anomaly": bool(pers),
              "severity": sev, "rule_fired": rule, "recommendation": REC[sev],
              "requires_human_review": sev in ("CRITICAL", "REVIEW"), "auto_generated": True,
              "model_version": model_version, "top_degradation_sensors": top_sensors}
        path = self.out / f"{wo['work_order_id']}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(wo, indent=2))
        os.replace(tmp, path)
        self.last[eid] = sev
        return wo