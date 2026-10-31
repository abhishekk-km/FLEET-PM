from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["root"] = ROOT
    return cfg


def art(cfg, *parts):
    p = (ROOT / cfg["paths"]["artifacts"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p