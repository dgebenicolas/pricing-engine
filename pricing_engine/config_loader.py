"""Locate and read config files. No actuarial meaning lives here."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "config"


def config_path(name: str) -> Path:
    """Resolve a file inside pricing_engine/config/; raise if missing."""
    path = CONFIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file and return a dict (an empty file gives {})."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
