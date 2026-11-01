from pathlib import Path

import yaml

# config.py lives in src/credit_card_default/, so the project root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml (by default the one at the project root) into a dict."""
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)