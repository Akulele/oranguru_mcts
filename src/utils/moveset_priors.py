import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PRIORS_PATH = Path(__file__).resolve().parents[2] / "data" / "moveset_priors.json"


@lru_cache(maxsize=1)
def load_moveset_priors(path: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    target = Path(path) if path else DEFAULT_PRIORS_PATH
    if not target.exists():
        return {}
    try:
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}
