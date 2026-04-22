from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_query(path: str, payload: dict[str, Any]) -> None:
    log_file = Path(path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    payload_with_time = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload_with_time, ensure_ascii=True) + "\n")
