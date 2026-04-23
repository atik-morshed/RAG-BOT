from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    port = os.getenv("PORT", "8501")
    api_port = os.getenv("API_PORT", "8001")

    # Keep runtime state on persistent disk when available.
    data_root = Path(os.getenv("DATA_ROOT", "/data"))
    uploads_dir = data_root / "uploads"
    chroma_dir = data_root / "chroma"
    logs_dir = data_root / "logs"

    uploads_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CHROMA_MODE", "persistent")
    env.setdefault("CHROMA_PERSIST_DIR", str(chroma_dir))
    env.setdefault("QUERY_LOG_PATH", str(logs_dir / "query_log.jsonl"))
    env.setdefault("UPLOAD_DIR", str(uploads_dir))
    env.setdefault("API_BASE_URL", f"http://127.0.0.1:{api_port}")

    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rag_chatbot.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            api_port,
        ],
        env=env,
    )

    try:
        ui_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/rag_chatbot/ui/app.py",
                "--server.port",
                port,
                "--server.address",
                "0.0.0.0",
            ],
            env=env,
            check=False,
        )
        return ui_process.returncode
    finally:
        api_process.terminate()
        try:
            api_process.wait(timeout=10)
        except Exception:
            api_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
