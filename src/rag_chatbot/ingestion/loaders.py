from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
from docx import Document


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def load_documents(data_dir: str) -> list[dict[str, Any]]:
    root = Path(data_dir)
    docs: list[dict[str, Any]] = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            docs.extend(_load_pdf(path))
        elif path.suffix.lower() == ".docx":
            docs.extend(_load_docx(path))
        else:
            docs.extend(_load_text(path))

    return docs


def _load_pdf(path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "document_name": path.name,
                        "document_path": str(path),
                        "page": i,
                        "source_type": "pdf",
                    },
                }
            )
    return pages


def _load_docx(path: Path) -> list[dict[str, Any]]:
    document = Document(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [
        {
            "text": text,
            "metadata": {
                "document_name": path.name,
                "document_path": str(path),
                "page": 1,
                "source_type": "docx",
            },
        }
    ]


def _load_text(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [
        {
            "text": text,
            "metadata": {
                "document_name": path.name,
                "document_path": str(path),
                "page": 1,
                "source_type": "text",
            },
        }
    ]
