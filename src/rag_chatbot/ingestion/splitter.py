from __future__ import annotations

from typing import Any

import tiktoken


def _tokenize(text: str, encoding_name: str = "cl100k_base") -> list[int]:
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)


def _detokenize(tokens: list[int], encoding_name: str = "cl100k_base") -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.decode(tokens)


def chunk_text(
    text: str,
    base_metadata: dict[str, Any],
    chunk_size: int = 512,
    overlap_ratio: float = 0.12,
) -> list[dict[str, Any]]:
    tokens = _tokenize(text)
    if not tokens:
        return []

    overlap_tokens = max(1, int(chunk_size * overlap_ratio))
    step = max(1, chunk_size - overlap_tokens)

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_value = _detokenize(chunk_tokens).strip()
        if chunk_text_value:
            metadata = {
                **base_metadata,
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": end,
                "token_count": len(chunk_tokens),
            }
            chunks.append({"text": chunk_text_value, "metadata": metadata})
            chunk_index += 1
        start += step

    return chunks
