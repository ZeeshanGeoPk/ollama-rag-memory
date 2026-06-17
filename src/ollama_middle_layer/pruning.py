from __future__ import annotations

import hashlib
import math
import re


FILLER_PATTERNS = [
    re.compile(r"^\s*(hello|hi|hey)[!. ]*$", re.IGNORECASE),
    re.compile(r"^\s*how can i help you( today)?[?.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*i('?m| am) happy to help[!. ]*$", re.IGNORECASE),
]


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 160) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def remove_filler_and_duplicates(chunks: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for chunk in chunks:
        sentences = []
        for sentence in split_sentences(chunk):
            if any(pattern.match(sentence) for pattern in FILLER_PATTERNS):
                continue
            sentences.append(sentence)
        cleaned = "\n".join(sentences).strip()
        if not cleaned:
            continue
        fingerprint = content_hash(cleaned)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(cleaned)
    return kept


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def prune_by_budget(sections: list[str], max_tokens: int) -> list[str]:
    kept: list[str] = []
    used = 0
    for section in sections:
        cost = estimate_tokens(section)
        if used + cost > max_tokens:
            remaining = max_tokens - used
            if remaining > 20:
                kept.append(section[: remaining * 4].rstrip())
            break
        kept.append(section)
        used += cost
    return kept
