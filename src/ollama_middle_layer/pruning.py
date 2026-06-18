from __future__ import annotations

import hashlib
import math
import re


FILLER_PATTERNS = [
    re.compile(r"^\s*(hello|hi|hey)[!. ]*$", re.IGNORECASE),
    re.compile(r"^\s*how can i help you( today)?[?.! ]*$", re.IGNORECASE),
    re.compile(r"^\s*i('?m| am) happy to help[!. ]*$", re.IGNORECASE),
]

GLOBAL_HISTORY_PATTERNS = [
    re.compile(r"\bwhole (chat|conversation|history)\b", re.IGNORECASE),
    re.compile(r"\b(entire|all) (chat|conversation|history)\b", re.IGNORECASE),
    re.compile(r"\bwhat (did|have) i (ask|asked|say|said)\b", re.IGNORECASE),
    re.compile(r"\bsummar(?:y|ize|ise).*\b(chat|conversation|history)\b", re.IGNORECASE),
    re.compile(r"\brecap.*\b(chat|conversation|history)\b", re.IGNORECASE),
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

    units = re.split(r"(?<=\n)\s*\n+|(?<=[.!?])\s+(?=[A-Z0-9*`])", cleaned)
    chunks: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(unit, max_chars, overlap))
            continue
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:].strip()
        current = f"{tail}\n{unit}".strip() if tail else unit
    if current:
        chunks.append(current)
    return chunks


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


def is_global_history_query(text: str) -> bool:
    return any(pattern.search(text) for pattern in GLOBAL_HISTORY_PATTERNS)


def relevance_score(
    query: str,
    sentence: str,
    query_embedding: list[float],
    sentence_embedding: list[float],
) -> float:
    semantic = (cosine_similarity(query_embedding, sentence_embedding) + 1.0) / 2.0
    query_terms = _meaningful_terms(query)
    sentence_terms = _meaningful_terms(sentence)
    lexical = (
        len(query_terms & sentence_terms) / len(query_terms)
        if query_terms
        else 0.0
    )
    return min(1.0, max(0.0, semantic * 0.85 + lexical * 0.15))


def compact_context(text: str) -> str:
    """Compact formatting while preserving words, code, and ordering."""
    compacted: list[str] = []
    blank_pending = False
    in_code = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            compacted.append(line.strip())
            blank_pending = False
            continue
        if not line.strip():
            blank_pending = bool(compacted)
            continue
        if blank_pending:
            compacted.append("")
            blank_pending = False
        if in_code:
            compacted.append(line)
        else:
            normalized = re.sub(r"[ \t]+", " ", line.strip())
            normalized = re.sub(r"^(?:\$\s*){2,}", "$ ", normalized)
            compacted.append(normalized)
    return "\n".join(compacted).strip()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _meaningful_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9_./:-]{3,}", text.lower())
        if term not in {"the", "and", "that", "this", "with", "from", "for", "you", "your"}
    }


def normalized_sentence(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_near_duplicate(left: str, right: str, threshold: float = 0.68) -> bool:
    left_terms = set(normalized_sentence(left).split())
    right_terms = set(normalized_sentence(right).split())
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms)
    containment = overlap / min(len(left_terms), len(right_terms))
    union = overlap / len(left_terms | right_terms)
    return containment >= 0.85 or union >= threshold


def limit_items_by_budget(
    items: list[str],
    max_tokens: int,
    separator: str = "\n",
) -> list[str]:
    kept: list[str] = []
    used = 0
    for item in items:
        cost = estimate_tokens(item)
        if used + cost > max_tokens:
            continue
        kept.append(item)
        used += cost + estimate_tokens(separator)
    return kept


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


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]
