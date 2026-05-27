from __future__ import annotations

DEFAULT_BROWSER_ACTUAL_VALUE_MAX_CHARS = 512
DEFAULT_BROWSER_HISTORY_ENV_FEEDBACK_MAX_CHARS = 300
DEFAULT_BROWSER_TOOL_RESPONSE_MAX_CHARS = 512


def normalize_feedback_text(text: object) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def truncate_feedback_text(
    text: object,
    max_chars: int,
    *,
    head_ratio: float = 0.7,
) -> str:
    normalized = normalize_feedback_text(text)
    max_chars = int(max_chars or 0)
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized

    marker = f" ... [truncated {len(normalized) - max_chars} chars] ... "
    if max_chars <= len(marker):
        return normalized[:max_chars]

    head_chars = max(int((max_chars - len(marker)) * head_ratio), 1)
    tail_chars = max(max_chars - len(marker) - head_chars, 1)
    return normalized[:head_chars] + marker + normalized[-tail_chars:]
