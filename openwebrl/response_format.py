from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BrowserResponseFormatMode:
    name: str
    policy_relpath: str
    thinking_open_tag: str
    thinking_close_tag: str
    requires_action_block: bool
    chat_template_enable_thinking: bool | None
    append_visible_thinking_prefill: bool


_BROWSER_RESPONSE_FORMAT_MODES: dict[str, BrowserResponseFormatMode] = {
    "slime": BrowserResponseFormatMode(
        name="slime",
        policy_relpath="env/prompts/system_prompt.md",
        thinking_open_tag="<thinking>",
        thinking_close_tag="</thinking>",
        requires_action_block=True,
        chat_template_enable_thinking=None,
        append_visible_thinking_prefill=False,
    ),
    "browser_env": BrowserResponseFormatMode(
        name="browser_env",
        policy_relpath="env/prompts/system_prompt_browser_env.md",
        thinking_open_tag="<think>",
        thinking_close_tag="</think>",
        requires_action_block=False,
        # Keep visible <think>...</think> text in model outputs so rollout
        # parsing and format reward stay aligned with the browser_env prompt.
        chat_template_enable_thinking=False,
        append_visible_thinking_prefill=True,
    ),
}

def _get_browser_env_chat_template_enable_thinking_override() -> bool | None:
    value = os.environ.get("SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING")
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _get_optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _get_browser_thinking_tag_overrides() -> tuple[str | None, str | None]:
    return (
        _get_optional_env("SLIME_BROWSER_THINKING_OPEN_TAG"),
        _get_optional_env("SLIME_BROWSER_THINKING_CLOSE_TAG"),
    )


def _get_append_visible_thinking_prefill_override() -> bool | None:
    value = os.environ.get("SLIME_BROWSER_APPEND_THINKING_PREFILL")
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def get_browser_response_format_mode(mode_name: str | None) -> BrowserResponseFormatMode:
    normalized = (mode_name or "slime").strip().lower().replace("-", "_")
    try:
        mode = _BROWSER_RESPONSE_FORMAT_MODES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_BROWSER_RESPONSE_FORMAT_MODES))
        raise ValueError(
            f"Unsupported browser_response_format_mode={mode_name!r}. Supported modes: {supported}."
        ) from exc
    if mode.name == "browser_env":
        override = _get_browser_env_chat_template_enable_thinking_override()
        if override is not None and override != mode.chat_template_enable_thinking:
            mode = replace(mode, chat_template_enable_thinking=override)
    open_tag_override, close_tag_override = _get_browser_thinking_tag_overrides()
    if open_tag_override and close_tag_override:
        mode = replace(
            mode,
            thinking_open_tag=open_tag_override,
            thinking_close_tag=close_tag_override,
        )
    prefill_override = _get_append_visible_thinking_prefill_override()
    if prefill_override is not None and prefill_override != mode.append_visible_thinking_prefill:
        mode = replace(mode, append_visible_thinking_prefill=prefill_override)
    return mode


def rewrite_policy_thinking_tags(
    policy: str,
    mode: BrowserResponseFormatMode,
) -> str:
    """
    Rewrite prompt text to use the currently configured thinking tags.
    """
    if not policy:
        return policy
    rewritten = policy
    rewritten = rewritten.replace("</thinking>", mode.thinking_close_tag)
    rewritten = rewritten.replace("</think>", mode.thinking_close_tag)
    rewritten = rewritten.replace("<thinking>", mode.thinking_open_tag)
    rewritten = rewritten.replace("<think>", mode.thinking_open_tag)
    return rewritten


def ensure_prompt_ends_with_visible_thinking_tag(
    prompt_text: str,
    mode: BrowserResponseFormatMode,
) -> str:
    """
    Ensure the generation prefill ends inside a visible thinking block.

    Some Qwen chat templates ignore `enable_thinking=True` and stop at
    `<|im_start|>assistant\\n`, so browser rollouts never see a visible
    `<think>` / `<thinking>` prefix unless we append it explicitly.

    This helper is intentionally tolerant:
    - If the prompt already ends with the currently configured thinking open
      tag (optionally followed by whitespace/newlines), it leaves the prefix
      in place.
    - Otherwise, it appends the mode-specific opening tag plus a trailing
      newline so generation starts inside the thinking block.
    """
    if not mode.append_visible_thinking_prefill:
        return prompt_text
    stripped = prompt_text.rstrip()
    if stripped.endswith(mode.thinking_open_tag):
        return stripped + "\n"
    return f"{stripped}\n{mode.thinking_open_tag}\n"
