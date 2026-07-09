"""
Shared helpers for the per-benchmark evaluation protocols in this package.

These are kept here (rather than in ``openwebrl/reward_browser.py`` or
``openwebrl/base/utils.py``) so the benchmark evaluators can be dropped in
without modifying the training-reward path. The scoring-relevant pieces
(``_extract_final_answer``, the tool list, parser selection) mirror the slime
browser evaluators so the protocols behave identically to the reference.
"""

import functools
import json
import os

from openwebrl.base.utils import ToolParser

# Load the tool schema so evaluators can build a ToolParser that understands
# the browser agent's tool calls (used to extract the final ``done`` answer).
_TOOL_LIST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "env", "prompts", "tool_list.json"
)
with open(_TOOL_LIST_PATH, "r") as _f:
    _TOOLS_INFO = json.load(_f)
    VALID_TOOL_NAMES = {tool["function"]["name"] for tool in _TOOLS_INFO}


@functools.lru_cache(maxsize=None)
def resolve_parser_type(name_or_path: str, override: str | None = None) -> str:
    """Pick the tool-call parser for a checkpoint.

    Prefers an explicit override; otherwise reads the model config's
    ``model_type`` so detection works no matter how the checkpoint dir is
    named (the base model type is often absent from the path). Cached so the
    config is read from disk only once per checkpoint.

    Returns:
        "qwen3_coder" for Qwen3.5-family models, else "qwen25".
    """
    if override and override != "auto":
        return override
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", "") or ""
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            model_type = model_type or (getattr(text_cfg, "model_type", "") or "")
        if "qwen3_5" in model_type.lower():  # qwen3_5 / qwen3_5_text
            return "qwen3_coder"
    except Exception as e:
        print(f"resolve_parser_type: config probe failed for {name_or_path}: {e}")
    return "qwen25"


def _extract_final_answer(_tool_parser: ToolParser, response: str) -> str:
    """Return the ``response`` argument of a trailing ``done`` tool call, or ""."""
    parsed = _tool_parser.parse(response)
    if parsed.success and parsed.calls:
        last_call = parsed.calls[-1]
        if last_call.name != "done":
            return ""
        params = (
            json.loads(last_call.parameters)
            if isinstance(last_call.parameters, str)
            else last_call.parameters
        )
        return params.get("response", "").strip()
    return ""
