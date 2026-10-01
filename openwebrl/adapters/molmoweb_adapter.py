"""Rollouts in the molmoweb `computer_use` tool-call format.

The checkpoints we now train from (`baseline_3_ft` and its lineage) were SFT'd on
molmoweb's screenshot agent format: a one-word system prompt, plain-text reasoning
followed by a single `<tool_call>` block calling `computer_use`, and 0-1000 display
coordinates. OpenWebRL's own prompt is a different action space and tool set, so RL
on those checkpoints has to speak this one or every action is rejected.

This mirrors molmoweb's `agent/fara15_vllm.py` (served as `agent_type=fara15_mw`)
turn for turn:
  - user turn  = screenshot, newline, then text; step 0 carries the task, later steps
                 "Current URL: ...\n<observation>\nHere is the next screenshot. ..."
  - observation = the first-person sentence the action mapper produces, with click
                 coordinates reported in screenshot pixels
  - coordinates go to the env untouched: env/config.yaml sets resize_scale 1000, so
    WebEnv already reads point_2d as a 0-1000 space and scales it to the screenshot.

Two deliberate deviations from molmoweb, both because this env's action set differs:
  - `type` maps onto WebEnv's `write`, which selects-all before typing (it replaces
    the field contents instead of appending).
  - actions with no env counterpart (`pause_and_memorize_fact`,
    `read_page_answer_question`, `ask_user_question`, Control+F) execute nothing and
    only produce their observation text, exactly as molmoweb's no-ops do.
"""

from __future__ import annotations

import ast
import base64
import json
import re
from typing import Any
from urllib.parse import quote_plus

from openwebrl.adapters.browser_adapter import BrowserAdapter
from openwebrl.base.types import ParsedToolResult, ToolCall

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
# Coordinate space the model emits, and what WebEnv's resize_scale expects.
DISPLAY_SIZE = 1000
USER_MESSAGE = "Here is the next screenshot. Think about what to do next."
AUTO_USER_REPLY = "keep going so far as you don't make up information, you have my approval"
MAX_URL_LENGTH = 100
NO_TOOL_CALL_OBSERVATION = "My last response did not contain a valid tool call."
DEFAULT_SYSTEM_MESSAGE = "molmoweb"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# CUA key names -> Playwright key names (molmoweb agent/fara15_vllm.py).
CUA_KEY_TO_PLAYWRIGHT_KEY = {
    "/": "Divide",
    "\\": "Backslash",
    "alt": "Alt",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "up": "ArrowUp",
    "backspace": "Backspace",
    "capslock": "CapsLock",
    "cmd": "Meta",
    "ctrl": "Control",
    "delete": "Delete",
    "end": "End",
    "enter": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "shift": "Shift",
    "space": " ",
    "super": "Meta",
    "tab": "Tab",
    "win": "Meta",
}

# Chords that address browser chrome rather than the page; Playwright delivers key
# events to the page only, so they are dispatched as browser actions instead.
GO_BACK_CHORDS = (frozenset({"Alt", "ArrowLeft"}),)
REFRESH_CHORDS = (
    frozenset({"Control", "r"}),
    frozenset({"Control", "Shift", "r"}),
    frozenset({"F5"}),
)
FIND_CHORD = frozenset({"Control", "f"})


def _trimmed_url(url: str, max_length: int = MAX_URL_LENGTH) -> str:
    return url if len(url) <= max_length else url[:max_length] + "..."


def _utf8_safe(text: str) -> str:
    """Drop lone surrogates; a half-formed pair kills the whole trajectory later."""
    return text.encode("utf-8", "replace").decode("utf-8")


def parse_molmoweb_output(raw: str) -> tuple[str, dict]:
    """Split a raw response into (thoughts, `computer_use` arguments)."""
    match = _TOOL_CALL_RE.search(raw)
    if match is None:
        return raw.strip(), {}
    thoughts = raw[: match.start()].strip()
    body = match.group(1)
    try:
        call = json.loads(body)
    except json.JSONDecodeError:
        try:
            call = ast.literal_eval(body)
        except Exception:
            return thoughts, {}
    if not isinstance(call, dict):
        return thoughts, {}
    args = call.get("arguments", {})
    return thoughts, args if isinstance(args, dict) else {}


def _map_keys(keys: Any) -> list[str]:
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, (list, tuple)):
        return []
    return [CUA_KEY_TO_PLAYWRIGHT_KEY.get(str(k).lower(), str(k)) for k in keys]


def _chord_set(mapped: list[str]) -> frozenset[str]:
    return frozenset(k if len(k) > 1 else k.lower() for k in mapped)


def _normalize_url(url: str) -> str:
    if url.startswith(("https://", "http://", "file://", "about:")):
        return url
    if " " in url:
        return f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH"
    return "https://" + url


class MolmowebToolParser:
    """Stands in for ToolParser: one `computer_use` call per response."""

    def parse(self, text: str) -> ParsedToolResult:
        thoughts, args = parse_molmoweb_output(text or "")
        if not args:
            return ParsedToolResult(success=False, normal_text=thoughts, calls=[],
                                    error="no parseable computer_use call")
        return ParsedToolResult(
            success=True,
            normal_text=thoughts,
            calls=[ToolCall(name="computer_use", parameters=args)],
        )


class MolmowebAdapter(BrowserAdapter):
    """Browser adapter that prompts and parses in molmoweb's tool-call format."""

    def __init__(self, config, rollout_args: Any):
        super().__init__(config, rollout_args)
        self._parser = MolmowebToolParser()
        self._cursor: tuple[float, float] = (0.0, 0.0)
        self._current_url = ""
        self._screen_size = (1280, 720)
        self._pending_observation = ""
        self._pending_user_response = ""
        # Benchmarks have no user in the loop; molmoweb's eval answers questions itself.
        self._auto_user_reply = True

    # -- prompt ------------------------------------------------------------
    @property
    def tool_parser(self) -> Any:
        return self._parser

    def parse_env_info(self, info: dict[str, Any]) -> tuple[list[dict], str]:
        """No tools block in the prompt: the schema lived in the system message the
        SFT replaced with a one-word sentinel."""
        policy = (info.get("policy") or "").strip() or DEFAULT_SYSTEM_MESSAGE
        return [], policy

    def _remember_observation_state(self, observation: dict[str, Any] | None) -> None:
        if not observation:
            return
        url = observation.get("active_tab_url") or ""
        if url:
            self._current_url = url
        size = observation.get("screen_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                self._screen_size = (int(size[0]), int(size[1]))
            except (TypeError, ValueError):
                pass

    def _user_turn(self, observation: dict[str, Any] | None, text: str) -> dict[str, Any]:
        self._remember_observation_state(observation)
        text = _utf8_safe(text)
        screenshot = (observation or {}).get("screenshot")
        if not isinstance(screenshot, bytes):
            return {"role": "user", "content": [{"type": "text", "text": text}]}
        screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{IMAGE_PLACEHOLDER}\n{text}"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{screenshot_b64}"},
            ],
        }

    def _get_latest_user_message(
        self, user_intent: str, observation: dict[str, Any], step_id: int = None
    ) -> dict[str, Any]:
        return self._user_turn(observation, user_intent)

    def _convert_tool_responses_to_msgs(
        self,
        tool_responses: list[dict],
        observation: dict[str, Any] | None = None,
        include_tool_response: bool = True,
    ) -> list[dict[str, Any]]:
        """Next user turn: current URL, what the last action did, then the prompt."""
        self._remember_observation_state(observation)
        url = self._current_url
        prefix = f"Current URL: {_trimmed_url(url)}\n" if url else ""

        if self._pending_user_response:
            text = prefix + self._pending_user_response
            self._pending_user_response = ""
        else:
            observation_text = self._pending_observation
            failure = self._failure_text(tool_responses)
            if failure:
                observation_text = f"{observation_text} {failure}".strip()
            text = f"{prefix}{observation_text}\n{USER_MESSAGE}" if observation_text else prefix + USER_MESSAGE
        self._pending_observation = ""
        return [self._user_turn(observation, text)]

    def _failure_text(self, tool_responses: list[dict] | None) -> str:
        """The mapper's observation is optimistic; surface env-side failures so a
        rejected action is not silently retried for the rest of the episode."""
        lines = []
        for item in tool_responses or []:
            if not isinstance(item, dict):
                continue
            response = str(item.get("tool_response", ""))
            if response.startswith(("Failed", "Error")):
                lines.append(self._truncate_tool_response(response))
        return " ".join(lines)

    # -- actions -----------------------------------------------------------
    def actions_from_parsed(self, parsed_calls: list[ToolCall]) -> list[dict[str, Any]]:
        if not parsed_calls:
            self._pending_observation = NO_TOOL_CALL_OBSERVATION
            return []
        args = parsed_calls[0].parameters
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        actions, observation = self._map_action(args if isinstance(args, dict) else {})
        self._pending_observation = observation
        return actions

    def _to_pixels(self, x: float, y: float) -> tuple[float, float]:
        """Display space -> screenshot pixels, for the observation text only."""
        width, height = self._screen_size
        px = max(0, min(round(x / DISPLAY_SIZE * width), width - 1))
        py = max(0, min(round(y / DISPLAY_SIZE * height), height - 1))
        return float(px), float(py)

    @staticmethod
    def _point(coordinate: Any) -> tuple[float, float] | None:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            return None
        try:
            return float(coordinate[0]), float(coordinate[1])
        except (TypeError, ValueError):
            return None

    def _map_action(self, args: dict) -> tuple[list[dict[str, Any]], str]:
        """One `computer_use` call -> (WebEnv actions, observation text)."""
        action = str(args.get("action", ""))

        if action in ("left_click", "right_click", "double_click", "triple_click",
                      "mouse_move", "left_click_drag"):
            point = self._point(args.get("coordinate"))
            if point is None:
                return [], f"{action} requires a 2-element 'coordinate', got {args.get('coordinate')!r}"
            x, y = point
            px, py = self._to_pixels(x, y)

            if action == "mouse_move":
                self._cursor = (x, y)
                return [{"name": "hover", "args": {"point_2d": [x, y]}}], f"I moved the cursor to ({px}, {py})."
            if action == "left_click_drag":
                from_x, from_y = self._cursor
                self._cursor = (x, y)
                return (
                    [{"name": "drag", "args": {"start_point_2d": [from_x, from_y], "end_point_2d": [x, y]}}],
                    f"I dragged to ({px}, {py}).",
                )

            self._cursor = (x, y)
            clicks = {"double_click": 2, "triple_click": 3}.get(action, 1)
            button = "right" if action == "right_click" else "left"
            verb = {
                "left_click": "clicked",
                "right_click": "right-clicked",
                "double_click": "double-clicked",
                "triple_click": "triple-clicked",
            }[action]
            return (
                [{"name": "click", "args": {"point_2d": [x, y], "clicks": clicks, "button": button}}],
                f"I {verb} at coordinates ({px}, {py}).",
            )

        if action == "key":
            mapped = _map_keys(args.get("keys"))
            if not mapped:
                return [], f"key requires a non-empty 'keys' list, got {args.get('keys')!r}"
            keyset = _chord_set(mapped)
            if keyset in GO_BACK_CHORDS:
                return [{"name": "go_back", "args": {}}], "I clicked the browser back button."
            if keyset in REFRESH_CHORDS:
                if not self._current_url:
                    return [], "cannot reload: the current URL is unknown"
                return [{"name": "goto_url", "args": {"url": self._current_url}}], "I reloaded the page."
            if keyset == FIND_CHORD:
                return [], (
                    "Find-in-page is not available in this browser, so nothing happened. "
                    "Scroll the page or use the site's own search box instead."
                )
            chord = "+".join(mapped)
            return (
                [{"name": "press_keys", "args": {"keys": mapped, "is_hotkey": len(mapped) > 1}}],
                f"I pressed the following keys: {chord}",
            )

        if action == "type":
            text = str(args.get("text", args.get("text_value", "")))
            if not text:
                return [], "type requires a non-empty 'text'"
            return [{"name": "write", "args": {"message": text}}], f"I typed '{text}'."

        if action in ("scroll", "hscroll"):
            try:
                pixels = float(args.get("pixels", 0))
            except (TypeError, ValueError):
                pixels = 0.0
            amount = max(0.0, min(1.0, abs(pixels) / DISPLAY_SIZE))
            if action == "scroll":
                # `pixels` is positive-up in display space; WebEnv takes a direction
                # plus a viewport fraction.
                direction = "up" if pixels > 0 else "down"
                return (
                    [{"name": "scroll", "args": {"direction": direction, "amount": amount}}],
                    f"I scrolled {direction}.",
                )
            direction = "right" if pixels > 0 else "left"
            delta_x = int(pixels * self._screen_size[0] / DISPLAY_SIZE)
            return (
                [{"name": "scroll", "args": {"direction": direction, "amount": amount}}],
                f"I scrolled horizontally by {delta_x} pixels.",
            )

        if action == "visit_url":
            url = str(args.get("url", ""))
            if not url:
                return [], "visit_url requires a 'url'"
            return [{"name": "goto_url", "args": {"url": _normalize_url(url)}}], f"I navigated to {url}."

        if action == "history_back":
            return [{"name": "go_back", "args": {}}], "I clicked the browser back button."

        if action == "web_search":
            query = str(args.get("query", ""))
            url = f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH"
            return [{"name": "goto_url", "args": {"url": url}}], f"I searched for '{query}'."

        if action == "wait":
            try:
                seconds = int(float(args.get("time", 3)))
            except (TypeError, ValueError):
                seconds = 3
            seconds = max(1, min(seconds, 10))
            return [{"name": "wait", "args": {"seconds": seconds}}], "I waited for the page to change."

        if action == "pause_and_memorize_fact":
            fact = str(args.get("fact", ""))
            return [], f"I memorized the following fact: {fact}"

        if action == "read_page_answer_question":
            question = str(args.get("question", ""))
            return [], (
                f"I could not read the page text to answer '{question}' -- reading the page as "
                f"text is unavailable here. Answer from the screenshot instead."
            )

        if action == "ask_user_question":
            question = str(args.get("question", ""))
            if self._auto_user_reply:
                self._pending_user_response = AUTO_USER_REPLY
            return [], f"I asked the user: {question}"

        if action == "terminate":
            answer = str(args.get("answer", ""))
            return [{"name": "done", "args": {"response": answer}}], answer

        return [], f"Unknown action: {action!r}"
