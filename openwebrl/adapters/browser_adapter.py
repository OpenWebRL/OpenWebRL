"""
Browser environment adapter.

This module provides the adapter for integrating browser with slime's
training pipeline using the standard gym interface.
"""

import time
import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from openwebrl.base.adapter import BaseGymEnvAdapter
from openwebrl.base.types import EnvConfig, ParsedToolResult, ToolCall
from openwebrl.feedback_utils import (
    DEFAULT_BROWSER_TOOL_RESPONSE_MAX_CHARS,
    truncate_feedback_text,
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class BrowserEnvConfig(EnvConfig):
    """Configuration specific to browser environment."""
    # Environment mode: "sandbox", "local_process", or "browser-use"
    mode: str = "sandbox"
    # Sandbox-specific config (used when mode: sandbox)
    sandbox: dict | None = None
    # Local subprocess-specific config (used when mode: local_process)
    local_process: dict | None = None
    # Browser-Use-specific config (used when mode: browser-use)
    browser_use: dict | None = None

    # Web browser environment configuration
    width: int = 1280
    height: int = 720
    dpr: int = 1

    max_retries: int = 3
    wait_timeout: int = 5
    screenshot_timeout: int | None = None

    resize_output_coords: bool = True
    resize_scale: int = 1000
    image_patch_size: int = 32 # not used if resize_scale > 0

    path_to_tool_list: str | None = None
    path_to_policy: str | None = None
    path_to_task_file: str | None = None

    # Additional configuration for browser adapter
    use_screenshot: bool = True
    use_a11ytree: bool = False


class BrowserAdapter(BaseGymEnvAdapter):
    """
    Adapter for browser gymnasium environment.
    
    Browser provides multi-domain task-oriented dialogue benchmarks
    with a gymnasium-compatible interface for agent training.
    """
    
    def __init__(self, config: BrowserEnvConfig, rollout_args: Any):
        """
        Initialize the browser adapter.
        
        Args:
            config: Browser-specific configuration
            rollout_args: Rollout arguments from slime
        """
        super().__init__(config, rollout_args)
        default_max_chars = DEFAULT_BROWSER_TOOL_RESPONSE_MAX_CHARS
        self._tool_response_max_chars = int(
            getattr(rollout_args, "browser_tool_response_max_chars", default_max_chars) or default_max_chars
        )

    async def create_env(self, task_id: str) -> Any:
        """
        Create and return the gym environment.
        
        Args:
            task_id: Task identifier for the environment
            
        Returns:
            Gymnasium-compatible environment instance
        """
        pass

    def parse_env_info(self, info: dict[str, Any]) -> tuple[list[dict], str]:
        """
        Extract tools and policy from browser env info.
        
        Args:
            info: Info dict from env.reset()
            
        Returns:
            Tuple of (tools_info, policy)
        """
        tool_list = info.get("tool_list", [])
        policy = info.get("policy", "")
        return tool_list, policy

    def process_observation(
        self, 
        observation: str | None, 
        parsed_result: ParsedToolResult,
        messages: list[dict],
    ) -> dict[str, Any] | None:
        """
        Convert environment observation to a message.
        
        Args:
            observation: Raw observation from environment step
            parsed_result: Parsed tool call result from LLM response
            messages: Current conversation messages
            
        Returns:
            Message dict to append to conversation, or None if no message
        """
        pass

    def _build_observation_message(
        self,
        observation: dict[str, Any] | None,
        prefix_text: str = "",
        role: str = "user",
    ) -> dict[str, Any]:
        """Build a multimodal message that optionally prefixes an observation block."""
        message = {
            "role": role,
            "content": [{"type": "text", "text": prefix_text}],
        }

        if not observation:
            return message

        screen_size = observation.get("screen_size") or ["", ""]
        all_tabs = observation.get("all_tab_url") or []
        active_tab_index = None
        active_tab_url = observation.get("active_tab_url", "")
        for tab in all_tabs:
            if tab.get("active"):
                active_tab_index = tab.get("index")
                break

        obs_lines = [
            "<observation>",
            f"screen size: {screen_size[0]} x {screen_size[1]}",
        ]
        if active_tab_index is not None:
            obs_lines.append(f"Current tab: {active_tab_index}")
            obs_lines.append("Available tabs:")
            for tab in all_tabs:
                idx = tab.get("index", "")
                url = tab.get("url", "")
                title = tab.get("title", "")
                active_suffix = " (active)" if tab.get("active") else ""
                tail = f" - {title}" if title else ""
                obs_lines.append(f"- Tab {idx}{active_suffix}: {url}{tail}")
        else:
            obs_lines.append(f"url: {active_tab_url}")

        obs_text = "\n".join(obs_lines)

        if self.config.use_a11ytree:
            a11ytree = json.dumps(observation.get("a11ytree", None))
            obs_text += f"\na11ytree: {a11ytree}"

        if self.config.use_screenshot:
            screenshot = observation.get("screenshot", None)
            if not isinstance(screenshot, bytes):
                raise TypeError(f"Screenshot must be bytes, got {type(screenshot)}")
            screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
            obs_text += "\nscreenshot:\n<|vision_start|><|image_pad|><|vision_end|>"
            message["content"].append(
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                }
            )

        obs_text += "\n</observation>"
        message["content"][0]["text"] += obs_text
        return message

    def _convert_tool_responses_to_msgs(
        self,
        tool_responses: list[dict],
        observation: dict[str, Any] | None = None,
        include_tool_response: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Convert tool responses into the next user turn, aligned with web_browser style.
        """
        if len(tool_responses) == 0:
            raise ValueError("Tool responses is empty.")

        prefix_text = ""
        if include_tool_response:
            lines = [
                self._truncate_tool_response(tool_result.get("tool_response", ""))
                for tool_result in tool_responses
            ]
            prefix_text = "<tool_response>\n" + "\n".join(lines) + "\n"
            if observation:
                prefix_text += "\n"
        msg = self._build_observation_message(observation, prefix_text=prefix_text, role="user")
        if include_tool_response:
            msg["content"][0]["text"] += "\n</tool_response>"
        return [msg]

    def _truncate_tool_response(self, content: str) -> str:
        """Clamp verbose env feedback before it enters the next-turn prompt."""
        return truncate_feedback_text(content, max(self._tool_response_max_chars, 1))

    def _get_latest_user_message(self, user_intent: str, observation: dict[str, Any], step_id: int = None) -> dict[str, Any]:
        """
        Build latest user message for browser environment.
        
        Args:
            user_intent: Current user intent or query
            observation: Observation from environment
            action_history: Agent history
            
        Returns:
            User message
        """
        prefix_text = f"<user_request>\n{user_intent}\n</user_request>\n\n"
        return self._build_observation_message(observation, prefix_text=prefix_text, role="user")
    
    def _process_multimodal_messages(self, mm_messages: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Process multimodal messages: extract images and convert to text
        Args:
            mm_messages: a list of messages, each message has a role and content.
                role: "system", "user", "tool", or "assistant"
                content: a list of items, each item has a type and value, The type can be "text" or "image_url".
                If the a message contains "image_url", it must contain "text" as well, and the content of "text" should contain the same number of image placeholders "<|vision_start|><|image_pad|><|vision_end|>".
                See `_get_latest_user_message()` for more details.
        Returns:
            processed_message: a list of messages, each message has a role and content `string`.
            image_data_list: a list of image data. Each item is the value of "image_url" in the input messages.
        """
        # Check if messages contain images and extract them
        image_data_list = []
        processed_messages = []
        
        cnt_image_placeholder = 0
        for mm_msg in mm_messages:
            content = mm_msg.get("content", "")
            
            # Handle multimodal messages (list format with text and images)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":  
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        # Extract image URL/data
                        image_url = item.get("image_url", "")
                        image_data_list.append(image_url)
                
                # Create text-only version of this message
                combined_text = "\n".join(text_parts)

                # update the number of image placeholders
                cnt_image_placeholder += combined_text.count("<|vision_start|><|image_pad|><|vision_end|>")

                processed_messages.append({"role": mm_msg["role"], "content": combined_text})
            else:
                # Non-multimodal message (string content), pass through as-is
                processed_messages.append(mm_msg)
        
        if cnt_image_placeholder != len(image_data_list):
            raise ValueError("Number of image placeholders does not match number of images.")

        return processed_messages, image_data_list
    
    def _clean_messages(self,
        messages: list[dict],
        roles_to_check: list[str]=["user"],
        text_to_remove: str="screenshot:\n<|vision_start|><|image_pad|><|vision_end|>\n"
    ) -> list[dict]:
        """
        Remove text_to_remove from messages with the given roles.
        Args:
            messages: List of messages
            text_to_remove: Text to remove
        Returns:
            List of cleaned messages
        """
        # Remove text_to_remove from all messages
        cleaned_messages = []
        for i, msg in enumerate(messages):
            msg_copy = dict(msg)
            if (msg_copy.get("role") in roles_to_check) and msg_copy.get("content"):
                if isinstance(msg_copy["content"], str):
                    msg_copy["content"] = msg_copy["content"].replace(text_to_remove, "")
                else:
                    raise TypeError(f"Content must be string, got {type(msg_copy['content'])}")
            cleaned_messages.append(msg_copy)
        return cleaned_messages


    def get_initial_messages(self, policy: str, initial_observation: str | None) -> list[dict[str, Any]]:
        """
        Build initial conversation messages for browser environment.
        
        Args:
            policy: Domain policy from environment
            initial_observation: Initial user query
            
        Returns:
            List of system and user messages
        """
        raise NotImplementedError


    def _get_action_description(self, parsed: ParsedToolResult) -> str:
        """
        Get action description from parsed tool call.
        
        Args:
            parsed: Parsed tool result from LLM response. Properties:
                - calls: List of ToolCall objects
                - normal_text: Normal text, expected to contain <thinking>...</thinking> and <action>...</action> tags
                - success: Whether the tool call was successful
            
        Returns:
            Action description
        """
        pattern = r"<action>(.*?)</action>"
        match = re.search(pattern, parsed.normal_text, re.DOTALL)
        if match:
            action_description = match.group(1).strip()
        else:
            action_description = ""
        return action_description

    def actions_from_parsed(self, parsed_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """
        Convert parsed tool result to list of actions.
        For single action parse, please refer to self.action_from_parsed().
        Args:
            parsed_calls: List of tool calls from LLM response
            
        Returns:
            List of actions
        """
        actions = []
        for tool_call in parsed_calls:
            params = (
                json.loads(tool_call.parameters) 
                if isinstance(tool_call.parameters, str) 
                else tool_call.parameters
            )
            actions.append({
                "name": tool_call.name,
                "args": params
            })
        return actions
    
    async def _run_inference_step(
        self,
        url: str,
        tokens: list[int],
        sampling_params: dict,
        image_data,
    ):
        """
        Run a single inference step.
        """
        pass
