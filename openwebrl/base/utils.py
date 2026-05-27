"""
Utility functions for gym environment adapters.

This module provides shared utilities for:
- Tool/function call parsing
- Token handling and loss mask computation
- Message formatting
"""

import json
import re
from typing import Any

from transformers import AutoTokenizer

from .types import ToolCall, ParsedToolResult


# Tool instruction template for agentic environments
TOOL_INSTRUCTION = (
    "At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n\n"
    "**Tool Call Format**: Use `point_2d` as an array [x, y] for coordinates.\n"
    "Example: `{\"name\": \"click\", \"arguments\": {\"point_2d\": [379, 69]}}`\n"
)


class ToolParser:
    """
    Parser for extracting tool calls from LLM responses.
    
    Supports multiple formats:
    - Native tool calling (Qwen3, etc.) via sglang's FunctionCallParser
    - XML-style tool calls: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    - JSON tool calls
    """
    
    def __init__(self, tools_info: list[dict] | None = None, parser_type: str = "auto"):
        """
        Initialize the tool parser.
        
        Args:
            tools_info: List of tool definitions in OpenAI format
            parser_type: Parser type - "auto", "qwen25", "regex", "json"
        """
        self.tools_info = tools_info or []
        self.parser_type = parser_type
        self._sglang_parser = None
    
    def _get_sglang_parser(self):
        """Lazy-load sglang's FunctionCallParser."""
        if self._sglang_parser is None and self.tools_info:
            try:
                from sglang.srt.function_call.function_call_parser import FunctionCallParser
                from sglang.srt.managers.io_struct import Function, Tool
                
                tools_list = []
                for tool in self.tools_info:
                    if tool.get("type") == "function" and "function" in tool:
                        func = tool["function"]
                        tools_list.append(
                            Tool(
                                function=Function(
                                    name=func.get("name", ""),
                                    description=func.get("description", ""),
                                    parameters=func.get("parameters", {}),
                                ),
                                type=tool["type"],
                            )
                        )
                if tools_list:
                    self._sglang_parser = FunctionCallParser(
                        tools=tools_list, tool_call_parser="qwen25"
                    )
            except ImportError:
                pass
        return self._sglang_parser
    
    def parse(self, response: str) -> ParsedToolResult:
        """
        Parse tool calls from LLM response.
        
        Args:
            response: Raw response text from LLM
            
        Returns:
            ParsedToolResult with parsed calls and normal text
        """
        if self.parser_type == "regex" or (self.parser_type == "auto" and not self.tools_info):
            return self._parse_regex(response)
        
        # Try sglang parser first for native tool calling
        try:
            sglang_parser = self._get_sglang_parser()
            if sglang_parser:
                normal_text, calls = sglang_parser.parse_non_stream(response)
                parsed_calls = []
                if calls:
                    for call in calls:
                        call_dict = call.model_dump()
                        parsed_calls.append(ToolCall(
                            name=call_dict.get("name", ""),
                            parameters=call_dict.get("parameters", {}),
                        ))
                
                # Clean up thinking tokens if present
                if normal_text:
                    # normal_text = normal_text.split('</think>')[-1]
                    normal_text = normal_text
                
                return ParsedToolResult(
                    success=True,
                    normal_text=normal_text or "",
                    calls=parsed_calls,
                )
        except Exception:
            pass
        
        # Fallback to regex parsing
        return self._parse_regex(response)
    
    def _parse_regex(self, response: str) -> ParsedToolResult:
        """Regex-based fallback parser for tool calls."""
        # Try XML-style tool calls
        tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(tool_call_pattern, response, re.DOTALL)
        
        calls = []
        if matches:
            try:
                for match in matches:
                    tool_data = json.loads(match)
                    if "name" in tool_data:
                        calls.append(ToolCall(
                            name=tool_data["name"],
                            parameters=tool_data.get("arguments", tool_data.get("parameters", {})),
                        ))
                
                # Extract text before tool call
                text_before = re.split(tool_call_pattern, response)[0].strip()
                return ParsedToolResult(
                    success=True,
                    normal_text=text_before,
                    calls=calls,
                )
            except json.JSONDecodeError as e:
                return ParsedToolResult(
                    success=False,
                    normal_text=response,
                    calls=[],
                    error=f"JSON decode error: {e}",
                )
        
        # No tool call found, treat entire response as normal text
        return ParsedToolResult(
            success=True,
            normal_text=response.strip(),
            calls=[],
        )


class TokenHandler:
    """
    Handler for tokenization and loss mask computation.
    
    Handles multi-turn conversation tokenization and computes loss masks
    for training (masking environment/tool responses, keeping assistant responses).
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        """
        Initialize token handler.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
    
    def reformulate_tool_instruction(self, text: str) -> str:
        """
        Reformulate tool call instruction for environments that need it.
        
        Args:
            text: Original text with tool instructions
            
        Returns:
            Text with reformulated tool instructions
        """
        return text.replace(
            "You may call one or more functions to assist with the user query.",
            TOOL_INSTRUCTION
        )
    
    def apply_chat_template(
        self,
        messages: list[dict],
        tools_info: list[dict] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        reformulate_tool_instr: bool = False,
        enable_thinking: bool | None = None,
    ) -> str:
        """
        Apply chat template to messages.
        
        Args:
            messages: List of conversation messages
            tools_info: Tool definitions for the template
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to return tokens instead of text
            reformulate_tool_instr: Whether to replace the tokenizer's
                default tool instruction with TOOL_INSTRUCTION.
            enable_thinking: Optional Qwen-style thinking switch forwarded
                to tokenizer.apply_chat_template when supported.
            
        Returns:
            Formatted text (or tokens if tokenize=True)
        """
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
            "tools": tools_info,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            text = self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )
        return self.reformulate_tool_instruction(text) if (isinstance(text, str) and reformulate_tool_instr) else text
    
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def get_token_delta(
        self,
        messages: list[dict],
        tools_info: list[dict] | None = None,
    ) -> tuple[list[int], list[int]]:
        """
        Calculate token delta for multi-turn conversations.
        
        Computes the tokens and loss mask for the last message in the conversation,
        masking appropriately based on whether it's an assistant response or
        environment/tool response.
        
        Args:
            messages: Full conversation messages
            tools_info: Tool definitions
            
        Returns:
            Tuple of (token_ids, loss_mask)
        """
        curr = self.apply_chat_template(
            messages, tools_info, add_generation_prompt=False
        )
        
        token_ids = []
        loss_mask = []
        
        # Case 1: Last message is an assistant response (should be trained on)
        if messages[-1]["role"] == "assistant":
            prev = self.apply_chat_template(
                messages[:-1], tools_info, add_generation_prompt=True
            )
            new_tokens = self.encode(curr[len(prev):])
            token_ids = new_tokens
            loss_mask = [1] * len(new_tokens)
        else:
            # Case 2: Last message is tool response or user message (mask out)
            prev = self.apply_chat_template(
                messages[:-1], tools_info, add_generation_prompt=False
            )
            new_tokens = self.encode(curr[len(prev):])
            token_ids = new_tokens
            loss_mask = [0] * len(new_tokens)
            # import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()
        
        return token_ids, loss_mask
    
    def get_prompt_tokens(
        self,
        messages: list[dict],
        tools_info: list[dict] | None = None,
    ) -> tuple[str, list[int]]:
        """
        Get prompt text and token IDs.
        
        Args:
            messages: Conversation messages
            tools_info: Tool definitions
            
        Returns:
            Tuple of (prompt_text, token_ids)
        """
        prompt_text = self.apply_chat_template(messages, tools_info, add_generation_prompt=True)
        token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, token_ids


def tools_to_openai_format(tools: list) -> list[dict]:
    """
    Convert environment tool objects to OpenAI-compatible format.
    
    Args:
        tools: List of tool objects (tau2-bench, etc.)
        
    Returns:
        List of tool definitions in OpenAI format
    """
    tools_info = []
    for tool in tools:
        # Try openai_schema property (tau2-bench tools)
        if hasattr(tool, 'openai_schema'):
            tools_info.append(tool.openai_schema)
        elif hasattr(tool, 'to_openai_schema'):
            tools_info.append(tool.to_openai_schema())
        else:
            # Fallback for basic tools
            tool_info = {
                "type": "function",
                "function": {
                    "name": getattr(tool, 'name', str(tool)),
                    "description": (
                        tool._get_description() 
                        if hasattr(tool, '_get_description') 
                        else str(tool)
                    ),
                    "parameters": (
                        tool.params.model_json_schema() 
                        if hasattr(tool, 'params') 
                        else {}
                    ),
                }
            }
            tools_info.append(tool_info)
    return tools_info


def calls_to_action(calls: list[ToolCall], text_response: str) -> str:
    """
    Convert parsed tool calls to action string.
    
    Args:
        calls: List of parsed tool calls
        text_response: Text content of the response
        
    Returns:
        Action string in functional format or plain text
    """
    if calls:
        # Use first tool call
        tool_call = calls[0]
        params = (
            json.loads(tool_call.parameters) 
            if isinstance(tool_call.parameters, str) 
            else tool_call.parameters
        )
        
        # Build functional format: tool_name(arg1='val1', arg2='val2')
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{tool_call.name}({args_str})"
    
    # No tool call, return text response
    return text_response
