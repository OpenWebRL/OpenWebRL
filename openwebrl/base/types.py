"""
Common data types for gym environment adapters.

This module defines standard data structures used across all gym-like
environment integrations with slime training.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(Enum):
    """Status of the agent-environment interaction."""
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class ToolCall:
    """Represents a parsed tool/function call."""
    name: str
    parameters: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parameters": self.parameters}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            name=data.get("name", ""),
            parameters=data.get("parameters", data.get("arguments", {})),
        )


@dataclass
class ParsedToolResult:
    """Result of parsing a tool call from LLM response."""
    success: bool
    normal_text: str
    calls: list[ToolCall]
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "parsed_result": {
                "normal_text": self.normal_text,
                "calls": [c.to_dict() for c in self.calls],
            },
            "error": self.error,
        }


@dataclass
class TurnResult:
    """
    Result of a single turn in an agent-environment interaction.
    
    Used for turn-level training where each turn is a separate training example.
    A turn consists of one assistant response and the corresponding environment feedback.
    """
    # Turn identification
    turn_index: int = 0
    
    # Core turn data
    prompt: str = ""  # The prompt for this turn (rendered text)
    response: str = ""  # Assistant response for this turn
    messages: list[dict[str, Any]] = field(default_factory=list)  # Messages up to (not including) this response
    
    # Turn-level reward (intermediate feedback, may be 0 for most turns)
    turn_reward: float = 0.0
    
    # Token-level data for this turn only
    tokens: list[int] = field(default_factory=list)  # Full tokens up to this turn
    loss_mask: list[int] = field(default_factory=list)  # Loss mask for this turn's response only
    response_length: int = 0  # Length of this turn's response
    
    # Additional metadata
    info: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "prompt": self.prompt,
            "response": self.response,
            "messages": self.messages,
            "turn_reward": self.turn_reward,
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "response_length": self.response_length,
            "info": self.info,
        }


@dataclass
class InteractionResult:
    """
    Result of an agent-environment interaction.
    
    This is the unified output format for all gym-like environment adapters,
    containing the complete trajectory and training-relevant data.
    """
    # Core trajectory data
    prompt: str = ""
    response: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    
    # Reward and status
    reward: float = 0.0
    status: Status = Status.PENDING
    
    # Token-level data for training
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    response_length: int = 0
    
    # Turn boundaries for splitting (populated during rollout)
    # Each entry: (turn_index, response_start_idx, response_end_idx)
    turn_boundaries: list[tuple[int, int, int]] = field(default_factory=list)
    
    # Additional metadata
    info: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "messages": self.messages,
            "reward": self.reward,
            "status": self.status.value,
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "response_length": self.response_length,
            "turn_boundaries": self.turn_boundaries,
            "info": self.info,
        }
    
    def split_into_turns(self, token_handler: Any, tools_info: list[dict]) -> list["TurnResult"]:
        """
        Split the trajectory into turn-level results.
        
        Each turn corresponds to one assistant response. The turn's prompt
        includes all messages up to (but not including) that response.
        
        Args:
            token_handler: TokenHandler for encoding prompts
            tools_info: Tool definitions for chat template
            
        Returns:
            List of TurnResult objects, one per assistant turn
        """
        turns: list[TurnResult] = []
        
        # Find assistant message indices
        assistant_indices = [
            i for i, msg in enumerate(self.messages) 
            if msg.get("role") == "assistant"
        ]
        
        if not assistant_indices:
            return turns
        
        # Use turn_boundaries if available (more accurate)
        if self.turn_boundaries:
            for turn_idx, (step, resp_start, resp_end) in enumerate(self.turn_boundaries):
                # Build prompt from messages up to this assistant response
                msg_idx = assistant_indices[turn_idx] if turn_idx < len(assistant_indices) else -1
                prompt_messages = self.messages[:msg_idx] if msg_idx >= 0 else self.messages
                
                turn_prompt = token_handler.apply_chat_template(
                    prompt_messages, tools_info, add_generation_prompt=True
                )
                
                # Get this turn's response
                response = self.messages[msg_idx].get("content", "") if msg_idx >= 0 else ""
                
                # Extract loss mask for this turn
                turn_loss_mask = self.loss_mask[resp_start:resp_end]
                
                # Tokens up to end of this turn's response
                prompt_token_len = len(self.tokens) - len(self.loss_mask)
                turn_tokens = self.tokens[:prompt_token_len + resp_end]
                
                turn = TurnResult(
                    turn_index=turn_idx,
                    prompt=turn_prompt,
                    response=response,
                    messages=[dict(m) for m in prompt_messages],  # Deep copy messages
                    turn_reward=0.0,  # Will be set by caller
                    tokens=turn_tokens,
                    loss_mask=turn_loss_mask,
                    response_length=len(turn_loss_mask),
                    info={"step": step},
                )
                turns.append(turn)
        else:
            # Fallback: reconstruct from messages (less accurate for loss_mask)
            current_mask_idx = 0
            
            for turn_idx, msg_idx in enumerate(assistant_indices):
                # Prompt is all messages before this assistant message
                prompt_messages = self.messages[:msg_idx]
                turn_prompt = token_handler.apply_chat_template(
                    prompt_messages, tools_info, add_generation_prompt=True
                )
                
                # Response is this assistant message
                response = self.messages[msg_idx].get("content", "")
                
                # Estimate tokens for this response
                # This is approximate - boundaries are more accurate
                response_tokens = token_handler.tokenizer.encode(
                    response, add_special_tokens=False
                )
                resp_len = len(response_tokens)
                
                turn_loss_mask = self.loss_mask[current_mask_idx:current_mask_idx + resp_len]
                
                prompt_token_len = len(self.tokens) - len(self.loss_mask)
                turn_tokens = self.tokens[:prompt_token_len + current_mask_idx + resp_len]
                
                turn = TurnResult(
                    turn_index=turn_idx,
                    prompt=turn_prompt,
                    response=response,
                    messages=[dict(m) for m in prompt_messages],  # Deep copy messages
                    turn_reward=0.0,
                    tokens=turn_tokens,
                    loss_mask=turn_loss_mask,
                    response_length=len(turn_loss_mask),
                    info={"step": turn_idx},
                )
                turns.append(turn)
                
                # Move past this response and any observation tokens
                # Find next assistant or end
                if turn_idx + 1 < len(assistant_indices):
                    # Estimate observation tokens between responses
                    next_msg_idx = assistant_indices[turn_idx + 1]
                    for obs_idx in range(msg_idx + 1, next_msg_idx):
                        obs_content = self.messages[obs_idx].get("content", "")
                        obs_tokens = token_handler.tokenizer.encode(
                            obs_content, add_special_tokens=False
                        )
                        current_mask_idx += len(obs_tokens)
                
                current_mask_idx += resp_len
        
        return turns


@dataclass
class EnvConfig:
    """
    Base configuration for gym environment adapters.
    
    Subclass this for environment-specific configurations.
    """
    max_steps: int = 30
    
    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
