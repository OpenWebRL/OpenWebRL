"""
Base module for gym environment adapters.

This module provides the abstract interface and common utilities for integrating
various gym-like environments with slime's training pipeline.
"""

from .types import (
    InteractionResult,
    Status,
    ToolCall,
    ParsedToolResult,
    EnvConfig,
)
from .adapter import BaseGymEnvAdapter
from .registry import EnvRegistry, DEFAULT_REGISTRY
from .utils import (
    ToolParser,
    TokenHandler,
    TOOL_INSTRUCTION,
)

__all__ = [
    # Types
    "InteractionResult",
    "Status",
    "ToolCall",
    "ParsedToolResult",
    "EnvConfig",
    # Adapter
    "BaseGymEnvAdapter",
    # Registry
    "EnvRegistry",
    "DEFAULT_REGISTRY",
    # Utils
    "ToolParser",
    "TokenHandler",
    "TOOL_INSTRUCTION",
]
