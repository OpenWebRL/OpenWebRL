"""OpenWebRL browser-agent training and evaluation package."""

import os
import sys

# Core types from base module
from openwebrl.base.types import (
    Status,
    InteractionResult,
    EnvConfig,
    ToolCall,
    ParsedToolResult,
)

# Base adapter class for extensions
from openwebrl.base.adapter import BaseGymEnvAdapter

# Registry
from openwebrl.base.registry import DEFAULT_REGISTRY as ENV_REGISTRY
from openwebrl.base.registry import EnvRegistry

# Utilities
from openwebrl.base.utils import (
    ToolParser,
    TokenHandler,
    tools_to_openai_format,
    calls_to_action,
    TOOL_INSTRUCTION,
)

# New extensible interface (uses registry from base)
# from openwebrl.generate_new import (
#     ENV_REGISTRY,
#     set_active_env,
#     get_task_prompts as get_task_prompts_new,
#     get_task_ids as get_task_ids_new,
# )

# Legacy interface (for backward compatibility)
from openwebrl.generate_browser import (
    generate_trajectory_sample,
    generate_turn_sample,
    # get_task_prompts,
    # get_task_ids,
    # res_to_sample,
    # TAU2_CONFIGS,
)

__all__ = [
    # Core types
    "Status",
    "InteractionResult",
    "EnvConfig",
    "ToolCall",
    "ParsedToolResult",
    # Base class for extensions
    "BaseGymEnvAdapter",
    # Registry
    "EnvRegistry",
    # Utilities
    "ToolParser",
    "TokenHandler",
    "tools_to_openai_format",
    "calls_to_action",
    "TOOL_INSTRUCTION",
    # New interface
    "ENV_REGISTRY",
    # "set_active_env",
    # "get_task_prompts_new",
    # "get_task_ids_new",
    # Legacy interface
    # "generate",
    "generate_trajectory_sample",
    "generate_turn_sample"
    # "get_task_prompts",
    # "get_task_ids",
    # "res_to_sample",
]
