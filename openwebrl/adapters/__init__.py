"""
Adapters for various gym-like environments.

This module contains concrete implementations of BaseGymEnvAdapter
for different environments.

Built-in Adapters:
    - BrowserAdapter: Browser multi-website task-oriented dialogue

Adding New Adapters:
    1. Create your adapter in this package (e.g., my_env_adapter.py)
    2. Import and export from this __init__.py
    3. Register with ENV_REGISTRY in generate_new.py
"""

from .browser_adapter import BrowserAdapter, BrowserEnvConfig

__all__ = [
    "BrowserAdapter",
    "BrowserEnvConfig",
]
