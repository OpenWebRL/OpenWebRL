"""
Environment adapter registry.

This module provides a centralized registry for managing gym environment
adapters, enabling pluggable environment support.
"""

from typing import Any, Type

from .types import EnvConfig
from .adapter import BaseGymEnvAdapter


class EnvRegistry:
    """
    Registry for gym environment adapters.
    
    Provides a centralized way to register and retrieve environment adapters
    by name, enabling easy extension with new environments.
    
    Usage:
        # Create registry
        registry = EnvRegistry()
        
        # Register an adapter
        registry.register(
            name="my_env",
            adapter_class=MyAdapter,
            config_class=MyConfig,
            default_config={"max_steps": 30},
        )
        
        # Create adapter instance
        adapter = registry.create_adapter("my_env", rollout_args)
        
        # List available environments
        print(registry.list_environments())
    """
    
    def __init__(self):
        self._adapters: dict[str, Type[BaseGymEnvAdapter]] = {}
        self._configs: dict[str, Type[EnvConfig]] = {}
        self._default_config: dict[str, dict[str, Any]] = {}
    
    def register(
        self, 
        name: str, 
        adapter_class: Type[BaseGymEnvAdapter],
        config_class: Type[EnvConfig] | None = None,
        default_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Register an environment adapter.
        
        Args:
            name: Unique identifier for the environment
            adapter_class: Adapter class implementing BaseGymEnvAdapter
            config_class: Configuration class for this adapter
            default_config: Default configuration values
        """
        self._adapters[name] = adapter_class
        if config_class:
            self._configs[name] = config_class
        if default_config:
            self._default_config[name] = default_config
    
    def unregister(self, name: str) -> None:
        """
        Unregister an environment adapter.
        
        Args:
            name: Environment name to unregister
        """
        self._adapters.pop(name, None)
        self._configs.pop(name, None)
        self._default_config.pop(name, None)
    
    def get_adapter_class(self, name: str) -> Type[BaseGymEnvAdapter]:
        """Get adapter class by name."""
        if name not in self._adapters:
            available = ", ".join(self._adapters.keys()) or "(none)"
            raise ValueError(f"Unknown environment '{name}'. Available: {available}")
        return self._adapters[name]
    
    def get_config_class(self, name: str) -> Type[EnvConfig] | None:
        """Get config class by name."""
        return self._configs.get(name)
    
    def get_default_config(self, name: str) -> dict[str, Any]:
        """Get default config values by name."""
        return self._default_config.get(name, {})
    
    def create_adapter(
        self, 
        name: str, 
        rollout_args: Any,
        config_overrides: dict[str, Any] | None = None,
    ) -> BaseGymEnvAdapter:
        """
        Create an adapter instance.
        
        Args:
            name: Environment name
            rollout_args: Rollout arguments from slime
            config_overrides: Override default config values
            
        Returns:
            Configured adapter instance
        """
        adapter_class = self.get_adapter_class(name)
        config_class = self.get_config_class(name)
        
        # Build config
        config_values = self.get_default_config(name).copy()
        if config_overrides:
            config_values.update(config_overrides)
        
        if config_class:
            config = config_class(**config_values)
        else:
            config = EnvConfig(**config_values)
        
        return adapter_class(config, rollout_args)
    
    def list_environments(self) -> list[str]:
        """List all registered environment names."""
        return list(self._adapters.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if an environment is registered."""
        return name in self._adapters


# Global default registry instance
DEFAULT_REGISTRY = EnvRegistry()
