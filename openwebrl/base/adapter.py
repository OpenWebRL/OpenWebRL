"""
Abstract base class for gym environment adapters.

This module defines the standard interface that all gym-like environment
adapters must implement to integrate with slime's training pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from .types import (
    InteractionResult,
    Status,
    EnvConfig,
    ParsedToolResult,
)
from .utils import TokenHandler, ToolParser, tools_to_openai_format, calls_to_action


class RolloutArgs(Protocol):
    """Protocol for rollout arguments from slime."""
    sglang_router_ip: str
    sglang_router_port: int


class BaseGymEnvAdapter(ABC):
    """
    Abstract base class for gym environment adapters.
    
    This class defines the standard interface for integrating gym-like
    environments with slime's training pipeline. Concrete implementations
    should handle environment-specific setup, action conversion, and
    observation processing.
    
    Subclasses must implement:
    - create_env(): Create and return the gym environment
    - get_initial_messages(): Build initial conversation messages
    - parse_env_info(): Extract tools and policy from env info
    - process_observation(): Convert env observation to message
    
    The main interaction loop (_run_interaction) is provided by the base class
    and handles:
    - Multi-turn conversation with LLM
    - Tool call parsing and action execution
    - Token tracking and loss mask computation
    """
    
    def __init__(self, config: EnvConfig, rollout_args: RolloutArgs):
        """
        Initialize the adapter.
        
        Args:
            config: Environment configuration
            rollout_args: Rollout arguments from slime
        """
        self.config = config
        self.rollout_args = rollout_args
        self._token_handler: TokenHandler | None = None
        self._tool_parser: ToolParser | None = None
        self._tools_info: list[dict] = []
    
    @property
    def token_handler(self) -> TokenHandler:
        """Get or create token handler (lazy initialization)."""
        if self._token_handler is None:
            from slime.rollout.sglang_rollout import GenerateState
            state = GenerateState(self.rollout_args)
            self._token_handler = TokenHandler(state.tokenizer)
        return self._token_handler
    
    @property
    def tool_parser(self) -> ToolParser:
        """Get or create tool parser (lazy initialization)."""
        if self._tool_parser is None:
            self._tool_parser = ToolParser(self._tools_info)
        return self._tool_parser
    
    @abstractmethod
    def create_env(self, task_id: str) -> Any:
        """
        Create and return the gym environment.
        
        Args:
            task_id: Task identifier for the environment
            
        Returns:
            Gymnasium-compatible environment instance
        """
        pass
    
    @abstractmethod
    def get_initial_messages(
        self, 
        policy: str, 
        initial_observation: str | None
    ) -> list[dict[str, Any]]:
        """
        Build initial conversation messages.
        
        Args:
            policy: Domain policy/system prompt
            initial_observation: Initial observation from environment
            
        Returns:
            List of initial messages for the conversation
        """
        pass
    
    @abstractmethod
    def parse_env_info(self, info: dict[str, Any]) -> tuple[list[dict], str]:
        """
        Extract tools and policy from environment info.
        
        Args:
            info: Info dict from env.reset()
            
        Returns:
            Tuple of (tools_info in OpenAI format, policy string)
        """
        pass
    
    @abstractmethod
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
    
    def action_from_parsed(
        self, 
        parsed_result: ParsedToolResult
    ) -> str:
        """
        Convert parsed tool result to action for environment.
        
        Default implementation converts to functional format.
        Override for environment-specific action formats.
        
        Args:
            parsed_result: Parsed tool call result
            
        Returns:
            Action string for environment.step()
        """
        return calls_to_action(parsed_result.calls, parsed_result.normal_text)
    
    def preprocess_response(self, response: str) -> str:
        """
        Preprocess LLM response before parsing.
        
        Override for environment-specific preprocessing.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Preprocessed response
        """
        # Remove common end tokens
        response = response.rstrip()
        if response.endswith("<|im_end|>"):
            response = response[:-10]
        return response
    
    def get_tasks(self) -> list[str]:
        """
        Get list of task identifiers for training.
        
        Override to return task IDs specific to your environment.
        
        Returns:
            List of task identifier strings
        """
        raise NotImplementedError("Subclass should implement get_tasks()")
    
    async def run_interaction(
        self,
        task_id: str,
        sampling_params: dict[str, Any],
    ) -> InteractionResult:
        """
        Execute a complete agent-environment interaction.
        
        This method handles the full interaction loop:
        1. Create and reset environment
        2. Build initial messages
        3. Loop: LLM call -> parse -> action -> env step
        4. Track tokens and compute loss masks
        5. Return complete trajectory
        
        Args:
            task_id: Task identifier
            sampling_params: LLM sampling parameters
            
        Returns:
            InteractionResult with complete trajectory
        """
        from slime.utils.http_utils import post
        
        # Create environment
        env = self.create_env(task_id)
        
        # Reset and get initial info
        observation, info = env.reset()
        
        # Extract tools and policy
        tools_info, policy = self.parse_env_info(info)
        self._tools_info = tools_info
        self._tool_parser = ToolParser(tools_info)
        
        # Build initial messages
        messages = self.get_initial_messages(policy, observation)
        
        # Get prompt tokens
        prompt_text, prompt_token_ids = self.token_handler.get_prompt_tokens(
            messages, tools_info
        )
        
        # Initialize tracking
        response_token_ids: list[int] = []
        loss_masks: list[int] = []
        total_reward = 0.0
        env_info = info
        result = InteractionResult(prompt=prompt_text)
        
        # Turn boundary tracking for split_into_turns()
        turn_boundaries: list[tuple[int, int, int]] = []
        
        # Build URL for LLM server
        url = (
            f"http://{self.rollout_args.sglang_router_ip}:"
            f"{self.rollout_args.sglang_router_port}/generate"
        )
        
        # Main interaction loop
        terminated = False
        truncated = False
        
        for step in range(self.config.max_steps):
            # Track turn boundary start
            turn_start_idx = len(loss_masks)
            
            # Prepare LLM input
            text_input = self.token_handler.apply_chat_template(
                messages, tools_info, add_generation_prompt=True
            )
            payload = {"text": text_input, "sampling_params": sampling_params}
            
            # Call LLM
            output = await post(url, payload)
            
            # Check for abort
            if output.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
                result.status = Status.ABORTED
                break
            
            # Preprocess and parse response
            response = self.preprocess_response(output["text"])
            parsed = self.tool_parser.parse(response)
            
            if not parsed.success:
                result.status = Status.ABORTED
                break
            
            # Add assistant message
            messages.append({"role": "assistant", "content": response})
            token_ids, mask = self.token_handler.get_token_delta(messages, tools_info)
            response_token_ids.extend(token_ids)
            loss_masks.extend(mask)
            
            # Record turn boundary (step, response_start, response_end)
            turn_end_idx = len(loss_masks)
            turn_boundaries.append((step, turn_start_idx, turn_end_idx))
            
            # Convert to action and execute
            action = self.action_from_parsed(parsed)
            
            try:
                observation, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                result.status = Status.ABORTED
                result.info["error"] = str(e)
                break
            
            # Process observation and add to messages
            obs_message = self.process_observation(observation, parsed, messages)            
            if obs_message:
                messages.append(obs_message)
                token_ids, mask = self.token_handler.get_token_delta(messages, tools_info)
                response_token_ids.extend(token_ids)
                loss_masks.extend(mask)
            
            # Update tracking
            total_reward = reward
            env_info = info
            
            # Check termination
            if terminated or truncated:
                result.status = Status.COMPLETED if terminated else Status.TRUNCATED
                break
        
        # Handle max steps reached
        if not terminated and not truncated and result.status == Status.PENDING:
            result.status = Status.TRUNCATED
        
        # Cleanup
        env.close()
        
        # Build final result
        result.reward = total_reward
        result.info = env_info if isinstance(env_info, dict) else {}
        result.messages = messages
        result.loss_mask = loss_masks
        result.tokens = prompt_token_ids + response_token_ids
        result.response = "".join([
            msg.get("content", "") for msg in messages if msg.get("role") == "assistant"
        ])
        result.response_length = len(loss_masks)
        result.turn_boundaries = turn_boundaries
        
        return result
