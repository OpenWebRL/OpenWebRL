"""Run the output layer only where the loss reads it.

The policy loss uses logits at each sample's *response* positions and nothing else
(see :func:`slime.backends.megatron_utils.loss.get_responses`). For turn-level browser
RL a response is ~69 tokens of a ~1,950-token sample, so the output layer was projecting
~28x more positions into the vocabulary than anyone looks at. With a 248,320-token
vocabulary that is roughly 2 GB of fp32 logits per sample, and it is what caps the
micro-batch size -- which in turn leaves the GPU starved on 8k-token forwards.

Enabled with ``SLIME_RESPONSE_ONLY_LOGITS=1``. When off, nothing here runs.

The patch replaces the bound ``forward`` of the model's output layer. With sequence
parallelism the hidden states arrive sharded along the sequence, so they are gathered
first and the inner layer's own ``sequence_parallel`` is turned off for the call --
the same dance Megatron's ``materialize_only_last_token_logits`` inference path does.
"""

import logging
import os

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

logger = logging.getLogger(__name__)

__all__ = ["enabled", "install", "set_selection", "clear_selection", "selection_offsets"]

# Flat positions into the [batch, seq] view of one micro-batch, and where each sample's
# run starts inside the compacted output.
_selection: torch.Tensor | None = None
_offsets: list[int] | None = None


def enabled() -> bool:
    return os.environ.get("SLIME_RESPONSE_ONLY_LOGITS", "0") == "1"


def set_selection(total_lengths, response_lengths, max_seq_lens, device) -> None:
    """Select the positions ``get_responses`` will read, in its own index convention.

    ``get_responses`` takes ``logits[start - 1 : end - 1]`` with
    ``end = max_seq_len * i + total_length`` and ``start = end - response_length``.
    """
    global _selection, _offsets
    if not enabled() or max_seq_lens is None:
        _selection, _offsets = None, None
        return

    positions: list[int] = []
    offsets: list[int] = []
    for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
        offsets.append(len(positions))
        base = max_seq_lens[i] * i + total_length - response_length - 1
        assert base >= 0, f"sample {i}: total={total_length} response={response_length}"
        positions.extend(range(base, base + response_length))

    _selection = torch.tensor(positions, dtype=torch.long, device=device)
    _offsets = offsets


def clear_selection() -> None:
    global _selection, _offsets
    _selection, _offsets = None, None


def selection_offsets() -> list[int] | None:
    return _offsets


def _find_output_layer(model_chunk):
    """Find the output layer anywhere under DDP / Float16Module / VLM wrappers.

    A plain GPTModel owns ``output_layer`` directly; megatron-bridge's VL models hold the
    GPTModel as ``language_model``, so walk every submodule rather than a fixed path.
    """
    for name, module in model_chunk.named_modules():
        layer = getattr(module, "output_layer", None)
        if isinstance(layer, torch.nn.Module):
            return name or "<root>", layer
    return None, None


def install(model_chunk) -> None:
    """Patch one model chunk's output layer, if it has one (last pipeline stage)."""
    if not enabled():
        return
    owner, layer = _find_output_layer(model_chunk)
    if layer is None:
        logger.warning("response-only logits requested but no output_layer found on %s", type(model_chunk).__name__)
        return
    if getattr(layer, "_slime_response_only", False):
        return

    inner_forward = layer.forward

    def forward(hidden_states, weight=None, runtime_gather_output=None, **kwargs):
        selection = _selection
        if selection is None:
            return inner_forward(
                hidden_states, weight=weight, runtime_gather_output=runtime_gather_output, **kwargs
            )

        was_sequence_parallel = layer.sequence_parallel
        if was_sequence_parallel:
            # [s/tp, b, h] -> [s, b, h]. With sequence_parallel switched off below, the inner
            # layer wraps its input in copy_to_tensor_model_parallel_region, whose backward
            # already all-reduces the input grad across TP. The gather's backward must then
            # only scatter it (tensor_parallel_output_grad=False); the default reduce-scatter
            # would sum it a second time and double every decoder gradient.
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=mpu.get_tensor_model_parallel_group(),
            )

        seq_len, batch_size, hidden = hidden_states.shape
        # [s, b, h] -> [b, s, h] -> [b*s, h] so a flat index is max_seq_len * i + position.
        flat = hidden_states.transpose(0, 1).reshape(batch_size * seq_len, hidden)
        selected = flat.index_select(0, selection).unsqueeze(1)  # [n_selected, 1, h]

        if was_sequence_parallel:
            layer.sequence_parallel = False
        try:
            return inner_forward(
                selected, weight=weight, runtime_gather_output=runtime_gather_output, **kwargs
            )
        finally:
            if was_sequence_parallel:
                layer.sequence_parallel = True

    layer.forward = forward
    layer._slime_response_only = True
    logger.warning("response-only logits enabled on %s.output_layer (%s)", owner, type(layer).__name__)
