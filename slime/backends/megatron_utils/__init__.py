import logging
import os
from functools import wraps

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import get_rope_index, split_deepstack_embs
    from megatron.core import InferenceParams, mpu, tensor_parallel

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    def _normalize_qwen3_vl_vision_output(vision_output):
        if vision_output is None:
            return None, None

        if hasattr(vision_output, "pooler_output"):
            return vision_output.pooler_output, getattr(vision_output, "deepstack_features", None)

        if isinstance(vision_output, tuple):
            if len(vision_output) == 2:
                return vision_output
            if len(vision_output) >= 3:
                return vision_output[1], vision_output[2]

        raise TypeError(f"Unsupported Qwen3-VL vision output type: {type(vision_output)!r}")

    def patch_qwen3_vl_model_forward(cls):
        original_forward = cls.forward
        vision_debug_enabled = os.environ.get("SLIME_DEBUG_QWEN3_VL_VISION_FORWARD", "0") == "1"

        def _get_dist_rank():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank()
            return "uninit"

        @wraps(original_forward)
        def _patched_forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            labels: torch.Tensor = None,
            loss_mask: torch.Tensor = None,
            inference_params: InferenceParams = None,
            packed_seq_params=None,
            extra_block_kwargs: dict = None,
            pixel_values: torch.Tensor = None,
            pixel_values_videos: torch.Tensor = None,
            image_grid_thw: torch.Tensor = None,
            video_grid_thw: torch.Tensor = None,
            image_input_mask: torch.Tensor = None,
        ) -> torch.Tensor:
            assert pixel_values_videos is None and video_grid_thw is None, "not support video now"
            assert inference_params is None, "not support inference"

            video_start_index = 0
            vision_grid_thw = None
            vision_data = None
            image_mask = None
            deepstack_feature_lists = None
            position_ids = None

            if self.pre_process:
                if image_grid_thw is not None:
                    image_mask = image_input_mask
                    if image_mask is None:
                        image_mask = (input_ids == self.image_token_id).contiguous()
                    vision_grid_thw = image_grid_thw
                    vision_data = pixel_values
                    video_start_index = image_mask.sum().item()
                    assert video_start_index > 0

                vision_embeds = None
                if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                    if vision_debug_enabled:
                        logging.getLogger(__name__).info(
                            "Qwen3VL vision forward start rank=%s image_grid_thw_shape=%s pixel_values_shape=%s",
                            _get_dist_rank(),
                            tuple(vision_grid_thw.shape),
                            tuple(vision_data.shape) if vision_data is not None else None,
                        )
                    vision_output = self.vision_model(
                        hidden_states=vision_data,
                        grid_thw=vision_grid_thw,
                        return_dict=True,
                    )
                    vision_embeds, deepstack_feature_lists = _normalize_qwen3_vl_vision_output(vision_output)
                    if vision_debug_enabled:
                        logging.getLogger(__name__).info(
                            "Qwen3VL vision forward end rank=%s vision_embeds_shape=%s deepstack_features=%s",
                            _get_dist_rank(),
                            tuple(vision_embeds.shape) if vision_embeds is not None else None,
                            None if deepstack_feature_lists is None else len(deepstack_feature_lists),
                        )

                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,
                ).clone()

                if vision_embeds is not None:
                    if video_start_index == 0:
                        image_embeds = None
                        video_embeds = vision_embeds
                    elif video_start_index == vision_embeds.shape[0]:
                        image_embeds = vision_embeds
                        video_embeds = None
                    elif 0 < video_start_index < vision_embeds.shape[0]:
                        image_embeds = vision_embeds[:video_start_index]
                        video_embeds = vision_embeds[video_start_index:]
                    else:
                        raise ValueError(
                            f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
                            f"{video_start_index}"
                        )
                    assert video_embeds is None, "not support video now"

                    if image_embeds is not None:
                        combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                        combined_embeddings[image_mask] = image_embeds
                        combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if self.config.sequence_parallel:
                    combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                    combined_embeddings = combined_embeddings.contiguous()
            else:
                combined_embeddings = None

            cu_seqlens_padded = None
            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_padded = packed_seq_params.cu_seqlens_q
            if position_ids is None:
                input_ids_for_rope_index = input_ids
                if cu_seqlens_padded is not None:

                    def thd_to_bshd(packed_values: torch.Tensor, cu_seqlens: torch.Tensor):
                        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                        max_seq_len = seqlens.max()
                        bs = len(cu_seqlens) - 1
                        results = packed_values.new_zeros(size=(bs, max_seq_len, *packed_values.shape[2:]))
                        for i, seqlen in enumerate(seqlens):
                            results[i, :seqlen] = packed_values[0, cu_seqlens[i] : cu_seqlens[i] + seqlen]
                        return results

                    def bshd_to_thd(unpacked_values: torch.Tensor, cu_seqlens: torch.Tensor):
                        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                        total_len = cu_seqlens[-1]
                        results = unpacked_values.new_zeros(size=(1, total_len, *unpacked_values.shape[2:]))
                        for i, seqlen in enumerate(seqlens):
                            results[0, cu_seqlens[i] : cu_seqlens[i] + seqlen] = unpacked_values[i, :seqlen]
                        return results

                    input_ids_for_rope_index = thd_to_bshd(input_ids, cu_seqlens_padded)

                position_ids, _ = get_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    input_ids_for_rope_index,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )
                if cu_seqlens_padded is not None:
                    position_ids = bshd_to_thd(position_ids.permute(1, 2, 0), cu_seqlens_padded).permute(2, 0, 1)

            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_feature_lists
            if self.config.sequence_parallel:
                visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    tp_size=mpu.get_tensor_model_parallel_world_size(),
                    tp_rank=mpu.get_tensor_model_parallel_rank(),
                )

            if vision_debug_enabled:
                logging.getLogger(__name__).info(
                    "Qwen3VL language forward start rank=%s decoder_input_shape=%s visual_pos_masks=%s",
                    _get_dist_rank(),
                    tuple(combined_embeddings.shape) if combined_embeddings is not None else None,
                    None if visual_pos_masks is None else tuple(visual_pos_masks.shape),
                )

            return self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=combined_embeddings,
                labels=labels,
                loss_mask=loss_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **(extra_block_kwargs or {}),
            )

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
    patch_qwen3_vl_model_forward(Qwen3VLModel)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)
