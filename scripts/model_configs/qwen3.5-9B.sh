# Qwen3.5-9B (Qwen3_5ForConditionalGeneration): dense 9B early-fusion VLM, 32 layers
# of which 24 are gated-delta linear attention and 8 full gated attention
# (full_attention_interval 4). Values read from the HF config's text_config.
#
# With --megatron-to-hf-mode bridge these args do NOT build the model: megatron-bridge
# constructs the full vision+language provider from the HF config. They still have to
# match it, because slime uses them for vocab padding, parallelism and the optimizer.
# Do not switch this model to direct mode -- megatron_to_hf/qwen3_5.py has no vision
# mappings, so the vision tower would silently drop out of training and weight sync.
MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 4
   --kv-channels 256
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 12288
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)
