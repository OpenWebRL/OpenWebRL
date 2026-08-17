# Browser Judge Utilities

This folder contains optional scripts for evaluating judge models or API judge
endpoints on prepared browser-judge JSON/JSONL files.

The judge-SFT training workflow is not part of this release. The remaining
scripts are standalone evaluation helpers:

- `run_eval_openai_judge_models.sh`: evaluate OpenAI-compatible or Azure judge endpoints.
- `run_eval_instruct_models.sh`: evaluate Hugging Face instruct models.
- `run_eval_judge_sft_ckpts.sh`: evaluate local converted judge checkpoints.
- `run_eval_webjudge_7b.sh`: evaluate WebJudge-7B.
- `single_file_judge_inference.py`: local model inference over JSON/JSONL judge inputs.
- `single_file_openai_judge_inference.py`: API judge inference over JSON/JSONL judge inputs.

Set `VAL_DATA` to your judge-evaluation JSONL before running the shell scripts.

## Updates

- 2026.08.17: Fixed native judge verdict parsing in
  `single_file_judge_inference.py` to prioritize the final explicit verdict
  marker (`Verdict`, `Final Verdict`, or `Definitive Verdict`) instead of the
  first verdict-like phrase in the reasoning text.
