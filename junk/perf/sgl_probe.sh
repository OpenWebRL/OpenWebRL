#!/usr/bin/env bash
cd /weka/oe-training-default/new_peters/OpenWebRL && python -c "import sglang, transformers; print('sglang', sglang.__version__, 'transformers', transformers.__version__)" && python junk/perf/sgl_logprob_probe.py
