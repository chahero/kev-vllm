# Upstream attribution

This integration depends on the following independent projects and model artifacts:

- [Kev](https://github.com/jaredpalmer/kev), by Jared Palmer and contributors. Setup downloads encoder and checkpoint-loading source at commit `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63` into the ignored artifacts directory.
- [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b), revision `139fdd94f1b6a6ad80cc15e08fcb99cac885a101`: adapter and pointer head.
- [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base), revision `1001bb4d826a52d1f399e183466143f4da7b741b`: base model.
- [vLLM](https://github.com/vllm-project/vllm), version 0.30.0: inference engine and model/pooling interfaces.
- [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [PyTorch](https://github.com/pytorch/pytorch): preparation and reference inference dependencies.

Each upstream component retains its own license and model terms. The repository does not include upstream source copies or model weights. No license for this project's original code has been selected yet.
