"""Shared local runtime settings; Linux keeps the original BF16 profile."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = sys.platform == "win32"
AUDIT = ROOT / "artifacts" / ("windows-vllm-audit" if WINDOWS else "vllm-audit")
DTYPE = "float16" if WINDOWS else "bfloat16"
GPU_MEMORY = float(os.environ.get("KEV_GPU_MEMORY_UTILIZATION", "0.75" if WINDOWS else "0.18"))

def engine_options():
    options = dict(model=str(AUDIT / "merged-text"), runner="pooling",
                   dtype=DTYPE, enforce_eager=True, max_model_len=2048,
                   max_num_seqs=2, max_num_batched_tokens=2048,
                   gpu_memory_utilization=GPU_MEMORY,
                   enable_prefix_caching=False, enable_chunked_prefill=False,
                   pooler_config={"task": "token_embed", "use_activation": False},
                   attention_config={"backend": "TRITON_ATTN"})
    if WINDOWS:
        options.update(block_size=32, kv_cache_dtype="auto")
    return options

def server_command():
    import json
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
               str(AUDIT / "merged-text"), "--served-model-name", "kev-4b-experimental",
               "--host", "127.0.0.1", "--port", "18089", "--runner", "pooling",
               "--dtype", DTYPE, "--enforce-eager", "--max-model-len", "2048",
               "--max-num-seqs", "2", "--max-num-batched-tokens", "2048",
               "--gpu-memory-utilization", str(GPU_MEMORY),
               "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
               "--pooler-config", json.dumps({"task": "token_embed", "use_activation": False}),
               "--attention-config", json.dumps({"backend": "TRITON_ATTN"})]
    if WINDOWS:
        command += ["--block-size", "32"]
    return command

def verify_export():
    """Reject an incomplete export or a stale export from different inputs."""
    import json
    inputs = json.loads((AUDIT / "inputs.json").read_text(encoding="utf-8"))
    out = AUDIT / "merged-text"
    exported = json.loads((out / "export-manifest.json").read_text(encoding="utf-8"))
    adapter_key = (AUDIT / "kev-checkpoint/adapter_model.safetensors").relative_to(ROOT).as_posix()
    if (exported["base_revision"] != inputs["base_revision"]
            or exported["adapter_sha256"] != inputs["sha256"][adapter_key]
            or exported["dtype"] != DTYPE):
        raise ValueError("Export does not match pinned inputs or the runtime precision")
    for name in ("config.json", "model.safetensors", "head.pt", "tokenizer.json", "tokenizer_config.json"):
        if not (out / name).is_file():
            raise FileNotFoundError(out / name)
