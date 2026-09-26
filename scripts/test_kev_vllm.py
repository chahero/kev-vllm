"""Compare a native vLLM custom pooler against the pinned Kev BF16 reference."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/vllm-audit"


def main():
    import torch
    from vllm import LLM, PoolingParams
    reference = json.loads((AUDIT / "reference.json").read_text())
    started = time.perf_counter()
    model = LLM(model=str(AUDIT / "merged-text"), runner="pooling",
                dtype="bfloat16", enforce_eager=True,
                max_model_len=2048, max_num_seqs=2,
                max_num_batched_tokens=2048, gpu_memory_utilization=0.18,
                enable_prefix_caching=False, enable_chunked_prefill=False,
                pooler_config={"task": "token_embed", "use_activation": False},
                attention_config={"backend": "TRITON_ATTN"})
    load_seconds = time.perf_counter() - started
    prompts = [{"prompt_token_ids": r["input_ids"]} for r in reference["rows"]]
    started = time.perf_counter()
    outputs = model.encode(prompts, pooling_task="token_embed", pooling_params=PoolingParams(task="token_embed"), use_tqdm=False)
    elapsed = time.perf_counter() - started
    results = []
    for ref, output in zip(reference["rows"], outputs):
        probs = output.outputs.data.detach().cpu().flatten()
        expected = torch.tensor(ref["probabilities"])
        result = {"case": ref["case"], "question": ref["question"],
                  "reference": expected.tolist(), "vllm": probs.tolist(),
                  "max_abs_probability_error": float((probs-expected).abs().max()),
                  "same_argmax": int(probs.argmax()) == int(expected.argmax())}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"vllm": "0.30.0", "load_seconds": load_seconds,
              "inference_seconds": elapsed, "rows": results,
              "passed": all(r["same_argmax"] and r["max_abs_probability_error"] < 0.02 for r in results),
              "scope": "5 short text questions; BF16 merged baseline; no image or throughput validation"}
    (AUDIT / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
