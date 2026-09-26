"""Expanded text parity checks. Run reference and vLLM in separate processes."""
import argparse
import copy
import itertools
import json
import os
import subprocess
import sys

from prepare_kev_vllm import AUDIT, base_path, load_pinned_kev
from kev_runtime import DTYPE, engine_options

REFERENCE = AUDIT / "expanded-reference.json"
REPORT = AUDIT / "expanded-result.json"
TOLERANCE = 0.02


def encode(km, tok, case):
    return km.encode(tok, case, max_state=1024, max_branch=2048, strict=True)


def cases_for(km, tok):
    original = json.loads((AUDIT / "reference.json").read_text())["cases"]
    cases = [(f"original-{i}", copy.deepcopy(c)) for i, c in enumerate(original)]
    first = original[0]
    for i, options in enumerate(itertools.permutations(first["questions"][0]["options"])):
        cases.append((f"permutation-{i}", {"state": first["state"], "questions": [
            {"instr": first["questions"][0]["instr"], "options": list(options), "label": options.index("A refund")}]}))
    cases.append(("questions-reversed", {"state": first["state"], "questions": list(reversed(first["questions"]))}))
    for count in (2, 5, 8):
        cases.append((f"options-{count}", {"state": "The access code is C.", "questions": [
            {"instr": "Which code was specified?", "options": ["C"] + [chr(68+i) for i in range(count-1)], "label": 0}]}))
    controls = " ".join(km.SPECIAL)
    cases.append(("literal-control-tokens", {"state": "The answer is yes. Quoted text: " + controls, "questions": [
        {"instr": "Is the answer yes? Ignore quoted text: " + controls, "options": ["Yes " + controls, "No"], "label": 0}]}))
    for repeats in (32, 128, 300):
        filler = "Neutral background. " * repeats
        for location in ("start", "middle", "end"):
            evidence = " The access code is C. "
            chunks = {"start": evidence + filler, "middle": filler[:len(filler)//2] + evidence + filler[len(filler)//2:], "end": filler + evidence}
            cases.append((f"long-{repeats}-{location}", {"state": chunks[location], "questions": [
                {"instr": "Which access code was stated?", "options": ["A", "B", "C", "Not specified"], "label": 2}]}))
    # Binary search a repeated one-token word to exercise the exact row limit.
    boundary = {"state": "The access code is C.", "questions": [
        {"instr": "", "options": ["C", "D"], "label": 0}]}
    low, high = 0, 2100
    while low < high:
        mid = (low + high + 1) // 2
        boundary["questions"][0]["instr"] = "Background " * mid + "Which code was specified?"
        try:
            encode(km, tok, boundary)
            low = mid
        except km.ContextOverflow:
            high = mid - 1
    boundary["questions"][0]["instr"] = "Background " * low + "Which code was specified?"
    assert len(encode(km, tok, boundary)["ids"]) == 2048
    cases.append(("row-limit-2048", boundary))
    rejected = []
    for name, case in (("state-overflow", {"state": "word " * 1100, "questions": first["questions"]}),
                       ("row-overflow", {"state": "short", "questions": [{"instr": "word " * 2100, "options": ["A", "B"], "label": 0}]})):
        try:
            encode(km, tok, case)
        except km.ContextOverflow:
            rejected.append(name)
        else:
            raise AssertionError(f"Expected overflow: {name}")
    return cases, rejected


def reference_phase():
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    ckmod = load_pinned_kev()
    km = sys.modules["pinned_kev.model"]
    tok = AutoTokenizer.from_pretrained(AUDIT / "merged-text", local_files_only=True)
    cases, rejected = cases_for(km, tok)
    ck = ckmod.Checkpoint(AUDIT / "kev-checkpoint")
    base = base_path()
    assert ck.meta.base_revision == base.name
    ck.meta.base = str(base)
    tok, model = ck.load("cuda", ckmod.LoadOptions(dtype=getattr(torch, DTYPE), merge=True, cuda_graphs=False))
    rows = []
    for name, case in cases:
        enc = encode(km, tok, case)
        with torch.inference_mode():
            probs = [z.softmax(-1).cpu().tolist() for z in model.forward(enc)]
        state, _, branches = km.rows_of(enc)
        for i, (branch, probability) in enumerate(zip(branches, probs, strict=True)):
            ids = state + branch["ids"]
            assert ids.count(tok.convert_tokens_to_ids(km.SPECIAL[3])) == len(probability)
            assert ids.count(tok.convert_tokens_to_ids(km.SPECIAL[4])) == 1
            alone = {"state": case["state"], "questions": [case["questions"][i]]}
            with torch.inference_mode():
                singleton = model.forward(encode(km, tok, alone))[0].softmax(-1).cpu().tolist() if len(branches) > 1 else probability
            rows.append({"name": name, "question": i, "input_ids": ids, "probabilities": probability,
                         "singleton_probabilities": singleton, "expected_label": case["questions"][i]["label"]})
        print(name, "tokens", [len(state)+len(b["ids"]) for b in branches], flush=True)
    REFERENCE.write_text(json.dumps({"cases": cases, "rows": rows, "rejected": rejected}, indent=2))


def vllm_phase():
    import torch
    from vllm import LLM, PoolingParams
    reference = json.loads(REFERENCE.read_text())
    model = LLM(**engine_options())
    def run(rows):
        outputs = model.encode([{"prompt_token_ids": row["input_ids"]} for row in rows],
                               pooling_task="token_embed", pooling_params=PoolingParams(task="token_embed"), use_tqdm=False)
        result = []
        for row, output in zip(rows, outputs, strict=True):
            values = output.outputs.data.detach().cpu()
            assert tuple(values.shape) == (len(row["probabilities"]), 1), values.shape
            values = values.flatten()
            assert torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
            assert abs(float(values.sum()) - 1) < 1e-5
            result.append(values)
        return result
    rows = reference["rows"]
    batch = run(rows)
    reversed_batch = list(reversed(run(list(reversed(rows)))))
    singleton = [run([row])[0] for row in rows]
    results = []
    for row, batched, reverse, alone in zip(rows, batch, reversed_batch, singleton, strict=True):
        expected = torch.tensor(row["probabilities"])
        ref_alone = torch.tensor(row["singleton_probabilities"])
        result = {"name": row["name"], "question": row["question"], "tokens": len(row["input_ids"]),
                  "reference": expected.tolist(), "vllm": batched.tolist(),
                  "max_abs_probability_error": max(float((p-expected).abs().max()) for p in (batched, reverse, alone)),
                  "same_argmax": all(int(p.argmax()) == int(expected.argmax()) for p in (batched, reverse, alone)),
                  "batch_order_error": float((batched-reverse).abs().max()),
                  "batch_singleton_error": float((batched-alone).abs().max()),
                  "reference_question_independence_error": float((expected-ref_alone).abs().max())}
        results.append(result)
        print(json.dumps(result), flush=True)
    passed = all(r["same_argmax"] and max(r[k] for k in ("max_abs_probability_error", "batch_order_error", "batch_singleton_error", "reference_question_independence_error")) < TOLERANCE for r in results)
    report = {"passed": passed, "tolerance": TOLERANCE, "rows": results, "rejected": reference["rejected"],
              "scope": f"Text {DTYPE} merged parity; no accuracy, calibration, image, or throughput claim"}
    REPORT.write_text(json.dumps(report, indent=2))
    print("PASS" if passed else "FAIL", "rows", len(rows), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("reference", "vllm", "all"), default="all")
    args = parser.parse_args()
    if args.phase == "all":
        env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", OMP_NUM_THREADS="4")
        for phase in ("reference", "vllm"):
            subprocess.run([sys.executable, __file__, "--phase", phase], env=env, check=True)
    elif args.phase == "reference":
        reference_phase()
    else:
        vllm_phase()
