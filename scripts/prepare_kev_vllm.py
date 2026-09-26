"""Build a text-only merged Kev checkpoint and reference outputs (pinned source)."""
import importlib.util
import json
import shutil
import sys
import time
import types
import hashlib
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/vllm-audit"
BASE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b"


def base_path():
    inputs = AUDIT / "inputs.json"
    return Path(json.loads(inputs.read_text())["base_path"]) if inputs.exists() else BASE


def load_pinned_kev():
    package = types.ModuleType("pinned_kev")
    package.__path__ = [str(AUDIT / "kev-source")]
    sys.modules[package.__name__] = package
    for name in ("model", "checkpoint"):
        spec = importlib.util.spec_from_file_location(f"pinned_kev.{name}", AUDIT / "kev-source" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["pinned_kev.checkpoint"]


def main():
    out = AUDIT / "merged-text"
    completed = out / "export-manifest.json"
    if completed.exists():
        raise FileExistsError("Export already complete; keep it or choose a separate experiment directory")
    torch.set_num_threads(4)
    ckmod = load_pinned_kev()
    ck = ckmod.Checkpoint(AUDIT / "kev-checkpoint")
    base = base_path()
    assert ck.meta.base_revision == base.name
    ck.meta.base = str(base)
    print("Loading pinned reference Kev (BF16, merged, no cache)", flush=True)
    tok, model = ck.load("cuda", ckmod.LoadOptions(dtype=torch.bfloat16, merge=True, cuda_graphs=False))
    cases = [
        {"state": "The package arrived broken. The customer requests a refund.", "questions": [
            {"instr": "What does the customer want?", "options": ["A refund", "Tracking information", "A new password"], "label": 0},
            {"instr": "Did the package arrive undamaged?", "options": ["Yes", "No"], "label": 1}]},
        {"state": "The package arrived broken. The customer requests a refund.", "questions": [
            {"instr": "What does the customer want?", "options": ["A new password", "Tracking information", "A refund"], "label": 2}]},
        {"state": "Office hours are Monday through Friday, 9 AM to 5 PM. The office is closed on weekends.", "questions": [
            {"instr": "Is the office open on Sunday?", "options": ["Yes", "No"], "label": 1}]},
        {"state": "The meeting is scheduled for Tuesday. Its location has not been announced.", "questions": [
            {"instr": "Where will the meeting take place?", "options": ["Room A", "Room B", "Online", "Not specified"], "label": 3}]},
    ]
    km = sys.modules["pinned_kev.model"]
    rows = []
    for case_idx, case in enumerate(cases):
        enc = km.encode(tok, case, max_state=1024, max_branch=2048, strict=True)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model.forward(enc)
            probabilities = [z.softmax(-1).cpu().tolist() for z in logits]
        torch.cuda.synchronize()
        state, _, branches = km.rows_of(enc)
        for question_idx, (branch, probs, z) in enumerate(zip(branches, probabilities, logits)):
            rows.append({"case": case_idx, "question": question_idx,
                         "input_ids": state + branch["ids"],
                         "decide": len(state) + branch["decide"],
                         "options": [len(state) + p for p in branch["opts"]],
                         "probabilities": probs, "logits": z.float().cpu().tolist(),
                         "expected_label": case["questions"][question_idx]["label"]})
        print("Reference case", case_idx, probabilities, "seconds", time.perf_counter()-started, flush=True)
    (AUDIT / "reference.json").write_text(json.dumps({"cases": cases, "rows": rows, "temperature": model.head.temperature, "dtype": "bf16", "merge": True}, indent=2))
    out.mkdir(exist_ok=True)
    config = model.lm.config.to_dict()
    config["architectures"] = ["KevQwen3_5ForPooling"]
    config["tie_word_embeddings"] = True
    config["kev_head_dim"] = ck.meta.head_dim
    config["kev_temperature"] = ck.meta.temperature
    config["kev_option_end_id"] = tok.convert_tokens_to_ids(km.SPECIAL[3])
    config["kev_decide_id"] = tok.convert_tokens_to_ids(km.SPECIAL[4])
    (out / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copyfile(AUDIT / "kev-checkpoint/head.pt", out / "head.pt")
    tok.save_pretrained(out)
    print("Saving merged text weights", flush=True)
    weights = {f"model.{name}": tensor.detach().cpu().contiguous() for name, tensor in model.lm.state_dict().items()}
    temp = out / "model.safetensors.partial"
    save_file(weights, temp, metadata={"format": "pt"})
    temp.replace(out / "model.safetensors")
    with (AUDIT / "kev-checkpoint/adapter_model.safetensors").open("rb") as f:
        adapter_hash = hashlib.file_digest(f, "sha256").hexdigest()
    completed.write_text(json.dumps({"base_revision": base.name, "adapter_sha256": adapter_hash,
                                    "dtype": "bfloat16", "merge": True, "tensor_count": len(weights)}, indent=2))
    print("Saved", out, "tensors", len(weights), flush=True)


if __name__ == "__main__":
    main()
