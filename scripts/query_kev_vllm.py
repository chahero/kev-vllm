"""Encode a text question with pinned Kev code and call stock vLLM /pooling."""
import argparse
import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/vllm-audit"


def encode_request(state, question, options):
    from transformers import AutoTokenizer
    spec = importlib.util.spec_from_file_location("kev_encoder", AUDIT / "kev-source/model.py")
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    tokenizer = AutoTokenizer.from_pretrained(AUDIT / "merged-text", local_files_only=True)
    record = {"state": state, "questions": [{"instr": question, "options": options, "label": 0}]}
    # label is unused by inference; the upstream encoder requires the field.
    encoded = encoder.encode(tokenizer, record, max_state=1024, max_branch=2048, strict=True)
    return {"model": "kev-4b-experimental", "task": "token_embed",
            "input": [encoded["ids"]], "encoding_format": "float", "use_activation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18089")
    parser.add_argument("--dry-run", action="store_true", help="Print encoded request without contacting the server")
    args = parser.parse_args()
    if len(args.options) < 2:
        parser.error("provide at least two options")
    body = encode_request(args.state, args.question, args.options)
    if args.dry_run:
        print(json.dumps(body))
        return
    request = urllib.request.Request(args.url.rstrip("/") + "/pooling", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            output = json.load(response)
    except urllib.error.HTTPError as error:
        parser.exit(1, f"HTTP {error.code}: {error.read().decode(errors='replace')}\n")
    except urllib.error.URLError as error:
        parser.exit(1, f"Cannot reach vLLM at {args.url}: {error.reason}\n")
    probabilities = [entry[0] for entry in output["data"][0]["data"]]
    if len(probabilities) != len(args.options):
        raise ValueError("Server returned a different number of options")
    selected = max(range(len(probabilities)), key=probabilities.__getitem__)
    print(json.dumps({"selected": args.options[selected], "selected_index": selected,
                      "probabilities": [{"option": o, "probability": p} for o,p in zip(args.options, probabilities)]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
