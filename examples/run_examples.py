"""Run the bundled decisions against a running local playground."""
import argparse
import json
from pathlib import Path
import urllib.request

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18090")
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("decisions.json").read_text(encoding="utf-8"))
    passed = 0
    for case in cases:
        body = {key: case[key] for key in ("state", "question", "options")}
        request = urllib.request.Request(args.url.rstrip("/") + "/api/decide",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        ok = result["selected_index"] == case["expected_index"]
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {case['id']}: {result['selected']}")
    print(f"{passed}/{len(cases)} expected choices matched")
    raise SystemExit(0 if passed == len(cases) else 1)

if __name__ == "__main__":
    main()
