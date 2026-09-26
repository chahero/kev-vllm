"""Start stock vllm serve with the Kev plugin, test /pooling, then stop it."""
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from kev_runtime import ROOT, AUDIT, server_command


def main():
    base = "http://127.0.0.1:18089"
    reference = json.loads((AUDIT / "reference.json").read_text())
    cmd = server_command()
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", OMP_NUM_THREADS="4")
    with (AUDIT / "http-server.log").open("w") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"))
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"vllm serve exited {process.returncode}; see http-server.log")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("vllm serve startup exceeded 900 seconds")
            with urllib.request.urlopen(base + "/v1/models") as r:
                models = json.load(r)
            body = {"model": "kev-4b-experimental", "task": "token_embed",
                    "input": [r["input_ids"] for r in reference["rows"]],
                    "encoding_format": "float", "use_activation": False}
            request = urllib.request.Request(base + "/pooling", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=120) as r:
                response = json.load(r)
            elapsed = time.perf_counter() - started
            checks = []
            for ref, item in zip(reference["rows"], response["data"], strict=True):
                actual = [v[0] for v in item["data"]]
                assert len(actual) == len(ref["probabilities"])
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in actual)
                assert abs(sum(actual) - 1) < 1e-5
                error = max(abs(a-b) for a,b in zip(actual,ref["probabilities"]))
                same = max(range(len(actual)), key=actual.__getitem__) == max(range(len(actual)), key=ref["probabilities"].__getitem__)
                checks.append({"max_abs_probability_error": error, "same_argmax": same})
            case = reference["cases"][0]
            question = case["questions"][0]
            client = subprocess.run([sys.executable, str(ROOT / "scripts/query_kev_vllm.py"),
                                     "--url", base, "--state", case["state"], "--question", question["instr"],
                                     "--options", *question["options"]], env=env, capture_output=True, text=True, timeout=120, check=True)
            client_result = json.loads(client.stdout)
            assert client_result["selected_index"] == question["label"]
            report = {"command": cmd, "models": models, "response": response, "checks": checks, "client": client_result,
                      "request_seconds": elapsed, "passed": all(c["same_argmax"] and c["max_abs_probability_error"] < 0.02 for c in checks)}
            (AUDIT / "http-result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"passed": report["passed"], "checks": checks, "request_seconds": elapsed}), flush=True)
            if not report["passed"]:
                raise SystemExit(1)
        finally:
            if process.poll() is None:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True) if os.name == "nt" else os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill() if os.name == "nt" else os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
