"""Local web playground for Kev. No Node runtime or cloud API is required."""
import argparse
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from kev_runtime import AUDIT, ROOT, server_command, verify_export
from query_kev_vllm import encode_request

MODEL = "kev-4b-experimental"
ASSETS = ROOT / "playground"


class DecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: StrictStr = Field(min_length=1, max_length=16000)
    question: StrictStr = Field(min_length=1, max_length=4000)
    options: list[StrictStr] = Field(min_length=2, max_length=12)

    @field_validator("state", "question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value.strip()

    @field_validator("options")
    @classmethod
    def valid_options(cls, values):
        values = [v.strip() for v in values]
        if any(not v or len(v) > 1000 for v in values):
            raise ValueError("Options must contain 1–1000 characters")
        if len(set(values)) != len(values):
            raise ValueError("Options must be distinct")
        return values


def create_app(backend_url="http://127.0.0.1:18089", start_model=False, open_url=None):
    backend_url = backend_url.rstrip("/")
    parsed = urllib.parse.urlparse(backend_url)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1") or parsed.username or parsed.path:
        raise ValueError("Backend must be a local HTTP address without a path")
    lock = threading.Lock()

    def backend_json(path, data=None, timeout=3):
        request = urllib.request.Request(backend_url + path, data=None if data is None else json.dumps(data).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    @asynccontextmanager
    async def lifespan(app):
        process = None
        log = None
        timer = None
        try:
            if start_model:
                # A listening server may still be loading; never start a duplicate.
                try:
                    with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=1):
                        listening = True
                except OSError:
                    listening = False
                if not listening:
                    verify_export()
                    AUDIT.mkdir(parents=True, exist_ok=True)
                    log = (AUDIT / "playground-model.log").open("w", encoding="utf-8")
                    command = server_command() + ["--port", str(parsed.port or 80)]
                    env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", OMP_NUM_THREADS="4")
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            app.state.model_process = process
            if open_url:
                timer = threading.Timer(1.5, lambda: webbrowser.open(open_url))
                timer.start()
            yield
        finally:
            if timer:
                timer.cancel()
            if process and process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if log:
                log.close()

    app = FastAPI(title="Kev Playground", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin requests are not accepted"}, status_code=403)
            # Bound the body before JSON parsing, including chunked requests.
            body = await request.body()
            if len(body) > 65536:
                return JSONResponse({"detail": "Request is too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ASSETS / "index.html")

    @app.get("/api/examples")
    def examples():
        return json.loads((ROOT / "examples/decisions.json").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health():
        process = getattr(app.state, "model_process", None)
        if process and process.poll() is not None:
            return {"ready": False, "status": "failed", "detail": "Model startup failed. See artifacts/*/playground-model.log."}
        try:
            models = backend_json("/v1/models")
            ready = any(m.get("id") == MODEL for m in models.get("data", []))
            return {"ready": ready, "status": "ready" if ready else "wrong_model", "model": MODEL}
        except (OSError, ValueError):
            return {"ready": False, "status": "starting" if process else "offline", "model": MODEL}

    @app.post("/api/decide")
    def decide(item: DecisionInput):
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "Another request is running. Please try again shortly.")
        try:
            started = time.perf_counter()
            try:
                payload = encode_request(item.state, item.question, item.options)
            except FileNotFoundError as error:
                raise HTTPException(503, "Prepare the local model first using the setup script.") from error
            except ValueError as error:
                raise HTTPException(422, str(error)[:500]) from error
            except Exception as error:
                if type(error).__name__ == "ContextOverflow":
                    raise HTTPException(422, "Input is too long: state limit 1024 tokens, full row limit 2048 tokens.") from error
                raise
            inference_started = time.perf_counter()
            try:
                output = backend_json("/pooling", payload, timeout=120)
            except urllib.error.HTTPError as error:
                raise HTTPException(502, f"Model server returned HTTP {error.code}. Check its log.") from error
            except (OSError, ValueError) as error:
                raise HTTPException(503, "Model server is unavailable or timed out. Wait for startup, then retry.") from error
            inference_ms = (time.perf_counter() - inference_started) * 1000
            try:
                rows = output["data"][0]["data"]
                probabilities = [float(row[0]) for row in rows]
                if len(probabilities) != len(item.options) or any(len(row) != 1 for row in rows):
                    raise ValueError("Invalid shape")
                if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-4:
                    raise ValueError("Invalid probabilities")
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise HTTPException(502, "Model returned an invalid probability matrix.") from error
            selected = max(range(len(probabilities)), key=probabilities.__getitem__)
            return {"model": MODEL, "selected": item.options[selected], "selected_index": selected,
                    "probabilities": [{"option": option, "probability": p} for option, p in zip(item.options, probabilities, strict=True)],
                    "tokens": len(payload["input"][0]), "inference_ms": round(inference_ms, 1),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
        finally:
            lock.release()

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--backend-url", default="http://127.0.0.1:18089")
    parser.add_argument("--start-model", action="store_true", help="Start a model server if none is listening; stop only the server started here on exit")
    parser.add_argument("--open", action="store_true", help="Open the UI in your browser")
    args = parser.parse_args()
    if args.port == (urllib.parse.urlparse(args.backend_url).port or 80):
        parser.error("UI and model server must use different ports")
    import uvicorn
    url = f"http://127.0.0.1:{args.port}"
    print(f"Kev Playground: {url}", flush=True)
    uvicorn.run(create_app(args.backend_url, args.start_model, url if args.open else None), host="127.0.0.1", port=args.port)
