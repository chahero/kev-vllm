# Kev Playground

[English README](../README.md) · [한국어 README](../README.ko.md)

## 실행 / Launch

먼저 운영체제에 맞는 setup 스크립트로 `.venv`와 모델을 준비하세요.
Windows에서는 프로젝트 루트의 `run_playground.bat`를 더블클릭합니다.
모델이 준비되면 실행 버튼이 활성화됩니다. 최초 커널 컴파일은 몇 분 걸릴 수 있습니다.

Prepare the environment and model with the setup script first. On Windows:

```powershell
.\run_playground.bat
```

Linux:

```bash
.venv/bin/python scripts/playground.py --start-model --open
```

- UI: `http://127.0.0.1:18090`
- Model: `http://127.0.0.1:18089`
- Interactive API docs: `http://127.0.0.1:18090/docs`
- Model startup log: `artifacts/<platform-profile>/playground-model.log`

No Node runtime or frontend build is required. FastAPI and Uvicorn are supplied
by the existing vLLM environment. Both services bind to loopback.
Ctrl+C stops the UI and any model process it started. A pre-existing model
process is reused, including while it is loading, and is not stopped by the UI.
Keep the launch console open while using the page.

To connect to an existing model without launching one:

```powershell
.\.venv\Scripts\python.exe scripts/playground.py --open
```

Use `--port 18091` to change the UI port and
`--backend-url http://127.0.0.1:18089` to select a local model port.
The two ports must differ. The UI is intended for local development, not public hosting.

## 사용 흐름 / Workflow

1. 예제를 선택합니다 / Select an example.
2. 상황·질문·선택지를 수정합니다 / Edit the state, question, and options.
3. 실행하고 선택지별 확률을 비교합니다 / Run and compare option probabilities.
4. `JSON 보기`에서 실제 응답을 확인하고 복사합니다 / Expand JSON to inspect or copy the response.

Inputs are cleared of stale results whenever edited. The language switch changes
UI labels; it does not translate the model input. The five included inputs are
English examples tested on the Windows FP16 profile.

Provide 2–12 distinct, nonempty options. The state has a 1,024-token limit and
the complete encoded row has a 2,048-token limit. These are token limits, not
character limits. Oversized input returns a validation error without truncation.
One decision is processed at a time; concurrent calls receive HTTP 429.

Probabilities compare the supplied options. They are not calibrated accuracy
estimates. Include an explicit `Not specified` option when missing evidence
should be a valid answer. The UI does not provide chain-of-thought explanations.

## API

`POST /api/decide` accepts raw text on the **playground server (18090)**.
This convenience API encodes the input, calls the vLLM `/pooling` endpoint on
18089, and maps the returned probability matrix back to options. It is not a
native vLLM endpoint. See [examples](../examples/README.md) for runnable clients.

```json
{
  "state": "The package arrived broken. The customer requests a refund.",
  "question": "What does the customer want?",
  "options": ["A refund", "Tracking information", "A new password"]
}
```

The response includes `selected`, zero-based `selected_index`, `probabilities`,
encoded `tokens`, `inference_ms`, and `elapsed_ms`. Timings include local HTTP
overhead; the first request also loads the encoder. They are not throughput benchmarks.

`GET /api/health` reports readiness; `GET /api/examples` returns the example data.
Common errors: 422 invalid or too-long input, 429 busy, 502 invalid backend
response, 503 model unavailable. When startup fails, inspect the model log.
An offline status requires starting the model, then waiting until it is ready.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe examples/run_examples.py
```

The first command tests API contracts with a mocked backend. The second requires
a running playground and checks all five examples against the real model.
Recorded Windows results: [playground-validation.json](playground-validation.json).
Screenshots in the READMEs show actual local inference, not simulated responses.
The new launcher was exercised on Windows; its Linux path has not been rerun.
