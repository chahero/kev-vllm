# Decision examples / 선택 예제

Start `run_playground.bat`, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe examples/run_examples.py
```

Linux: `.venv/bin/python examples/run_examples.py`.
No extra client library is needed. Use `--url` to select a different playground port.

| Example | 질문의 목적 | Expected choice | Recorded probability |
| --- | --- | --- | --- |
| Customer intent | 고객 요청 분류 | A refund | 99.66% |
| Route a request | 담당 팀 선택 | Billing | 96.83% |
| Read a policy | 문서 내용 확인 | No | 99.67% |
| Prioritize an issue | 규칙에 따른 우선순위 | P1: All customers blocked | 99.76% |
| Missing information | 근거 없는 답변 피하기 | Not specified | 93.45% |

Inputs and expected zero-based indices are in [decisions.json](decisions.json).
These five examples all matched on native Windows / TITAN RTX / FP16 on
2026-09-26. This is a smoke test, not a general accuracy benchmark.
Different environments or inputs can change the probabilities.

## One request / 단일 요청

Save this as `request.json` (UTF-8):

```json
{
  "state": "The package arrived broken. The customer requests a refund.",
  "question": "What does the customer want?",
  "options": ["A refund", "Tracking information", "A new password"]
}
```

Windows PowerShell:

```powershell
curl.exe http://127.0.0.1:18090/api/decide -H "Content-Type: application/json" --data-binary "@request.json"
```

Linux/macOS shell: use `curl` instead of `curl.exe`.
The endpoint belongs to the playground, which must be running. For direct
vLLM access on port 18089, use `scripts/query_kev_vllm.py` as shown in the README.

See [full recorded responses](../docs/playground-validation.json) and the
[API guide](../docs/playground.md). Responses contain real model scores;
expected labels are only checked by this example script and never sent to the model.
