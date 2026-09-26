# kev-vllm

[English](README.md) | **한국어**

[Kev](https://github.com/jaredpalmer/kev) 의사결정 모델을 위한 비공식 실험용 vLLM 플러그인입니다. [chahero](https://github.com/chahero)가 관리합니다.

상황, 질문, 선택지를 입력하면 각 선택지의 확률을 반환합니다. Qwen3.5 백본과 Kev의 pointer head가 vLLM GPU 워커 안에서 실행되며, 기본 `/pooling` API는 `[선택지 수, 1]` 형태의 확률 행렬을 반환합니다. 채팅 응답을 생성하는 모델이 아니라 주어진 선택지 중 하나를 고르는 모델입니다.

## 검증한 실행 환경

| 환경 | GPU | Python | vLLM | 정밀도 | 모델·결과 저장 경로 |
| --- | --- | --- | --- | --- | --- |
| Windows 네이티브 | TITAN RTX 24 GiB | 3.13 | 커뮤니티 Windows 빌드 0.29.0 | FP16 | `artifacts/windows-vllm-audit` |
| Linux aarch64 | NVIDIA GB10 | 3.12 | 0.30.0 | BF16 | `artifacts/vllm-audit` |

두 환경 모두 가상환경 이름은 **`.venv`**로 통일했습니다. PyTorch 2.13.0 / CUDA 13.0, Transformers 5.17.0, PEFT 0.21.0을 사용합니다. 가상환경은 운영체제별로 새로 만드세요. 다른 컴퓨터에서 `.venv`를 복사해서 사용하지 않습니다.

## Windows에서 시작하기

WSL이나 Docker는 필요하지 않습니다. `uv`와 호환되는 NVIDIA 드라이버를 준비한 뒤, 프로젝트 루트에서 PowerShell로 최초 설치와 모델 준비를 실행합니다.

```powershell
.\scripts\setup_kev_vllm.ps1
```

준비가 끝나면 프로젝트 루트의 **`run_windows.bat`를 더블클릭**하거나 다음 명령을 실행합니다.

```powershell
.\run_windows.bat
```

BAT 파일은 자신의 위치를 기준으로 `.venv`와 준비된 모델을 확인하고, UTF-8 모드로 서버를 시작합니다. 어느 폴더에서 호출해도 사용할 수 있으며, 패키지 설치나 모델 다운로드는 하지 않습니다. 실행 오류가 나면 콘솔에 메시지를 남깁니다. 서버를 종료하려면 Ctrl+C를 누르세요.

다른 PowerShell 창에서 프로젝트 루트로 이동한 뒤 질문을 보냅니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/query_kev_vllm.py --state "The package arrived broken. The customer requests a refund." --question "What does the customer want?" --options "A refund" "Tracking information" "A new password"
```

이 예제의 선택 결과는 `A refund`입니다.

Windows 설치, pip를 통한 설치 방법, 메모리 설정 및 문제 해결은 [Windows 상세 안내](docs/windows.md)를 참고하세요. 해당 상세 문서는 영문입니다. 사용하는 Windows wheel의 URL과 SHA256은 `requirements-windows.txt`에 고정되어 있습니다. 공식 PyPI에서 단순히 `pip install vllm`을 실행하는 것만으로는 이 검증 환경을 설치할 수 없습니다.

## Linux에서 시작하기

`uv`와 호환되는 NVIDIA 드라이버를 준비한 뒤 실행합니다.

```bash
git clone https://github.com/chahero/kev-vllm.git
cd kev-vllm
bash scripts/setup_kev_vllm.sh
bash scripts/serve_kev_vllm.sh
```

다른 터미널에서 질문을 보냅니다.

```bash
.venv/bin/python scripts/query_kev_vllm.py \
  --state "The package arrived broken. The customer requests a refund." \
  --question "What does the customer want?" \
  --options "A refund" "Tracking information" "A new password"
```

두 환경 모두 기본 서버 주소는 `http://127.0.0.1:18089`입니다. 설치 과정은 고정된 리비전의 원본 파일을 다운로드하고 약 7.83 GiB의 병합 가중치를 생성합니다. 입력 모델, 가상환경, 캐시를 포함해 수십 GB의 디스크 여유 공간을 확보하세요. 모델 준비에는 GPU를 사용하며, Windows 최초 실행은 커널 컴파일 때문에 몇 분 걸릴 수 있습니다.

## 테스트

HTTP 테스트는 자체 서버를 시작하므로 실행 중인 기존 서버를 먼저 종료하세요.

Windows에서 27개 질문 비교와 HTTP·CLI 테스트를 실행합니다.

```powershell
.\scripts\test_kev_vllm.ps1 -Expanded
```

Linux에서는 다음 명령을 실행합니다.

```bash
.venv/bin/python scripts/validate_kev_vllm.py
.venv/bin/python scripts/test_kev_vllm_http.py
```

| 검증 환경 | 질문 수 | 원본과 선택 결과 일치 | 최대 확률 차이 | HTTP·CLI |
| --- | --- | --- | --- | --- |
| Windows FP16 | 27 | 배치·역순 배치·단일 실행 모두 27/27 | 0.0008902252 | 통과 |
| Linux BF16 | 27 | 배치·역순 배치·단일 실행 모두 27/27 | 0.0080635548 | 통과 |

각 환경은 동일한 정밀도의 원본 Kev 결과와 비교했습니다. 입력 길이는 20~2,048토큰이며 확률 차이 허용치는 0.02입니다. 이는 구현 간 호환성 검증으로, 일반적인 정확도·확률 보정·성능 벤치마크가 아닙니다. Windows 결과는 로컬 TITAN RTX에서, Linux 결과는 기존 GB10 작업 환경에서 얻었습니다. Windows 지원 및 문서 변경 이후 Linux 테스트를 다시 실행하지는 않았습니다.

## 문서 안내

| 문서 | 내용 |
| --- | --- |
| [Windows 상세 안내](docs/windows.md) | 설치, 실행 파일, pip 설치, 메모리 설정, 문제 해결 및 로컬 검증 결과 |
| [기존 Linux 실험 기록](docs/experiment.md) | 설계, 고정 입력, GB10 실험 결과 및 재현 방법 |
| [Windows 수치 결과](docs/windows-validation-result.json) | 27개 질문의 FP16 비교 결과 |
| [Linux 수치 결과](docs/validation-result.json) | 27개 질문의 BF16 비교 결과 |
| [외부 구성요소 출처](THIRD_PARTY.md) | 원본 구성요소와 라이선스 |

## 지원 범위와 제한

- 텍스트만 지원합니다. 질문별 전체 입력은 최대 2,048토큰, 상황 입력은 최대 1,024토큰입니다.
- GPU 1개, eager 실행, 동시 처리 2개를 사용합니다. Prefix caching과 chunked prefill은 비활성화합니다.
- 신뢰할 수 있는 로컬 입력을 대상으로 한 실험입니다. 잘못된 원시 토큰 입력은 워커 오류를 일으킬 수 있으므로, 운영 서비스에는 요청 검증이 필요합니다.
- `/v1/chat/completions`, 원본 `choice/noul/score` 계약, 이미지, 양자화, SGLang은 지원하지 않습니다.

Kev 소스는 `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63` 커밋에 고정되어 있으며, 모델 리비전은 `scripts/fetch_kev_inputs.py`에 정의되어 있습니다. 원본 소스, 모델 가중치, 가상환경과 캐시는 로컬에서 다운로드하거나 생성하며 Git에 포함하지 않습니다. 이 프로젝트는 Kev 또는 vLLM의 공식 통합 구현이 아닙니다.
