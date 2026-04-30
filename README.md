# lmro-2phase-inverse

LMR(Li-rich Mn-rich) 양극 / Li metal 음극 half-cell 데이터에서
**2-phase(rhombohedral R3m + monoclinic C2m) OCP 및 전송 파라미터**를 역추정하는 파이프라인입니다.

```
TOYO SYSTEM ASCII/CSV
  → PyBaMM half-cell (SPMe / DFN)
  → 2-phase tanh OCP fitting
  → synthetic dataset 생성
  → DL inverse model 학습
  → 실측 프로파일 → phase별 D, R, fraction, OCP 추정
```

---

## 목차

1. [요구사항](#1-요구사항)
2. [환경 설치](#2-환경-설치)
3. [시스템별 설정](#3-시스템별-설정)
4. [데이터 파일 적응](#4-데이터-파일-적응)
5. [파이프라인 실행 순서](#5-파이프라인-실행-순서)
6. [프로젝트 구조](#6-프로젝트-구조)
7. [설정 파일 레퍼런스](#7-설정-파일-레퍼런스)
8. [테스트 실행](#8-테스트-실행)
9. [자주 묻는 질문](#9-자주-묻는-질문)

---

## 1. 요구사항

| 항목 | 최소 버전 | 비고 |
|------|-----------|------|
| Python | 3.11 | 3.12 권장 |
| PyBaMM | 26.4 | `< 27` |
| PyTorch | 2.0 | CPU 또는 CUDA |
| OS | Linux / macOS | Windows 미검증 |

GPU는 선택 사항입니다. CPU만으로도 Stage 0~2까지 실행 가능하며,
Stage 3(DL 학습) 이후부터는 CUDA가 있으면 속도가 크게 향상됩니다.

---

## 2. 환경 설치

### 2-1. 자동 설치 (권장)

```bash
cd lmro_2phase_inverse
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh`는 다음을 순서대로 수행합니다.

1. Python 가상환경 `.venv` 생성
2. pip/wheel 업그레이드
3. PyTorch 설치 (`configs/env.yaml`의 `torch.index_url` 참조)
4. 전체 패키지 editable 설치 (`pip install -e ".[dev]"`)

### 2-2. CUDA 버전 지정 설치

CUDA를 사용하려면 `setup_env.sh` 실행 **전에** `configs/env.yaml`을 수정합니다.

```yaml
# configs/env.yaml
torch:
  index_url: "https://download.pytorch.org/whl/cu121"  # CUDA 12.1
  device: "auto"
```

| CUDA 버전 | index_url |
|-----------|-----------|
| CUDA 12.4 | `https://download.pytorch.org/whl/cu124` |
| CUDA 12.1 | `https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `https://download.pytorch.org/whl/cu118` |
| ROCm 6.0  | `https://download.pytorch.org/whl/rocm6.0` |
| CPU only  | `""` (빈 문자열) |

### 2-3. 수동 설치

가상환경 없이 직접 설치하려면:

```bash
pip install pybamm>=26.4,<27
pip install torch                         # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install -e ".[dev]"
```

### 2-4. 설치 확인

```bash
python -c "import pybamm; print('PyBaMM:', pybamm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, '/ CUDA:', torch.cuda.is_available())"
python -c "import lmro2phase; print('lmro2phase:', lmro2phase.__version__)"
```

---

## 3. 시스템별 설정

다른 시스템으로 이식할 때 **`configs/env.yaml` 하나만 수정**하면 됩니다.

```yaml
# configs/env.yaml

python:
  venv_path: ".venv"           # 가상환경 경로 (절대경로도 가능)

torch:
  index_url: ""                # CUDA 버전에 맞는 URL (위 표 참조)
  device: "auto"               # "auto" | "cpu" | "cuda" | "cuda:0"

parallel:
  n_jobs: 4                    # synthetic 생성 병렬 프로세스 수
  pybamm_nthreads: 1           # PyBaMM 내부 BLAS 스레드

paths:
  data_root: "data"            # 데이터 루트 (절대경로 지정 가능)
  raw_data:  "data/raw/toyo"
  ...

seed: 42
log_level: "INFO"
```

### 경로를 절대경로로 지정하는 경우 (NFS 마운트 등)

```yaml
paths:
  data_root: "/mnt/storage/bat_sim/data"
  raw_data:  "/mnt/storage/bat_sim/data/raw/toyo"
  synthetic: "/mnt/storage/bat_sim/data/synthetic"
```

---

## 4. 데이터 파일 적응

TOYO SYSTEM 파일 형식이 달라지면 **`configs/toyo_ascii.yaml` 만 수정**합니다.
파서는 인코딩·구분자·컬럼명을 모두 자동 감지합니다.

### 4-1. 현재 지원 파일 형식

| 항목 | 지원 목록 |
|------|-----------|
| 인코딩 | utf-8, utf-8-sig, cp932, shift_jis, euc-jp |
| 구분자 | `,` `\t` `;` whitespace |
| 컬럼명 | 영어 / 일본어 혼재 가능 |
| 단위 | A 또는 mA, Ah 또는 mAh 자동 변환 |

### 4-2. 새로운 컬럼명 추가

파일에 없던 컬럼명이 나타나면 `configs/toyo_ascii.yaml`에 alias를 추가합니다.

```yaml
column_aliases:
  voltage_v:
    - "Voltage(V)"      # 기존
    - "Vbat"            # 새로 추가
    - "Cell Voltage"    # 새로 추가
```

### 4-3. 전류 단위가 mA인 파일

```yaml
column_aliases:
  current_ma:
    - "Current(mA)"    # 이 컬럼이 감지되면 자동으로 A로 변환

unit_conversion:
  current_ma_to_a: true
```

### 4-4. 전류 부호 규약이 다른 경우

```yaml
current_sign:
  discharge_positive_in_file: true   # 방전이 양수인 파일이면 true
  pybamm_discharge_positive: false   # PyBaMM half-cell 기본값
```

두 값이 다르면 자동으로 부호를 반전합니다.

### 4-5. Mode 컬럼이 없는 파일

`mode_label_map`에 매핑이 없으면 전류/전압 패턴으로 자동 분류합니다.
임계값은 `protocol_detection` 섹션에서 조정합니다.

```yaml
protocol_detection:
  rest_current_threshold_a: 1.0e-4   # |I| < 이 값이면 rest
  cc_std_ratio_threshold: 0.05       # std(I)/|I| < 이 값이면 CC
  cv_voltage_std_threshold_v: 0.002  # std(V) < 이 값이면 CV
```

---

## 5. 파이프라인 실행 순서

모든 스크립트는 `lmro_2phase_inverse/` 디렉토리 안에서 실행합니다.

```bash
cd lmro_2phase_inverse
source .venv/bin/activate
```

### Stage 0 — PyBaMM 환경 확인

```bash
python scripts/00_smoke_test_pybamm_halfcell.py
```

확인 항목:
- PyBaMM 버전
- SPM / SPMe / DFN positive half-cell 빌드
- positive 2-phase 지원 여부 → 실패 시 FallbackA(가중 혼합 OCP) 자동 선택
- tanh OCP 주입 테스트
- 측정 전류 drive cycle 시뮬레이션
- CC-CV-rest Experiment 시뮬레이션

결과: `data/reports/smoke_test_report.json`

### Stage 1a — TOYO 파일 파싱

```bash
python scripts/01_parse_toyo_ascii.py \
  --input data/raw/toyo/Toyo_LMR_HalfCell_Sample_50cycles.csv \
  --config configs/toyo_ascii.yaml \
  --out_dir data/processed
```

결과: `data/processed/Toyo_LMR_HalfCell_Sample_50cycles_processed.parquet`

### Stage 1b — tanh OCP 초기 피팅

```bash
python scripts/02_fit_tanh_ocp.py --config configs/stage1_fit_tanh.yaml
```

결과: `data/reports/stage1_fit/`
- `best_params.json` — 스칼라 파라미터
- `best_ocp_tanh_R3m.json` / `best_ocp_tanh_C2m.json` — OCP tanh 파라미터
- `ocp_phase_plot.png` — OCP 시각화

### Stage 2 — Synthetic dataset 생성

```bash
# pilot 규모 (1,000개)
python scripts/03_generate_synthetic_dataset.py

# 대규모 (50,000개)
python scripts/03_generate_synthetic_dataset.py --n_samples 50000
```

결과: `data/synthetic/`
- `params.parquet` — 성공 케이스 파라미터
- `profiles.zarr` — 시뮬레이션 프로파일
- `ocp_profiles.zarr` — OCP grid
- `failed_cases.parquet` — 실패 케이스 로그 (절대 무시하지 않음)

### Stage 3 — DL inverse model 학습

```bash
# overfit 확인 (100개, 빠른 검증)
python scripts/04_train_inverse_model.py --overfit_test

# 본 학습
python scripts/04_train_inverse_model.py --config configs/stage3_train_inverse.yaml
```

결과: `data/models/inverse/best_model.pt`

### Stage 4a — 실측 프로파일 추론

```bash
python scripts/05_infer_lmr_profile.py --config configs/stage4_infer_validate.yaml
```

결과: `data/reports/inference/cycle_XXX/`
- `predicted_params.json`
- `predicted_ocp_R3m.csv` / `predicted_ocp_C2m.csv`

### Stage 4b — Forward validation

```bash
python scripts/06_forward_validate.py --config configs/stage4_infer_validate.yaml
```

결과:
- `forward_validation.png` — 측정값 vs 시뮬레이션 비교
- `residual_summary.json` — RMSE, MAE, 최대 오차

---

## 6. 프로젝트 구조

```
lmro_2phase_inverse/
├── setup_env.sh                    # 환경 설치 스크립트
├── pyproject.toml
│
├── configs/                        # ← 이식성 핵심: 이 폴더만 수정
│   ├── env.yaml                    #   시스템별 설정 (CUDA, 경로, 병렬화)
│   ├── toyo_ascii.yaml             #   데이터 파일 형식 적응
│   ├── halfcell_lmr_base.yaml      #   LMR 물리 파라미터 초기값
│   ├── stage0_smoke.yaml
│   ├── stage1_fit_tanh.yaml
│   ├── stage2_generate_synthetic.yaml
│   ├── stage3_train_inverse.yaml
│   └── stage4_infer_validate.yaml
│
├── data/
│   ├── raw/toyo/                   # 원본 TOYO CSV 파일
│   ├── processed/                  # 파싱 결과 parquet
│   ├── synthetic/                  # synthetic dataset
│   ├── models/                     # 학습된 모델 체크포인트
│   └── reports/                    # 각 stage 결과물 및 플롯
│
├── src/lmro2phase/
│   ├── io/
│   │   ├── toyo_ascii.py           # TOYO 파서 (자동 인코딩/구분자/컬럼 감지)
│   │   ├── profile_schema.py       # BatteryProfile / ProfileSegment 데이터 구조
│   │   ├── profile_cleaning.py     # 전처리 (단위 변환, 이상값 제거)
│   │   └── dataset_store.py        # parquet 저장/로드
│   │
│   ├── physics/
│   │   ├── pybamm_env.py           # PyBaMM / PyTorch 환경 확인
│   │   ├── halfcell_model_factory.py  # half-cell 모델 빌더 + 2-phase 지원 판별
│   │   ├── positive_2phase_factory.py # FallbackA 가중 혼합 OCP surrogate
│   │   ├── lmr_parameter_set.py    # PyBaMM parameter set 생성
│   │   ├── ocp_tanh.py             # tanh basis OCP (numpy + PyBaMM 심볼릭)
│   │   ├── ocp_grid.py             # 256점 자유형 OCP grid
│   │   ├── ocp_perturbation.py     # GP / spline / shoulder 변동 생성
│   │   ├── protocol_builder.py     # 측정 전류 interpolant + Experiment 생성
│   │   └── simulator.py            # 시뮬레이션 공통 인터페이스 (실패 로깅)
│   │
│   ├── fitting/
│   │   ├── stage1_objective.py     # V(Q)/V(t)/dVdQ/dQdV/rest 다중 손실
│   │   └── stage1_optimizer.py     # Optuna → scipy L-BFGS-B
│   │
│   ├── generation/
│   │   ├── sampler.py              # R3m/C2m 파라미터 샘플러 (local/broad/edge)
│   │   ├── batch_simulate.py       # 병렬 배치 시뮬레이션
│   │   └── quality_filter.py       # 비물리적 결과 필터
│   │
│   ├── features/
│   │   ├── capacity_axis.py        # 정규화 용량 격자 보간
│   │   ├── differential_features.py # dV/dQ, dQ/dV + [10, 512] feature tensor
│   │   └── normalization.py        # feature 정규화 통계
│   │
│   ├── learning/
│   │   ├── dataset.py              # PyTorch Dataset (zarr + parquet)
│   │   ├── model_inverse.py        # 1D CNN residual + Transformer encoder
│   │   ├── losses.py               # permutation-invariant phase OCP loss
│   │   ├── train.py                # 학습 루프
│   │   └── infer.py                # 추론 + 결과 저장
│   │
│   └── validation/
│       ├── forward_validate.py     # 예측 파라미터 → PyBaMM → 잔차
│       └── plots.py                # 시각화
│
├── scripts/
│   ├── 00_smoke_test_pybamm_halfcell.py
│   ├── 01_parse_toyo_ascii.py
│   ├── 02_fit_tanh_ocp.py
│   ├── 03_generate_synthetic_dataset.py
│   ├── 04_train_inverse_model.py
│   ├── 05_infer_lmr_profile.py
│   └── 06_forward_validate.py
│
└── tests/
    ├── test_toyo_parser.py
    ├── test_ocp_tanh.py
    ├── test_ocp_grid.py
    ├── test_profile_resampling.py
    ├── test_stage1_objective.py
    └── test_inverse_model_overfit.py
```

---

## 7. 설정 파일 레퍼런스

### configs/env.yaml — 시스템별 설정

```yaml
python:
  venv_path: ".venv"

torch:
  index_url: ""          # "" = CPU / pip 기본. CUDA URL 입력 시 GPU 사용
  device: "auto"         # "auto" | "cpu" | "cuda" | "cuda:0"

parallel:
  n_jobs: 4              # synthetic 생성 병렬 프로세스 수
  pybamm_nthreads: 1     # PyBaMM OMP 스레드

paths:
  data_root: "data"
  raw_data:  "data/raw/toyo"
  processed: "data/processed"
  synthetic: "data/synthetic"
  models:    "data/models"
  reports:   "data/reports"

seed: 42
log_level: "INFO"        # DEBUG | INFO | WARNING
```

### configs/toyo_ascii.yaml — 데이터 파일 적응

주요 항목만 발췌합니다. 전체 내용은 파일을 직접 참조하세요.

```yaml
encoding_candidates: [utf-8-sig, utf-8, cp932, shift_jis, euc-jp]
delimiter_candidates: [",", "\t", ";", "whitespace"]

column_aliases:
  voltage_v: ["Voltage(V)", "Voltage", "電圧", ...]
  current_ma: ["Current(mA)"]   # mA 단위 컬럼은 별도 키

unit_conversion:
  current_ma_to_a: true         # mA → A 자동 변환
  capacity_mah_to_ah: true      # mAh → Ah 자동 변환

current_sign:
  discharge_positive_in_file: false  # 파일 내 방전 부호
  pybamm_discharge_positive: false   # PyBaMM half-cell 부호

mode_label_map:
  cc_charge:    ["CC-Chg", "CCCharge", "CC充電", ...]
  cc_discharge: ["CC-Dchg", "CCDischarge", "CC放電", ...]
  rest:         ["Rest", "OCV", "休止"]
```

### configs/halfcell_lmr_base.yaml — LMR 물리 파라미터

Stage 1 fitting의 초기값과 고정 파라미터를 정의합니다.

```yaml
phases:
  - id: "rhombohedral_R3m"
    pybamm_key: "Primary"
  - id: "monoclinic_C2m"
    pybamm_key: "Secondary"

positive_electrode:
  rhombohedral_R3m:
    particle_radius_m: 5.0e-6
    D_s_m2_s: 1.0e-14
    active_fraction: 0.6        # R3m + C2m = 1
  monoclinic_C2m:
    particle_radius_m: 3.0e-6
    D_s_m2_s: 1.0e-15
    active_fraction: 0.4

ocp_mode: "equilibrium"   # equilibrium | charge_discharge_split
```

---

## 8. 테스트 실행

```bash
# 전체 테스트 (PyBaMM, PyTorch 설치 필요)
pytest

# 빠른 테스트 (PyBaMM 없이도 실행 가능한 것만)
pytest tests/test_toyo_parser.py tests/test_ocp_tanh.py tests/test_ocp_grid.py tests/test_profile_resampling.py

# DL 모델 overfit 테스트 (PyTorch 필요, 수십 초 소요)
pytest tests/test_inverse_model_overfit.py -v

# coverage 포함
pytest --cov=lmro2phase --cov-report=term-missing
```

### Acceptance test 체크리스트

| 테스트 | 설명 |
|--------|------|
| `test_toyo_parser.py` | 실제 CSV + 합성 CSV + 일본어 헤더 파싱 |
| `test_ocp_tanh.py` | tanh OCP numpy/PyBaMM 평가, 직렬화 roundtrip |
| `test_ocp_grid.py` | 256점 grid 생성, smoothing, 6가지 perturbation |
| `test_profile_resampling.py` | 용량 격자 보간, feature tensor 형태/NaN 처리 |
| `test_stage1_objective.py` | Stage 1 loss 유한값 반환 (PyBaMM 필요) |
| `test_inverse_model_overfit.py` | 100샘플 200 epoch 과적합 확인 (PyTorch 필요) |

---

## 9. 자주 묻는 질문

**Q. `setup_env.sh` 실행 후 PyBaMM import 오류가 납니다.**

```bash
pip install pybamm>=26.4,<27 --upgrade
```

**Q. PyBaMM positive 2-phase half-cell이 실패합니다.**

정상입니다. `00_smoke_test_pybamm_halfcell.py` 실행 결과에 `"selected_strategy": "fallback_a"` 가 기록되면 이후 모든 단계에서 가중 혼합 OCP surrogate(FallbackA)를 자동 사용합니다.

**Q. Stage 2 생성 중 실패 케이스가 많습니다.**

`data/synthetic/failed_cases.parquet`에서 오류 메시지를 확인하세요.
파라미터 범위가 너무 넓으면 `configs/stage2_generate_synthetic.yaml`의 `param_bounds`를 좁혀보세요.

**Q. 다른 C-rate, 다른 전압 범위의 데이터를 사용하고 싶습니다.**

`configs/halfcell_lmr_base.yaml`의 `voltage.upper_v` / `voltage.lower_v`를 수정하고,
`configs/stage1_fit_tanh.yaml`의 `input.use_cycles`로 사용할 사이클을 지정하세요.

**Q. LMR charge/discharge hysteresis 보정이 필요합니다.**

`configs/halfcell_lmr_base.yaml`에서 아래와 같이 변경합니다.

```yaml
ocp_mode: "charge_discharge_split"   # equilibrium → 이것으로 변경
```

Stage 1 잔차가 충전/방전에서 체계적으로 반대 방향이면 이 모드를 사용하세요.

**Q. conda 환경을 사용합니다.**

`setup_env.sh`의 venv 생성 부분을 건너뛰고 conda 환경을 활성화한 뒤 직접 설치합니다.

```bash
conda activate my_env
pip install -e ".[dev]"
```
