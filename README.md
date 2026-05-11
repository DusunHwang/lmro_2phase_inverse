# lmro-2phase-inverse

---

<!-- ============================================================ -->
<!-- 업데이트: 2026-05-12 00:09:22 KST — fitting 자동 report wrapper 및 dynamic bounds -->
<!-- ============================================================ -->

## 2026-05-12 00:09:22 KST 업데이트 — fitting 자동 report wrapper 및 dynamic bounds

이번 업데이트에서는 LMR 2상 fitting 실행 후 같은 결과 폴더에 `피팅_결과_리포트.md`를 자동 생성하는 wrapper를 추가하고,
Optuna가 `true_params`를 초기값으로 쓰지 않도록 정리했다. 탐색은 warmup trial의 유효 결과만 이용해 dynamic bounds를 좁힌 뒤
남은 trial을 이어서 수행한다.

### 주요 변경사항

| 구분 | 변경 내용 |
|:---|:---|
| 자동 실행 wrapper | `scripts/run_lmr_fit_and_report.py` 추가 |
| DFN 전용 자동 wrapper | `scripts/run_dfn_2phase_fit_and_report.py` 추가 |
| report 생성 | fitting 완료 후 실제 timestamp 결과 폴더를 파싱하여 `generate_lmr_fit_report.py` 자동 실행 |
| dynamic bounds | warmup 유효 trial의 상위 loss 후보를 기준으로 refined bounds를 산출 |
| true parameter 사용 | `true_params`는 모니터/최종 비교용으로만 사용하고 Optuna 초기 trial에는 사용하지 않음 |
| history type 지원 | report 생성기가 `optuna`, `optuna_warmup`, `optuna_refined`를 모두 Optuna 이력으로 처리 |

### 권장 실행

DFN fitting과 report 생성을 한 번에 수행:

```bash
.venv/bin/python scripts/run_dfn_2phase_fit_and_report.py \
  --out-dir data/fit_results/DFN_2phase_dynamic_bounds_200trial_dqdv \
  --data-csv data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_0p1C_0p33C_0p5C_1C.csv \
  --true-params data/raw/toyo/lmr_dfn_2phase_sample/true_lmr_dfn_parameters.json \
  --n-optuna 200 \
  --n-scipy 0 \
  --n-jobs 20 \
  --optuna-cycle-workers 4 \
  --branch-points 200 \
  --dynamic-bounds-warmup 30 \
  --loss-w-vt 0 \
  --loss-w-vq 0 \
  --loss-w-dqdv 1
```

SPMe/DFN을 직접 선택하는 공용 wrapper:

```bash
.venv/bin/python scripts/run_lmr_fit_and_report.py \
  --fit-model DFN \
  --out-dir data/fit_results/DFN_2phase_dynamic_bounds_200trial_dqdv \
  --data-csv data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_0p1C_0p33C_0p5C_1C.csv \
  --n-optuna 200 \
  --n-scipy 0
```

생성되는 주요 파일:

- `best_params.json`
- `comparison.json`
- `dynamic_bounds.json`
- `fit_history.jsonl`
- `parallel_config.json`
- `피팅_결과_리포트.md`
- `figures/*.png`

검증 기록:

- fitting + report smoke 실행 통과: `data/fit_results/260512_0004_fit_and_report_smoke/피팅_결과_리포트.md`
- 전체 테스트 통과: `18 passed, 2 warnings`

---

<!-- ============================================================ -->
<!-- 업데이트: 2026-05-07 00:32:58 KST — LMR 2상 fitting 스크립트 일반화 -->
<!-- ============================================================ -->

## 2026-05-07 00:32:58 KST 업데이트 — LMR 2상 fitting/report 스크립트 일반화

이번 업데이트에서는 SPMe 전용으로 보이던 2상 역추정 스크립트명을 DFN/SPMe 공용 이름으로 정리하고,
dQ/dV 기반 fitting 입력 샘플링 및 리포트 기록 방식을 보강했다.

### 주요 변경사항

| 구분 | 기존 | 변경 후 |
|:---|:---|:---|
| 2상 fitting 엔진 | `scripts/run_spme_2phase_fit.py` | `scripts/run_lmr_2phase_fit.py` |
| fitting report 생성기 | `scripts/generate_spme_fit_report.py` | `scripts/generate_lmr_fit_report.py` |
| DFN 전용 실행 wrapper | 없음 | `scripts/run_dfn_2phase_fit.py` |
| fitting + report wrapper | 없음 | `scripts/run_lmr_fit_and_report.py`, `scripts/run_dfn_2phase_fit_and_report.py` |
| fitting model 선택 | SPMe 중심 | `--fit-model SPMe` 또는 `--fit-model DFN` |
| 결과 폴더명 | 사용자 지정 이름 그대로 사용 | `YYMMDD_hhmm_` prefix 자동 부여 |
| timestamp 기준 | Docker local time/UTC 영향 가능 | KST 고정 |
| dQ/dV fitting sampling | 균일 time subsample | 충전/방전 branch별 목표 point 수 기반 가변 sampling |
| dQ/dV loss | 방전 branch 중심 | 충전/방전 branch 모두 사용 |
| Optuna 종료 조건 | `--n-optuna` 고정 trial 수 | `--early-stop-loss`, `--early-stop-patience`, `--early-stop-min-delta` 선택 사용 |
| Optuna 탐색 범위 | 고정 bounds에서 전체 trial 수행 | warmup 유효 trial 기반 dynamic bounds 축소 |

### 현재 권장 실행 예시

DFN fitting model, dQ/dV-only loss, 충전/방전 각각 약 200 point, Optuna 120 trial:

```bash
.venv/bin/python scripts/run_dfn_2phase_fit_and_report.py \
  --out-dir data/fit_results/DFN_2phase_병렬_optun_dqdv_charge_discharge_200pt \
  --data-csv data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_0p1C_0p33C_0p5C_1C.csv \
  --true-params data/raw/toyo/lmr_dfn_2phase_sample/true_lmr_dfn_parameters.json \
  --n-optuna 120 \
  --n-scipy 0 \
  --n-jobs 20 \
  --optuna-cycle-workers 4 \
  --branch-points 200 \
  --dynamic-bounds-warmup 30 \
  --dynamic-bounds-top-fraction 0.35 \
  --early-stop-patience 30 \
  --early-stop-min-delta 1e-5 \
  --loss-w-vt 0 \
  --loss-w-vq 0 \
  --loss-w-dqdv 1
```

동일 엔진에서 SPMe를 직접 선택하려면:

```bash
.venv/bin/python scripts/run_lmr_fit_and_report.py \
  --fit-model SPMe \
  --out-dir data/fit_results/SPMe_2phase_병렬_optun_dqdv_charge_discharge_200pt \
  --branch-points 200 \
  --n-optuna 120 \
  --n-scipy 0 \
  --n-jobs 20 \
  --optuna-cycle-workers 4 \
  --dynamic-bounds-warmup 30 \
  --dynamic-bounds-top-fraction 0.35 \
  --early-stop-patience 30 \
  --early-stop-min-delta 1e-5 \
  --loss-w-vt 0 \
  --loss-w-vq 0 \
  --loss-w-dqdv 1
```

결과 리포트 생성:

```bash
.venv/bin/python scripts/generate_lmr_fit_report.py \
  --fit-dir data/fit_results/<YYMMDD_hhmm_...결과폴더> \
  --data-csv data/raw/toyo/lmr_dfn_2phase_sample/Toyo_LMR_DFN_2phase_0p1C_0p33C_0p5C_1C.csv
```

### 결과 폴더 및 리포트 기록 규칙

- fitting 실행 시 결과 폴더 앞에 KST 기준 `YYMMDD_hhmm_`가 자동으로 붙는다.
- `parallel_config.json`에는 `fit_model`, Optuna 병렬 조건, early stop 조건, branch별 sampling point 수, loss weight가 기록된다.
- `parallel_config.json`에는 초기 parameter bounds와 dynamic bounds 설정이 기록되고, `dynamic_bounds.json`에는 warmup 결과로 축소된 bounds가 기록된다.
- `피팅_결과_리포트.md` 생성일은 `YYYY-MM-DD HH:MM:SS KST`로 기록된다.
- `dQ/dV(V) loss point 수` 표에는 각 C-rate별 sample 전체점, 충전점, 방전점, rest점, dQ/dV finite overlap point가 기록된다.

---

<!-- ============================================================ -->
<!-- 업데이트: 2026-05-03  — 현재 상태 전체 재정리                -->
<!-- ============================================================ -->

## 📋 2026-05-03 업데이트 — 현재 상태 전체 정리

> 이 섹션은 2026-05-03 기준 실제 폴더 구조·실행 파일·입출력을 기술합니다.
> 초기 작성 README는 아래 [─── 초기 README (작성 당시) ───](#초기-readme-작성-당시) 에 보존됩니다.

---

### 목차 (2026-05-03)

1. [프로젝트 개요](#1-프로젝트-개요)
2. [전체 폴더 구조](#2-전체-폴더-구조)
3. [환경 설치](#3-환경-설치)
4. [파이프라인 스크립트 — 입출력 상세](#4-파이프라인-스크립트--입출력-상세)
5. [보조 스크립트 — 입출력 상세](#5-보조-스크립트--입출력-상세)
6. [설정 파일 레퍼런스](#6-설정-파일-레퍼런스)
7. [생성된 가상 데이터 샘플 목록](#7-생성된-가상-데이터-샘플-목록)
8. [데이터 흐름 요약](#8-데이터-흐름-요약)
9. [테스트 실행](#9-테스트-실행)

---

### 1. 프로젝트 개요

LMR(Li-rich Mn-rich) 양극 / Li metal 음극 half-cell에서 측정한 충방전 데이터로부터
**2-phase(R3m + C2m) 별 확산계수(D), 입자 반경(R), 상 분율(frac), OCP 형상**을 역추정하는 파이프라인.

```
TOYO 실측 CSV / 가상 시뮬레이션 CSV
  ↓ [Stage 1] PyBaMM native 2-phase 시뮬레이션 + tanh OCP fitting
  ↓ [Stage 2] 파라미터 샘플링 → synthetic dataset 대량 생성
  ↓ [Stage 3] DL inverse model (1D CNN + Transformer) 학습
  ↓ [Stage 4] 실측 프로파일 → 상별 D, R, frac, OCP 추정
```

**2가지 충방전 곡선 생성 경로:**

| 경로 | 스크립트 | 모델 | OCP 표현 |
|------|---------|------|---------|
| **Effective (FallbackA)** | `generate_toyo_sech2_pybamm_sample.py` | SPMe single-phase half-cell | tanh/sech² basis |
| **Native 2-phase** | `generate_toyo_native_2phase_sample.py` | SPM `particle_phases=("1","2")` | `_plateau_ocp()` 또는 `_gaussian_redox_ocp()` |

---

### 2. 전체 폴더 구조

```
lmro_2phase_inverse/
│
├── setup_env.sh                        # 환경 설치 스크립트 (aarch64 자동 패치 포함)
├── pyproject.toml                      # 패키지 메타데이터 & 콘솔 엔트리포인트 정의
│
├── configs/                            # ← 이식 시 이 폴더만 수정
│   ├── env.yaml                        #   시스템별 설정 (CUDA URL, 경로, 병렬화)
│   ├── toyo_ascii.yaml                 #   데이터 파일 형식 적응 (인코딩·컬럼·단위·부호)
│   ├── halfcell_lmr_base.yaml          #   LMR 전극 물리 파라미터 기반값
│   ├── stage0_smoke.yaml               #   Stage 0 옵션
│   ├── stage1_fit_tanh.yaml            #   Stage 1 Optuna/scipy 설정 + loss weights
│   ├── stage1_fit_tanh_quicktest.yaml  #   Stage 1 빠른 검증용 (trials 수 축소)
│   ├── stage2_generate_synthetic.yaml  #   Stage 2 샘플 수·범위·병렬화
│   ├── stage3_train_inverse.yaml       #   Stage 3 모델 아키텍처·학습 하이퍼파라미터
│   └── stage4_infer_validate.yaml      #   Stage 4 추론·forward validation 설정
│
├── scripts/                            # 실행 스크립트 전체
│   │
│   │   ── 메인 파이프라인 (번호 순서대로 실행) ──
│   ├── 00_smoke_test_pybamm_halfcell.py
│   ├── 01_parse_toyo_ascii.py
│   ├── 02_fit_tanh_ocp.py
│   ├── 03_generate_synthetic_dataset.py
│   ├── 04_train_inverse_model.py
│   ├── 05_infer_lmr_profile.py
│   ├── 06_forward_validate.py
│   ├── 07_generate_report.py
│   │
│   │   ── 보조 스크립트 (가상 데이터 생성·분석) ──
│   ├── generate_toyo_sech2_pybamm_sample.py    # Effective SPMe + tanh OCP 가상 데이터
│   ├── generate_toyo_native_2phase_sample.py   # Native 2-phase SPM 가상 데이터
│   ├── analyze_native_2phase_sample.py         # dQ/dV 분석·플롯 (위 스크립트 후처리)
│   │
│   └── patches/                                # 플랫폼별 패치 파일
│       ├── idaklu_stub.py                      #   Python/casadi idaklu stub
│       └── pybammsolvers_init.py               #   try/except __init__ 패치
│
├── src/lmro2phase/                     # 패키지 소스
│   ├── cli/commands.py                 #   lmro-* 콘솔 엔트리포인트
│   │
│   ├── io/
│   │   ├── toyo_ascii.py               #   TOYO CSV 파서 (자동 인코딩·구분자·헤더)
│   │   ├── profile_schema.py           #   BatteryProfile / ProfileSegment 구조
│   │   ├── profile_cleaning.py         #   전처리 (이상값·단위·부호)
│   │   └── dataset_store.py            #   parquet/zarr I/O
│   │
│   ├── physics/
│   │   ├── simulator.py                #   run_current_drive() / run_experiment() 공통 인터페이스
│   │   ├── halfcell_model_factory.py   #   SPM/SPMe/DFN half-cell 빌더 + 2-phase 지원 판별
│   │   ├── positive_2phase_factory.py  #   FallbackA: U_eff / D_eff / R_eff 가중 혼합
│   │   ├── lmr_parameter_set.py        #   build_pybamm_halfcell_params() — Chen2020 기반
│   │   ├── ocp_tanh.py                 #   tanh basis OCP (TanhOCPParams, numpy/pybamm)
│   │   ├── ocp_grid.py                 #   256점 자유형 OCPGrid
│   │   ├── ocp_perturbation.py         #   GP/spline/plateau/shoulder 등 6모드 변동 생성
│   │   ├── protocol_builder.py         #   측정 전류 interpolant + pybamm.Experiment
│   │   └── pybamm_env.py               #   환경 확인 유틸
│   │
│   ├── fitting/
│   │   ├── stage1_objective.py         #   compute_loss() — V(Q)/V(t)/dVdQ/dQdV/rest
│   │   └── stage1_optimizer.py         #   run_optuna_search() + run_scipy_refinement()
│   │
│   ├── generation/
│   │   ├── sampler.py                  #   SampleRecord 파라미터 샘플러 (local/broad/edge)
│   │   ├── batch_simulate.py           #   ProcessPoolExecutor 병렬 배치 시뮬레이션
│   │   └── quality_filter.py           #   비물리적 결과 필터 (전압 범위 등)
│   │
│   ├── features/
│   │   ├── capacity_axis.py            #   정규화 용량 격자 보간 (512점)
│   │   ├── differential_features.py    #   dV/dQ, dQ/dV → [10, 512] feature tensor
│   │   └── normalization.py            #   feature 정규화 통계
│   │
│   ├── learning/
│   │   ├── model_inverse.py            #   InverseModel: 1D CNN ResBlock + Transformer
│   │   ├── dataset.py                  #   SyntheticDataset (zarr + parquet)
│   │   ├── losses.py                   #   permutation-invariant OCP loss + smoothness
│   │   ├── train.py                    #   학습 루프 (AdamW, CosineAnnealingLR)
│   │   └── infer.py                    #   추론 + 결과 저장
│   │
│   └── validation/
│       ├── forward_validate.py         #   예측 파라미터 → PyBaMM → 잔차 계산
│       └── plots.py                    #   V-t / Q-V / dQ-dV 비교 플롯
│
├── tests/
│   ├── test_toyo_parser.py
│   ├── test_ocp_tanh.py
│   ├── test_ocp_grid.py
│   ├── test_profile_resampling.py
│   ├── test_stage1_objective.py
│   └── test_inverse_model_overfit.py
│
└── data/
    ├── raw/toyo/                                   # 원본 및 가상 데이터
    │   ├── Toyo_LMR_HalfCell_Sample_50cycles.csv   #   실측 TOYO 데이터 (50사이클)
    │   ├── sech2_pybamm_sample/                    #   가상: Effective SPMe + tanh OCP
    │   ├── native_2phase_sample/                   #   가상: Native 2-phase, plateau OCP
    │   ├── native_2phase_equal_radius_sample/      #   가상: R_R3m=R_C2m=1μm
    │   ├── native_2phase_equal_radius_150nm_sample/#   가상: R_R3m=R_C2m=150nm
    │   ├── native_2phase_gaussian_redox_sample/    #   가상: Gaussian OCP (초기 중심)
    │   ├── native_2phase_gaussian_redox_fullrange_sample/
    │   ├── native_2phase_gaussian_redox_fullrange_D100x_slow_sample/
    │   ├── native_2phase_gaussian_redox_swapped_centers_D100x_slow_sample/
    │   └── native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/  # ★ 최종 기준 데이터
    │
    ├── generated_initial_condition/                # 초기 조건 탐색 과정 전체 아카이브
    │   ├── README.md                               #   최종 케이스 + 재현 명령
    │   ├── comparisons/                            #   비교 실험 데이터 (중간 단계)
    │   ├── final/                                  #   최종 선택 케이스
    │   ├── reports/                                #   각 단계 검증 보고서 (md)
    │   └── scripts/                                #   생성 당시 스크립트 사본
    │
    ├── reports/                                    # 파이프라인 결과물
    │   ├── inference/cycle_001/ ... cycle_NNN/     #   Stage 4a 추론 결과
    │   └── result_report_YYMMDD_HHMMSS/            #   Stage 7 전체 보고서 (생성 시)
    │
    ├── processed/                                  # Stage 1a 파싱 결과 (생성 시)
    ├── synthetic/                                  # Stage 2 합성 데이터셋 (생성 시)
    └── models/inverse/                             # Stage 3 학습 모델 (생성 시)
```

---

### 3. 환경 설치

```bash
cd lmro_2phase_inverse
bash setup_env.sh          # .venv 생성 + 전체 의존성 설치
source .venv/bin/activate
```

`setup_env.sh`는 플랫폼을 자동 감지합니다:
- **x86_64**: `pip install -e ".[dev]"` 단순 설치
- **aarch64**: cmake 빌드 후 idaklu C++ 확장 로드 테스트 → 실패 시 `scripts/patches/` 의 Python/casadi stub 자동 적용 (casadi ABI 불일치 대응)

설치 확인:
```bash
python -c "import pybamm; print(pybamm.__version__)"
python -c "import lmro2phase; print(lmro2phase.__version__)"
```

---

### 4. 파이프라인 스크립트 — 입출력 상세

모든 스크립트는 프로젝트 루트(`lmro_2phase_inverse/`)에서 실행합니다.  
콘솔 명령(`lmro-*`)은 `pip install -e ".[dev]"` 후 사용 가능합니다.

---

#### `scripts/00_smoke_test_pybamm_halfcell.py`

**목적**: PyBaMM 설치 확인 및 2-phase 지원 전략 판별

| | |
|--|--|
| **입력** | 없음 |
| **출력** | `data/reports/smoke_test_report.json` |
| **콘솔 명령** | `lmro-smoke` |

```bash
python scripts/00_smoke_test_pybamm_halfcell.py
```

출력 JSON의 `selected_strategy` 값:
- `"native"` → PyBaMM이 `particle_phases=("1","2")`를 지원 (이후 native 2-phase 사용)
- `"fallback_a"` → 지원 안 됨 (이후 가중 혼합 OCP surrogate 사용)

---

#### `scripts/01_parse_toyo_ascii.py`

**목적**: TOYO CSV → 정규화된 BatteryProfile parquet

| | |
|--|--|
| **입력** | TOYO CSV 파일 (기본: `data/raw/toyo/Toyo_LMR_HalfCell_Sample_50cycles.csv`) |
| | `configs/toyo_ascii.yaml` (인코딩·구분자·컬럼·부호 설정) |
| **출력** | `data/processed/<원본파일명>_processed.parquet` |
| **콘솔 명령** | `lmro-parse [--input 경로] [--config 경로] [--out_dir 경로]` |

```bash
python scripts/01_parse_toyo_ascii.py \
  --input  data/raw/toyo/Toyo_LMR_HalfCell_Sample_50cycles.csv \
  --config configs/toyo_ascii.yaml \
  --out_dir data/processed
```

파서가 자동 처리하는 항목: 인코딩·구분자·헤더 행 탐색, `Current(mA)→A`, `Capacity(mAh)→Ah`, 전류 부호 통일, Mode 레이블 표준화.

---

#### `scripts/02_fit_tanh_ocp.py`

**목적**: 실측/가상 1사이클 데이터를 기반으로 R3m/C2m tanh OCP 파라미터 초기 피팅

| | |
|--|--|
| **입력** | `configs/stage1_fit_tanh.yaml` (`input.data_file`, `input.use_cycles` 등 포함) |
| | 실측/가상 CSV (`input.data_file`에 지정) |
| **출력** | `data/reports/stage1_fit/best_params.json` — D, R, frac, contact_R 스칼라 |
| | `data/reports/stage1_fit/best_ocp_tanh_R3m.json` — tanh basis (b0, b1, amps, centers, widths) |
| | `data/reports/stage1_fit/best_ocp_tanh_C2m.json` — 동일 구조 |
| | `data/reports/stage1_fit/ocp_phase_plot.png` — 피팅된 OCP 형상 플롯 |
| **콘솔 명령** | `lmro-fit-ocp [--config 경로]` |

```bash
# 기본 실행 (200 trials Optuna + scipy L-BFGS-B)
python scripts/02_fit_tanh_ocp.py

# 빠른 설정 확인 (20 trials)
python scripts/02_fit_tanh_ocp.py --config configs/stage1_fit_tanh_quicktest.yaml
```

최적화 흐름: `Optuna TPE (n_trials)` → `scipy L-BFGS-B (max_iter)` → best 저장  
손실 함수 항목: `w_v_q·V(Q) + w_v_t·V(t) + w_dvdq·dV/dQ + w_dqdv·dQ/dV + w_rest·rest잔차 + w_ocp_smooth·OCP평활 + w_bounds·경계`

---

#### `scripts/03_generate_synthetic_dataset.py`

**목적**: Stage 1 피팅 결과 주변 파라미터를 샘플링하여 대량의 synthetic 충방전 데이터 생성

| | |
|--|--|
| **입력** | `configs/stage2_generate_synthetic.yaml` |
| | `data/reports/stage1_fit/best_params.json` (Stage 1 결과) |
| | `data/reports/stage1_fit/best_ocp_tanh_*.json` (OCP base) |
| **출력** | `data/synthetic/params.parquet` — 성공 케이스 스칼라 파라미터 (N×9) |
| | `data/synthetic/profiles.zarr` — V/I 시계열 (N × 시간점) |
| | `data/synthetic/ocp_profiles.zarr` — OCP 그리드 (N × 2 phase × 256점) |
| | `data/synthetic/failed_cases.parquet` — 시뮬레이션 실패 로그 |
| **콘솔 명령** | `lmro-generate [--n_samples N] [--n_workers W]` |

```bash
# 기본 실행 (configs에서 n_samples 읽음)
python scripts/03_generate_synthetic_dataset.py

# 규모 지정
python scripts/03_generate_synthetic_dataset.py --n_samples 5000 --n_workers 4
```

샘플링 전략: `local` (Stage 1 주변 좁은 범위) / `broad` (사전 분포 전체) / `edge` (극단값)  
OCP 변동 방식 (6종): GP, spline, plateau, transition, shoulder, tanh  
시뮬레이션 모델: SPMe or DFN (configs에서 비율 설정), `pybamm.Experiment`

---

#### `scripts/04_train_inverse_model.py`

**목적**: Synthetic dataset으로 DL inverse model 학습 (features → physics parameters + OCP)

| | |
|--|--|
| **입력** | `configs/stage3_train_inverse.yaml` |
| | `data/synthetic/params.parquet` |
| | `data/synthetic/profiles.zarr` |
| | `data/synthetic/ocp_profiles.zarr` |
| **출력** | `data/models/inverse/best_model.pt` — 학습된 체크포인트 |
| **콘솔 명령** | `lmro-train [--overfit_test]` |

```bash
# 본 학습
python scripts/04_train_inverse_model.py

# overfit 빠른 검증 (100샘플, 500 epoch)
python scripts/04_train_inverse_model.py --overfit_test
```

모델 구조: 1D CNN ResBlock (kernel=7, 6 블록) → 선택적 Transformer (8-head, 2 layer) → GlobalAvgPool  
출력 헤드: 스칼라 [B,9] + OCP_R3m [B,256] + OCP_C2m [B,256]  
입력 feature: [B, 10, 512] — V_chg, I_chg, V_dchg, I_dchg, dV/dQ×2, dQ/dV×2 등 10채널

---

#### `scripts/05_infer_lmr_profile.py`

**목적**: 학습된 모델로 실측/가상 사이클에서 R3m/C2m 파라미터 추론

| | |
|--|--|
| **입력** | `configs/stage4_infer_validate.yaml` |
| | TOYO CSV 파일 (stage4 config에서 지정) |
| | `data/models/inverse/best_model.pt` |
| **출력** | `data/reports/inference/cycle_NNN/predicted_params.json` — 추론된 D, R, frac 등 |
| | `data/reports/inference/cycle_NNN/predicted_ocp_R3m.csv` — (stoichiometry, voltage) × 256 |
| | `data/reports/inference/cycle_NNN/predicted_ocp_C2m.csv` — 동일 구조 |
| **콘솔 명령** | `lmro-infer [--config 경로]` |

```bash
python scripts/05_infer_lmr_profile.py
```

---

#### `scripts/06_forward_validate.py`

**목적**: 추론된 파라미터로 PyBaMM 재시뮬레이션 → 측정값과 잔차 비교

| | |
|--|--|
| **입력** | `configs/stage4_infer_validate.yaml` |
| | `data/reports/inference/cycle_NNN/predicted_params.json` |
| | `data/reports/inference/cycle_NNN/predicted_ocp_*.csv` |
| | 원본 TOYO CSV (재시뮬레이션 기준 전류 프로파일) |
| **출력** | `data/reports/inference/cycle_NNN/forward_validation.png` — V-t / dQ-dV 오버레이 플롯 |
| | `data/reports/inference/cycle_NNN/residual_summary.json` — RMSE, MAE, max_err (V) |
| **콘솔 명령** | `lmro-validate [--config 경로]` |

```bash
python scripts/06_forward_validate.py
```

---

#### `scripts/07_generate_report.py`

**목적**: 파이프라인 전 단계 결과를 하나의 Markdown 보고서로 통합

| | |
|--|--|
| **입력** | `data/raw/toyo/*.csv` (입력 데이터 플롯) |
| | `data/reports/stage1_fit/` (OCP 피팅 결과) |
| | `data/reports/inference/` (추론 결과) |
| | `data/synthetic/` (데이터셋 통계) |
| **출력** | `data/reports/result_report_YYMMDD_HHMMSS/result_report_YYMMDD_HHMMSS.md` |
| | 동 디렉토리에 그림 파일들 (`fig_01_*.png`, ...) |
| **콘솔 명령** | `lmro-report` |

```bash
python scripts/07_generate_report.py
```

---

### 5. 보조 스크립트 — 입출력 상세

#### `scripts/generate_toyo_sech2_pybamm_sample.py`

**목적**: Effective SPMe + tanh/sech² OCP 가상 데이터 생성 (FallbackA 경로)

| | |
|--|--|
| **입력** | CLI 인수 (아래 참조) |
| **출력** | `--out-dir/Toyo_LMR_sech2_PyBaMM_*.csv` — TOYO 형식 충방전 CSV |
| | `--out-dir/ocp_dqdv_sech2_basis.csv` — OCP/dQ-dV 기저 그리드 |
| | `--out-dir/true_parameters.json` — 시뮬레이션 조건 + tanh OCP 파라미터 전체 |
| | `--out-dir/roundtrip_recovery_check.json` — 파서 round-trip 검증 |

```bash
python scripts/generate_toyo_sech2_pybamm_sample.py \
  --model    SPMe \
  --c-rates  0.1 0.33 0.5 1.0 2.0 \
  --out-dir  data/raw/toyo/sech2_pybamm_sample
```

**주요 인수:**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `SPMe` | `SPM`, `SPMe`, `DFN` 중 선택 |
| `--c-rates` | `0.1 0.33 0.5 1.0 2.0` | 사이클별 C-rate (복수 가능) |
| `--voltage-lower` | `2.5` V | 방전 종지 전압 |
| `--voltage-upper` | `4.65` V | 충전 종지 전압 |
| `--rest-minutes` | `10.0` | 충방전 사이 rest 시간 (분) |
| `--period-s` | `5.0` | 출력 시간 해상도 (초) |
| `--out-dir` | `data/raw/toyo/sech2_pybamm_sample` | 출력 폴더 |

---

#### `scripts/generate_toyo_native_2phase_sample.py`

**목적**: PyBaMM native 2-phase SPM 가상 데이터 생성 (R3m/C2m 개별 파라미터 주입)

| | |
|--|--|
| **입력** | CLI 인수 (아래 참조) |
| **출력** | `--out-dir/Toyo_LMR_native2phase_PyBaMM_{C-rate}.csv` — TOYO 형식 충방전 CSV |
| | `--out-dir/native_phase_ocp_basis.csv` — (stoichiometry, U_R3m, U_C2m, U_weighted) |
| | `--out-dir/true_native_2phase_parameters.json` — 전체 조건 (OCP 형상 포함) |
| | `--out-dir/phase_concentration_summary.json` — 사이클별 농도 추이 |
| | `--out-dir/roundtrip_check.json` — 파서 round-trip 검증 |

```bash
# 최종 기준 케이스 재현
python scripts/generate_toyo_native_2phase_sample.py \
  --ocp-shape    gaussian \
  --phase-radius-m 1.5e-7 \
  --d-r3m-m2-s   4.59e-17 \
  --d-c2m-m2-s   1.0e-18 \
  --r3m-center-v 3.7  --c2m-center-v 3.2 \
  --r3m-sigma-v  0.075 --c2m-sigma-v  0.18 \
  --frac-r3m     0.3333333333 \
  --c-rates      0.1 0.33 0.5 1.0 \
  --out-dir      data/raw/toyo/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample
```

**주요 인수:**

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--ocp-shape` | `plateau` | `plateau` 또는 `gaussian` |
| `--frac-r3m` | `0.60` | R3m 상 분율 (0~1) |
| `--d-r3m-m2-s` | *(default_truth)* | R3m 확산계수 (m²/s) |
| `--d-c2m-m2-s` | *(default_truth)* | C2m 확산계수 (m²/s) |
| `--phase-radius-m` | *(default_truth)* | 두 phase 공통 반경 (m) |
| `--r3m-center-v` | `3.18` | R3m Gaussian OCP 중심 전압 (V) |
| `--c2m-center-v` | `3.82` | C2m Gaussian OCP 중심 전압 (V) |
| `--r3m-sigma-v` | `0.075` | R3m Gaussian OCP 폭 σ (V) |
| `--c2m-sigma-v` | `0.080` | C2m Gaussian OCP 폭 σ (V) |
| `--c-rates` | `0.1 0.33 0.5 1.0` | 사이클별 C-rate |
| `--voltage-lower` | `2.5` | 방전 종지 전압 (V) |
| `--voltage-upper` | `4.65` | 충전 종지 전압 (V) |
| `--rest-minutes` | `10.0` | 충방전 사이 rest (분) |
| `--period-s` | `0.5` | 시간 해상도 (초) |
| `--initial-fraction` | `0.98` | 초기 SOC 분율 |
| `--out-dir` | `data/raw/toyo/native_2phase_sample` | 출력 폴더 |

---

#### `scripts/analyze_native_2phase_sample.py`

**목적**: native 2-phase 시뮬레이션 결과의 dQ/dV 분석·플롯 생성 (`generate_toyo_native_2phase_sample.py` 후처리)

| | |
|--|--|
| **입력** | `--out-dir` (위 스크립트 출력 폴더) |
| | `--csv` (폴더 내 CSV 파일명, 기본: 폴더 내 자동 탐색) |
| | `--grid-v` (dQ/dV 계산 전압 해상도, 기본: 0.01 V) |
| **출력** | `--out-dir/native_2phase_dqdv_overlay_by_crate.png` — C-rate별 방전 dQ/dV 오버레이 |
| | `--out-dir/native_2phase_dqdv_overlay_summary.json` — 피크 전압·진폭 수치 |
| | `--out-dir/native_2phase_protocol_step_summary.json` — 충방전 step 요약 |
| | `--out-dir/native_2phase_monotonicity_check.json` — OCP 단조성 검증 |

```bash
python scripts/analyze_native_2phase_sample.py \
  --out-dir data/raw/toyo/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample \
  --grid-v  0.01
```

---

### 6. 설정 파일 레퍼런스

#### `configs/env.yaml`

```yaml
python:
  venv_path: ".venv"

torch:
  index_url: ""          # "" = CPU / PyPI 기본값
  device: "auto"         # "auto" | "cpu" | "cuda" | "cuda:0"

parallel:
  n_jobs: 4              # Stage 2 병렬 프로세스 수
  pybamm_nthreads: 1     # PyBaMM 내부 OpenMP 스레드 수

paths:
  data_root:  "data"
  raw_data:   "data/raw/toyo"
  processed:  "data/processed"
  synthetic:  "data/synthetic"
  models:     "data/models"
  reports:    "data/reports"

seed: 42
log_level: "INFO"
```

#### `configs/toyo_ascii.yaml`

| 설정 항목 | 역할 |
|---------|------|
| `encoding_candidates` | UTF-8, CP932, Shift-JIS 등 순서대로 시도 |
| `column_aliases` | `Voltage(V)`, `電圧`, `V` 등 → `voltage_v` 로 통일 |
| `unit_conversion.current_ma_to_a` | `Current(mA)` → A 자동 변환 |
| `current_sign.discharge_positive_in_file` | 파일 내 방전 부호 (TOYO: `false`) |
| `mode_label_map` | `CC-Chg`, `CC-Dchg`, `Rest` → `StepMode` enum |
| `protocol_detection` | Mode 컬럼 없을 때 전류·전압 패턴으로 자동 분류 |

#### `configs/stage1_fit_tanh.yaml`

```yaml
input:
  data_file: "data/raw/toyo/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/
              Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv"
  toyo_config: "configs/toyo_ascii.yaml"
  use_cycles: [1]         # 사용할 사이클 번호 목록

fitting:
  tanh_ocp:
    n_terms: 5
  optimizer:
    global_search:
      n_trials: 200
      n_jobs: 1
      timeout_s: 3600
    local:
      max_iter: 500
  loss_weights:
    w_v_q: 1.0
    w_v_t: 0.3
    w_dvdq: 2.0
    w_dqdv: 2.0
    w_rest: 0.5
    w_ocp_smooth: 0.5
    w_bounds: 10.0

output:
  report_dir: "data/reports/stage1_fit"
```

---

### 7. 생성된 가상 데이터 샘플 목록

`data/raw/toyo/` 에 있는 모든 가상 데이터 샘플의 파라미터 요약:

| 샘플 폴더 | OCP 형상 | frac_R3m | D_R3m (m²/s) | D_C2m (m²/s) | R (m) | R3m 중심 (V) | C2m 중심 (V) |
|---------|---------|---------|-------------|-------------|------|------------|------------|
| `sech2_pybamm_sample` | tanh (Effective) | 0.62 | 4.59e-15 | 1.00e-16 | R3m:150nm / C2m:1μm | ~3.25† | ~3.75† |
| `native_2phase_sample` | plateau | 0.60 | 4.59e-15 | 1.00e-16 | R3m:150nm / C2m:1μm | 2.80† | 3.95† |
| `native_2phase_equal_radius_sample` | plateau | 0.60 | 4.59e-15 | 1.00e-16 | 1μm / 1μm | 2.80† | 3.95† |
| `native_2phase_equal_radius_150nm_sample` | plateau | 0.60 | 4.59e-15 | 1.00e-16 | 150nm / 150nm | 2.80† | 3.95† |
| `native_2phase_gaussian_redox_sample` | gaussian | 0.60 | 4.59e-15 | 1.00e-16 | 150nm / 150nm | 3.18 (σ=0.075) | 3.82 (σ=0.08) |
| `native_2phase_gaussian_redox_fullrange_sample` | gaussian | 0.60 | 4.59e-15 | 1.00e-16 | 150nm / 150nm | 3.18 (σ=0.075) | 3.82 (σ=0.08) |
| `native_2phase_gaussian_redox_fullrange_D100x_slow_sample` | gaussian | 0.60 | **4.59e-17** | **1.00e-18** | 150nm / 150nm | 3.18 (σ=0.075) | 3.82 (σ=0.08) |
| `native_2phase_gaussian_redox_swapped_centers_D100x_slow_sample` | gaussian | 0.60 | 4.59e-17 | 1.00e-18 | 150nm / 150nm | **3.70** (σ=0.075) | **3.20** (σ=0.08) |
| **`native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample` ★** | gaussian | **0.333** | 4.59e-17 | 1.00e-18 | 150nm / 150nm | 3.70 (σ=0.075) | **3.20 (σ=0.18)** |

†: plateau OCP는 dU/ds Gaussian 형상으로 설계된 피크 위치 (전압축 직접 기입 아님)  
★: 최종 기준 데이터 — `data/generated_initial_condition/final/` 에 사본 있음

---

### 8. 데이터 흐름 요약

```
[Native 2-phase 경로]

  CLI args
   ├─ frac_R3m, D_R3m, D_C2m, R
   └─ r3m_center_v, r3m_sigma_v, c2m_center_v, c2m_sigma_v
         │
         ▼ _gaussian_redox_ocp() 또는 _plateau_ocp()
  InterpOCP_R3m, InterpOCP_C2m
         │
         ▼ build_native_params(truth, ocp_R3m, ocp_C2m)
  pybamm.ParameterValues (Primary/Secondary 각각 주입)
         │
         ▼ build_native_model()
  pybamm.lithium_ion.SPM(particle_phases=("1","2"))
         │
         ▼ pybamm.Experiment([Charge/Rest/Discharge/Rest])
         ▼ pybamm.CasadiSolver(mode="safe", rtol=1e-5, atol=1e-7)
         ▼ sim.solve()
  time_s, voltage_v, current_a
         │
         ▼ TOYO 형식 CSV + true_native_2phase_parameters.json

────────────────────────────────────────────────

[Inverse pipeline 경로]

  TOYO CSV
   └─ ToyoAsciiParser(toyo_ascii.yaml)
         │
  BatteryProfile(time_s, voltage_v, current_a, segments)
         │
  [Stage 1] compute_loss() ← run_current_drive(SPMe model, pv, t, I)
         ↓ Optuna + scipy L-BFGS-B
  best_params.json + best_ocp_tanh_*.json
         │
  [Stage 2] perturb_ocp_grid() + run_experiment(SPMe/DFN)
         ↓ N=1000~50000 케이스
  synthetic/{params.parquet, profiles.zarr, ocp_profiles.zarr}
         │
  [Stage 3] SyntheticDataset → feature tensor [10,512]
         ↓ InverseModel (CNN+Transformer)
  best_model.pt
         │
  [Stage 4a] infer_profile() → predicted_params.json + predicted_ocp_*.csv
  [Stage 4b] forward_validate() → forward_validation.png + residual_summary.json
         │
  [Stage 7] 전체 보고서 생성 → result_report_YYMMDD_HHMMSS/
```

---

### 9. 테스트 실행

```bash
# 전체 테스트 (PyBaMM + PyTorch 필요)
pytest

# PyBaMM 없이 빠른 유닛 테스트
pytest tests/test_toyo_parser.py tests/test_ocp_tanh.py \
       tests/test_ocp_grid.py tests/test_profile_resampling.py

# DL 모델 overfit 테스트 (수십 초)
pytest tests/test_inverse_model_overfit.py -v

# coverage
pytest --cov=lmro2phase --cov-report=term-missing
```

| 테스트 파일 | 검증 내용 |
|---------|---------|
| `test_toyo_parser.py` | 실제 CSV·합성 CSV·일본어 헤더 파싱, round-trip |
| `test_ocp_tanh.py` | tanh OCP numpy/PyBaMM 평가, 직렬화 round-trip |
| `test_ocp_grid.py` | 256점 grid 생성, smoothing, 6가지 변동 모드 |
| `test_profile_resampling.py` | 용량 격자 보간, feature tensor 형태/NaN |
| `test_stage1_objective.py` | Stage 1 loss 유한값 반환 (PyBaMM 필요) |
| `test_inverse_model_overfit.py` | 100샘플 500 epoch 과적합 확인 (PyTorch 필요) |

---

<!-- ============================================================ -->
<!-- 초기 README (작성 당시)                                      -->
<!-- ============================================================ -->

---

## ─── 초기 README (작성 당시) ───

---

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

editable 설치 후에는 `python scripts/...` 대신 `lmro-*` 콘솔 명령도 사용할 수 있습니다.

### Stage 0 — PyBaMM 환경 확인

```bash
lmro-smoke
# 또는: python scripts/00_smoke_test_pybamm_halfcell.py
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
lmro-parse \
  --input data/raw/toyo/Toyo_LMR_HalfCell_Sample_50cycles.csv \
  --config configs/toyo_ascii.yaml \
  --out_dir data/processed
```

결과: `data/processed/Toyo_LMR_HalfCell_Sample_50cycles_processed.parquet`

### Stage 1b — tanh OCP 초기 피팅

```bash
lmro-fit-ocp --config configs/stage1_fit_tanh.yaml

# 빠른 설정 확인
lmro-fit-ocp --config configs/stage1_fit_tanh_quicktest.yaml
```

결과: `data/reports/stage1_fit/`
- `best_params.json` — 스칼라 파라미터
- `best_ocp_tanh_R3m.json` / `best_ocp_tanh_C2m.json` — OCP tanh 파라미터
- `ocp_phase_plot.png` — OCP 시각화

### Stage 2 — Synthetic dataset 생성

```bash
# pilot 규모 (1,000개)
lmro-generate

# 대규모 (50,000개)
lmro-generate --n_samples 50000
```

결과: `data/synthetic/`
- `params.parquet` — 성공 케이스 파라미터
- `profiles.zarr` — 시뮬레이션 프로파일
- `ocp_profiles.zarr` — OCP grid
- `failed_cases.parquet` — 실패 케이스 로그 (절대 무시하지 않음)

### Stage 3 — DL inverse model 학습

```bash
# overfit 확인 (100개, 빠른 검증)
lmro-train --overfit_test

# 본 학습
lmro-train --config configs/stage3_train_inverse.yaml
```

결과: `data/models/inverse/best_model.pt`

### Stage 4a — 실측 프로파일 추론

```bash
lmro-infer --config configs/stage4_infer_validate.yaml
```

결과: `data/reports/inference/cycle_XXX/`
- `predicted_params.json`
- `predicted_ocp_R3m.csv` / `predicted_ocp_C2m.csv`

### Stage 4b — Forward validation

```bash
lmro-validate --config configs/stage4_infer_validate.yaml
```

결과:
- `forward_validation.png` — 측정값 vs 시뮬레이션 비교
- `residual_summary.json` — RMSE, MAE, 최대 오차

### Stage 5 — 결과 보고서 생성

```bash
lmro-report
```

결과: `data/reports/result_report_YYMMDD_HHMMSS/`

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
│   ├── stage1_fit_tanh_quicktest.yaml
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
│   ├── cli/
│   │   └── commands.py             # lmro-* 콘솔 엔트리포인트
│   │
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
│   ├── 06_forward_validate.py
│   └── 07_generate_report.py
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
