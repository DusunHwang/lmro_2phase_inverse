# 최종 Initial Condition Report

## 목적

LMR native 2-phase PyBaMM 모델에서 R3m/C2m phase를 별도로 모델링하고, C2m이 낮은 전압에 넓고 큰 OCP contribution을 갖는 조건의 TOYO 형식 충방전 데이터를 생성했다.

## 최종 조건

| 항목 | 설정 |
|---|---:|
| 모델 | PyBaMM native positive-electrode 2-phase SPM |
| Phase mapping | Primary = R3m, Secondary = C2m |
| OCP shape | full-range Gaussian redox OCP |
| R3m center / sigma | `3.70 V` / `0.075 V` |
| C2m center / sigma | `3.20 V` / `0.18 V` |
| R3m diffusivity | `4.59e-17 m2/s` |
| C2m diffusivity | `1.00e-18 m2/s` |
| R3m radius | `1.5e-7 m` |
| C2m radius | `1.5e-7 m` |
| R3m active material fraction | `0.2217` |
| C2m active material fraction | `0.4433` |
| C-rate | `0.1C`, `0.33C`, `0.5C`, `1C` |
| 전압 범위 | `2.5 V` to `4.65 V` |
| Rest | 충전/방전 사이 `10 min`, 방전/충전 사이 `10 min` |
| dQ/dV 계산 | `Q(V)` 10 mV grid interpolation 후 finite difference |
| 2C | 제외 |

## OCP Q(V) 및 dQ/dV

![OCP QV and dQdV](../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/phase_ocp_qv_and_dqdv_profiles.png)

| Phase | weighted Q span | phase dQ/dV peak |
|---|---:|---:|
| R3m | `0.3267` | `3.700 V` |
| C2m | `0.6533` | `3.199 V` |

C2m의 weighted Q span은 R3m의 약 2배다.

## Terminal dQ/dV

![Terminal dQdV](../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/native_2phase_dqdv_overlay_by_crate.png)

| C-rate | C2m low/broad feature | R3m high/narrow feature |
|---:|---:|---:|
| `0.1C` | `3.100 V`, `-0.00671 Ah/V` | `3.600 V`, `-0.00655 Ah/V` |
| `0.33C` | `3.010 V`, `-0.00614 Ah/V` | `3.540 V`, `-0.00622 Ah/V` |
| `0.5C` | `2.970 V`, `-0.00529 Ah/V` | `3.520 V`, `-0.00594 Ah/V` |
| `1C` | `2.910 V`, `-0.00340 Ah/V` | `3.480 V`, `-0.00550 Ah/V` |

## 파일 위치

- TOYO CSV: [../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv](../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv)
- true parameter: [../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/true_native_2phase_parameters.json](../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/true_native_2phase_parameters.json)
- OCP profile summary: [../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/phase_ocp_qv_profile_summary.json](../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/phase_ocp_qv_profile_summary.json)
- terminal dQ/dV summary: [../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/native_2phase_dqdv_overlay_summary.json](../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/native_2phase_dqdv_overlay_summary.json)
- round-trip parse: [../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/roundtrip_check.json](../final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/roundtrip_check.json)

## 검증

- 분석 방식: `Q(V)` 10 mV interpolation 기반 dQ/dV
- 테스트: `pytest tests/test_toyo_parser.py tests/test_ocp_tanh.py` 결과 `8 passed`
- 재현 스크립트 복사본: [../scripts/generate_toyo_native_2phase_sample.py](../scripts/generate_toyo_native_2phase_sample.py), [../scripts/analyze_native_2phase_sample.py](../scripts/analyze_native_2phase_sample.py)
