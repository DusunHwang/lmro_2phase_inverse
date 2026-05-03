# Generated Initial Condition Data

## 최종 산출물

최종 추천 케이스는 C2m을 낮은 전압에 넓게 배치하고, R3m 대비 약 2배의 weighted Q contribution을 갖도록 만든 D100x slow native 2-phase 시뮬레이션이다.

- 최종 report: [reports/최종_InitialCondition_Report.md](reports/최종_InitialCondition_Report.md)
- 최종 데이터 폴더: [final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample](final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample)
- TOYO CSV: [final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv](final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv)
- OCP Q(V) / dQ/dV plot: [final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/phase_ocp_qv_and_dqdv_profiles.png](final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/phase_ocp_qv_and_dqdv_profiles.png)
- Terminal dQ/dV plot: [final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/native_2phase_dqdv_overlay_by_crate.png](final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/native_2phase_dqdv_overlay_by_crate.png)

## 폴더 구조

```text
data/generated_initial_condition/
  README.md
  final/
    native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample/
  comparisons/
    native_2phase_equal_radius_150nm_sample/
    native_2phase_gaussian_redox_fullrange_sample/
    native_2phase_gaussian_redox_fullrange_D100x_slow_sample/
    native_2phase_gaussian_redox_swapped_centers_D100x_slow_sample/
  reports/
  scripts/
```

## 재현 명령

```bash
.venv/bin/python scripts/generate_toyo_native_2phase_sample.py \
  --ocp-shape gaussian \
  --phase-radius-m 1.5e-7 \
  --d-r3m-m2-s 4.59e-17 \
  --d-c2m-m2-s 1.0e-18 \
  --r3m-center-v 3.7 \
  --c2m-center-v 3.2 \
  --r3m-sigma-v 0.075 \
  --c2m-sigma-v 0.18 \
  --frac-r3m 0.3333333333 \
  --out-dir data/generated_initial_condition/final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample

.venv/bin/python scripts/analyze_native_2phase_sample.py \
  --out-dir data/generated_initial_condition/final/native_2phase_gaussian_c2m_low_broad_2x_D100x_slow_sample \
  --grid-v 0.01
```
