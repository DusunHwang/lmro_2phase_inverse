# Native 2-phase PyBaMM 검증

## 결론

PyBaMM 26.4.1에서 positive half-cell native 2-phase 시뮬레이션을 다시 수행했다. 이번 데이터는 effective 평균 transport가 아니라 `Primary`와 `Secondary` phase에 서로 다른 `D`, `R`, `OCP`, active fraction을 넣은 native 2-phase 결과다.

```text
Primary phase   = R3m
Secondary phase = C2m
```

2C는 충전/방전을 모두 제외했다. 최종 데이터는 `0.1C`, `0.33C`, `0.5C`, `1C`만 포함한다.

## 생성 데이터

- TOYO CSV: `data/raw/toyo/native_2phase_sample/Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv`
- 출력 rows: `148703`
- 전압 범위: `2.500010 V` to `4.649990 V`
- C-rate: `0.1C`, `0.33C`, `0.5C`, `1C`
- Rest: 충전/방전 사이 `10 min`, 방전/충전 사이 `10 min`
- 출력 period: `0.5 s`
- TOYO round-trip parse: `148703` points, `4` cycles, `16` segments

![Native 2-phase dQ/dV overlay](data/raw/toyo/native_2phase_sample/native_2phase_dqdv_overlay_by_crate.png)

## 방전 기준 target peak

동일 OCP라면 방전 전압이 충전보다 낮게 나타나는 것이 자연스럽기 때문에, target voltage는 방전 dQ/dV 기준으로 맞췄다. 충전 dQ/dV는 양수, 방전 dQ/dV는 음수로 계산했다.

| C-rate | R3m 저전압 feature | C2m 고전압 peak |
|---:|---:|---:|
| `0.1C` 방전 | `3.350 V`, `-0.00117 Ah/V` | `3.859 V`, `-0.0140 Ah/V` |
| `0.33C` 방전 | `3.348 V`, `-0.00117 Ah/V` | `3.796 V`, `-0.00958 Ah/V` |
| `0.5C` 방전 | `3.323 V`, `-0.00103 Ah/V` | `3.773 V`, `-0.00600 Ah/V` |
| `1C` 방전 | `3.201 V`, `-0.000588 Ah/V` | `3.731 V`, `-0.00130 Ah/V` |

R3m는 terminal dQ/dV에서 강한 peak라기보다 저전압 shoulder로 나타난다. C2m는 `0.33C` 방전에서 약 `3.80 V`, `1C` 방전에서 약 `3.73 V`로 나타나며, 고율에서 더 낮은 전압으로 이동한다.

## Phase별 입력 파라미터

| Phase | PyBaMM phase | D (m2/s) | R (m) | active material fraction |
|---|---|---:|---:|---:|
| R3m | Primary | `4.59e-15` | `1.50e-7` | `0.3990` |
| C2m | Secondary | `1.00e-16` | `1.00e-6` | `0.2660` |

OCP도 phase별로 따로 주입했다.

```text
Primary: Positive electrode OCP [V]   = R3m monotonic interpolation OCP
Secondary: Positive electrode OCP [V] = C2m monotonic interpolation OCP
```

이번 targeted peak 데이터에서는 R3m를 저전압 feature, C2m를 고전압 feature로 분리하기 위해 sech^2-shaped dQ/dV를 적분한 monotonic OCP interpolation을 사용했다.

## PyBaMM native 2-phase 옵션

사용한 핵심 옵션은 다음과 같다.

```python
{
    "working electrode": "positive",
    "particle phases": ("1", "2"),
    "particle": "Fickian diffusion",
    "particle size": "single",
    "surface form": "differential",
}
```

`particle=("Fickian diffusion", "Fickian diffusion")`, `surface form=false` 조합은 PyBaMM 26.4.1 multiple particle phases에서 실패했다. 따라서 `particle`은 `"Fickian diffusion"` 단일 옵션으로 두고 `surface form`은 `"differential"`을 사용했다.

## 문헌 기반 현실성 검토

현재 native 2-phase 입력값은 다음과 같다.

| Phase mapping | Radius | Diffusivity | cm2/s 환산 |
|---|---:|---:|---:|
| R3m / Primary | `150 nm` | `4.59e-15 m2/s` | `4.59e-11 cm2/s` |
| C2m / Secondary | `1.0 um` | `1.00e-16 m2/s` | `1.00e-12 cm2/s` |

- `R3m / Primary = 150 nm, 4.59e-15 m2/s`는 Li-rich nanoscale CP-FD 문헌값과 잘 맞는다. 해당 문헌은 `100-300 nm` 입자와 `4.59 x 10^-11 cm2/s` Li+ chemical diffusion coefficient를 보고한다.
- `C2m / Secondary = 1 um, 1e-16 m2/s`는 micron-sized Li-rich layered oxide grain surrogate로는 합리적이다. Energy Storage Materials 2022 논문은 micron-sized Li-rich grain의 평균 크기 약 `940 nm`, diffusion coefficient order `10^-12 cm2/s`를 보고한다.
- 다만 이것을 “C2m 결정상 자체가 1 um 독립 입자이고 R3m 결정상 자체가 150 nm 독립 입자”라고 해석하면 과하다. Li-rich layered oxide에서 R-3m/LiMO2 성분과 C2/m/Li2MnO3 성분은 한 입자 안에서 intergrown 또는 복합 도메인으로 존재하는 경우가 많다.

## Reference

1. Bai et al., *Lithium-Rich Nanoscale Li1.2Mn0.54Ni0.13Co0.13O2 Cathode Material Prepared by Co-Precipitation Combined Freeze Drying (CP-FD) for Lithium-Ion Batteries*, Energy Technology, 2015.  
   https://www.anl.gov/argonne-scientific-publications/pub/125613

2. Abraham et al., *Improved electrochemical performance of SiO2-coated Li-rich layered oxides-Li1.2Ni0.13Mn0.54Co0.13O2*, Journal of Materials Science: Materials in Electronics, 2020.  
   https://link.springer.com/article/10.1007/s10854-020-04481-6

3. Zhang et al., *Revealing Li-ion diffusion kinetic limitations in micron-sized Li-rich layered oxides*, Energy Storage Materials, 2022.  
   https://www.sciencedirect.com/science/article/abs/pii/S2405829722005335

4. *Surface modification with oxygen vacancy in Li-rich layered oxide Li1.2Mn0.54Ni0.13Co0.13O2 for lithium-ion batteries*, Journal of Materials Science & Technology, 2019.  
   https://www.jmst.org/article/2019/1005-0302/1005-0302-35-6-994.shtml

## Solver 메모

SUNDIALS/CasADi에서 일부 corrector convergence 및 mxstep warning이 출력됐다. 이는 `D_C2m=1e-16 m2/s`, `R_C2m=1 um`, 4.65 V 상한, targeted plateau OCP 조합 때문에 stiff해진 영향으로 보인다. 2C는 이 조건에서 방전 전환이 불안정하므로 최종 데이터셋에서 충전/방전 모두 제외했다.


## 동일 반경 R=150 nm 추가 케이스

사용자 요청에 따라 두 phase의 particle radius를 모두 `1.5e-7 m`로 맞춘 추가 데이터를 생성했다. 이 케이스는 반경 차이를 제거하고 기존 diffusivity 차이만 남긴 비교용이다.

- TOYO CSV: `data/raw/toyo/native_2phase_equal_radius_150nm_sample/Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv`
- dQ/dV overlay: `data/raw/toyo/native_2phase_equal_radius_150nm_sample/native_2phase_dqdv_overlay_by_crate.png`
- 비교 요약: `data/raw/toyo/native_2phase_equal_radius_150nm_sample/equal_radius_comparison_summary.json`
- rows: `163621`
- C-rate: `0.1C`, `0.33C`, `0.5C`, `1C`
- radius: R3m `1.5e-7 m`, C2m `1.5e-7 m`
- diffusivity: R3m `4.59e-15 m2/s`, C2m `1.00e-16 m2/s`

동일 반경에서 diffusion time scale `R^2/D`는 R3m 약 `4.9 s`, C2m 약 `225 s`다. 따라서 반경 효과는 제거됐고, C2m가 약 46배 느린 확산 phase로 남는다.

| 케이스 | 0.33C C2m peak | 1C C2m peak | 해석 |
|---|---:|---:|---|
| R3m `150 nm`, C2m `1 um` | `3.796 V`, `-0.00958 Ah/V` | `3.731 V`, `-0.00130 Ah/V` | 느린 D와 큰 R이 같이 작용해 고율에서 peak가 크게 약해짐 |
| R3m=C2m=`1 um` | `3.796 V`, `-0.00954 Ah/V` | `3.731 V`, `-0.00131 Ah/V` | 두 phase 모두 큰 입자라 기존과 거의 동일 |
| R3m=C2m=`150 nm` | `3.799 V`, `-0.01272 Ah/V` | `3.741 V`, `-0.01198 Ah/V` | 반경이 작아져 C2m peak가 고율에서도 덜 소실됨. peak shift는 diffusivity 차이로 남음 |

결론적으로 `R=1.5e-7 m` 동일 반경 케이스가 diffusivity 차이만 보기에는 더 적절하다. 다만 terminal dQ/dV의 R3m 저전압 feature는 강한 peak라기보다 shoulder라서 자동 peak picker가 C-rate에 따라 주변 shoulder 중 다른 지점을 잡을 수 있다. C2m 고전압 peak는 더 안정적으로 비교 가능하다.

## 산출물

- 생성 스크립트: `scripts/generate_toyo_native_2phase_sample.py`
- 분석 스크립트: `scripts/analyze_native_2phase_sample.py`
- TOYO CSV: `data/raw/toyo/native_2phase_sample/Toyo_LMR_native2phase_PyBaMM_0p1C_0p33C_0p5C_1C.csv`
- true parameter: `data/raw/toyo/native_2phase_sample/true_native_2phase_parameters.json`
- phase concentration summary: `data/raw/toyo/native_2phase_sample/phase_concentration_summary.json`
- dQ/dV overlay: `data/raw/toyo/native_2phase_sample/native_2phase_dqdv_overlay_by_crate.png`
- dQ/dV peak summary: `data/raw/toyo/native_2phase_sample/native_2phase_dqdv_overlay_summary.json`
- protocol step summary: `data/raw/toyo/native_2phase_sample/native_2phase_protocol_step_summary.json`
- monotonicity check: `data/raw/toyo/native_2phase_sample/native_2phase_monotonicity_check.json`
- round-trip parse: `data/raw/toyo/native_2phase_sample/roundtrip_check.json`
