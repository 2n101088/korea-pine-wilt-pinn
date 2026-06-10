# 코드 상세 설명 — `pine_wilt_KPP_PINN.ipynb`

소나무재선충병 감염목 확산을 **Fisher–KPP 반응-확산 방정식**으로 모형화하고,
이를 **PINN(Physics-Informed Neural Network)** 으로 풀어 미지 물리계수 $D, r$ 를 역추정하고
미래 피해를 예측하는 노트북의 전 과정을 셀 단위로 설명한다.

> 이 문서는 코드의 *동작 원리*를 다룬다. 결과 해석·실행법 요약은 [README.md](README.md) 참고.

## 목차
1. [전체 구조 한눈에](#1-전체-구조-한눈에)
2. [실행 환경 / 방법](#2-실행-환경--방법)
3. [데이터 파이프라인 (셀 1·3)](#3-데이터-파이프라인-셀-13)
4. [EDA (셀 5)](#4-eda-셀-5)
5. [격자·관측밀도 $u_{obs}$ (셀 7·9)](#5-격자관측밀도-u_obs-셀-79)
6. [수학적 모형 — Fisher-KPP와 정규화 (셀 10)](#6-수학적-모형--fisher-kpp와-정규화-셀-10)
7. [PINN 정의 (셀 11)](#7-pinn-정의-셀-11)
8. [학습 — 2단계 최적화 Adam→L-BFGS (셀 13)](#8-학습--2단계-최적화-adaml-bfgs-셀-13) ★
9. [손실·파라미터 수렴 시각화 (셀 14)](#9-손실파라미터-수렴-시각화-셀-14)
10. [예측·검증 (셀 16)](#10-예측검증-셀-16)
11. [결과 시각화 (셀 18–22)](#11-결과-시각화-셀-1822)
12. [소진 고려 공간 SIR (셀 23–25)](#12-소진-고려-공간-sir-셀-2325)
13. [시도별·추이 분석 (셀 27–28)](#13-시도별추이-분석-셀-2728)
14. [요약 출력 (셀 30)](#14-요약-출력-셀-30)
15. [주요 결과·한계](#15-주요-결과한계)

---

## 1. 전체 구조 한눈에

```
CSV(감염목 2016–2023)
  └─[셀 3] 좌표 재투영·캐시 →  PX, PY, PYR (감염목 점 3,183,376건)
       └─[셀 7] 96×96 격자 집계 + 육지마스크 →  U_OBS(7년, 정규화 감염밀도)
            └─[셀 11] PINN  N(x,y,t)→u  +  물리계수 D,r(softplus)
                 └─[셀 13] 2단계 학습 (Adam 워밍업 → L-BFGS 정제)
                      ├─[셀 16] 160×160 예측·검증 (학습연도 재현 / 2023 검증 / 2024–26 예측)
                      ├─[셀 18–22] 지도 시각화
                      ├─[셀 24–25] 소진 고려 공간 SIR (~2030)
                      └─[셀 27–28] 시도별 순위·피해 추이
```

| 단계 | 셀 | 핵심 산출물 |
|---|---|---|
| 데이터 | 1, 3 | `PX, PY, PYR` (+ `cache/*.npz`) |
| 격자/관측 | 7, 9 | `U_OBS`, `LAND`, `CX/CY`, `CAP` |
| 모형 | 11 | `model (PINN)`, `pde_residual` |
| **학습** | **13** | 학습된 `model`, `D_phys`, `r_phys`, `hist` |
| 예측/검증 | 16 | `predict_year()`, `CORR23` |
| 확산모형 | 24 | `sir_I`, `sir_A` |

---

## 2. 실행 환경 / 방법

- **장치는 CPU 고정** (`DEVICE = torch.device("cpu")`). PINN은 PDE 잔차에서 **2계 미분**을 자동미분으로 계산하므로, 고계 미분 그래프의 수치 안정성을 위해 CPU를 의도적으로 사용한다.
- 재현성: `SEED = 0` 으로 `numpy`·`torch` 시드 고정.
- 좌표계: CSV의 `지역X/Y좌표`는 **EPSG:5186**(중부원점), 지도·작업 좌표계는 **EPSG:5179**. CSV 인코딩 **CP949**.
- 실행(콘솔 스크립트 shebang이 깨져 있으므로 모듈 형태로 실행):
  ```bash
  .venv/bin/python -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=kpp pine_wilt_KPP_PINN.ipynb
  ```

---

## 3. 데이터 파이프라인 (셀 1·3)

**셀 1** — import, 한글 폰트, 시드/장치, 경로·상수 정의.
- `YEARS_TRAIN = [2016..2022]` (완전관측 7년, 학습), `YEAR_VALID = 2023` (1~4월 부분관측, 검증), `YEARS_FCST = [2024,2025,2026]` (예측).

**셀 3** — 감염목 점 추출 (+ 캐시).
- `cache/infected_points_2016_2023.npz` 가 있으면 즉시 로드, 없으면 CSV에서 생성:
  1. 각 연도 CSV를 CP949로 읽어 `감염목구분 == "감염목"` 만 필터.
  2. `조사일자`에서 연도, `지역X/Y좌표` 추출, 유효성(`x,y>0`, NaN 제거) 마스킹.
  3. `Transformer(EPSG:5186 → EPSG:5179)` 로 좌표 재투영.
- 결과: `PX, PY`(미터 좌표), `PYR`(연도). 총 **3,183,376 건**.
- 2023은 1~4월 부분관측이라 **학습에서 제외**, 검증 참조로만 사용.

---

## 4. EDA (셀 5)

- 왼쪽: 연도별 감염목 **건수 막대그래프** (2023은 부분이라 색 구분).
- 오른쪽: `hexbin` **로그 밀도** 공간 분포 (전국, 2016–2023).
- 산출물 `01_eda_overview.png`. 총량은 2017년 정점(약 66만) 후 감소 추세(방제 효과).

---

## 5. 격자·관측밀도 $u_{obs}$ (셀 7·9)

**셀 7** — 경계·격자·관측밀도 생성.
- `skorea_provinces_2018.geojson`(시도 17개)을 EPSG:5179로 재투영 → 지도 배경 `KOREA_PLOT`, 통합 경계 `KOREA_UNION`.
- `shp_prepare(KOREA_UNION)` 로 공간 인덱스 준비 → **점-포함(육지) 판정 가속**. `land_mask()` 헬퍼.
- 도메인: 감염목 범위 + 여백 `PAD=15km`, **96×96 격자**. 셀 중심 `CX, CY`, 육지 마스크 `LAND`(육지 셀 약 3,609개).
- 연도별 밀도:
  - `year_grid(y)`: `np.histogram2d` 로 셀당 감염목 수 집계 후 **3×3 분리형 평활**(`smooth`).
  - **정규화 상수** `CAP = 학습연도 셀밀도의 99퍼센타일`(≈2896). 이상치에 강건한 수용능력 추정.
  - `U_OBS = clip(RAW/CAP, 0,1)·LAND` → **정규화 감염밀도** $u_{obs}\in[0,1]$, shape `(7, 96, 96)`.
  - `obs_density(y)`: 임의 연도(예: 2023)에 대해 같은 방식으로 정규화 밀도 반환.

**셀 9** — 시각화 도구.
- `CMAP`: 저밀도 투명→노랑→주황→빨강→진홍 커스텀 컬러맵.
- `plot_field(ax, gx, gy, U, ...)`: 지도 위에 밀도장을 매핑하는 **공용 함수**(이후 모든 지도 그림이 재사용). `02_observed_density_by_year.png` 생성.

---

## 6. 수학적 모형 — Fisher-KPP와 정규화 (셀 10)

지배방정식 (물리 좌표):

$$\frac{\partial u}{\partial t} \;=\; D\,\nabla^2 u \;+\; r\,u\,(1-u)$$

- $u(x,y,t)$: 정규화 감염밀도, $D$: 확산계수(매개충 확산), $r$: 국소 성장률, $u(1-u)$: 수용능력 포화항.

**입력 정규화** (수치 조건 개선): 중심 $x_c,y_c$, 등방 길이 스케일 $S=\tfrac12\max(\text{범위})$, 시간폭 $\Delta T = 2022-2016 = 6$.

$$X=\frac{x-x_c}{S},\quad Y=\frac{y-y_c}{S},\quad T=\frac{t-t_0}{\Delta T}$$

정규화 좌표에서 PDE는

$$u_T = \tilde D\,(u_{XX}+u_{YY}) + \tilde r\,u(1-u),\qquad
\boxed{\tilde D = D\frac{\Delta T}{S^2},\quad \tilde r = r\,\Delta T}$$

신경망은 **정규화 좌표**에서 학습하고 $\tilde D,\tilde r$ 를 추정한다. 물리 단위 환산(셀 13 말미):

$$D = \tilde D\,\frac{S_{km}^2}{\Delta T}\;[\text{km}^2/\text{년}],\qquad r = \frac{\tilde r}{\Delta T}\;[1/\text{년}],\qquad
\text{(Fisher 파면속도)}\;c \approx 2\sqrt{D\,r}$$

콜로케이션 시간 범위는 **예측 지평(2026)** 까지 → `tb = (0, nt(2026))`.

---

## 7. PINN 정의 (셀 11)

**데이터 텐서** (육지 셀 × 학습연도): 각 학습연도 $k$에 대해 육지 셀 좌표(정규화)와 시간, 관측밀도를 쌓아
`dX, dY, dT, dU` (각 `(25263, 1)`, = 3609 육지셀 × 7년) 생성.

**신경망** `PINN(nn.Module)`:
- 입력 $(X,Y,T)$ → **은닉 4층 × 너비 64**, 활성화 `tanh` → 출력 1, `sigmoid` → $u\in[0,1]$.
- 물리 파라미터를 **학습 변수**로:
  ```python
  self.raw_D = nn.Parameter(tensor(-1.0))   # D̃ = softplus(raw_D) ≈ 0.31 초기값
  self.raw_r = nn.Parameter(tensor(-0.5))   # r̃ = softplus(raw_r) ≈ 0.47 초기값
  ```
  `softplus`로 $\tilde D,\tilde r \ge 0$ 양수 보장. (이 `softplus`가 §8에서 순수 L-BFGS 붕괴 시 0 고정의 원인이 됨.)

**PDE 잔차** `pde_residual` — 자동미분으로 도함수 계산:

$$R = u_T - \tilde D\,(u_{XX}+u_{YY}) - \tilde r\,u(1-u)$$

```python
g = lambda o, i: torch.autograd.grad(o, i, torch.ones_like(o), create_graph=True)[0]
ux, uy, ut = g(u,x), g(u,y), g(u,t)
uxx, uyy   = g(ux,x), g(uy,y)        # create_graph=True → 2계 미분, θ에 대한 역전파 가능
return ut - model.D*(uxx+uyy) - model.r*u*(1-u)
```
`create_graph=True` 가 핵심: 잔차가 다시 파라미터 $\theta$ 에 대해 미분 가능해야 하므로 **고계 미분 그래프**를 유지한다.

또한 셀 11은 (a) **육지 내부 콜로케이션 풀**과 (b) **해안 경계점+법선**(폴리곤 컴포넌트별 추출 후 실제 해안만 필터·바깥정렬), (c) 해안 무흐름 잔차 `bc_residual(model,x,y,t,n_x,n_y)=∇u·n` 도 함께 정의한다 → §8.5.

---

## 8. 학습 — 2단계 최적화 Adam→L-BFGS (셀 13) ★

이 노트북의 핵심. **표준 PINN 레시피**인 *Adam 워밍업 → L-BFGS 정제* 2단계를 쓴다.

### 8.0 최소화 목적함수

$$L(\theta) = L_{data}(\theta) + w_{pde}\cdot L_{pde}(\theta) + w_{pde}\,w_{bc}\cdot L_{bc}(\theta)$$

$$L_{data}=\frac1{N_d}\sum_i w_i\big(u_\theta(x_i,y_i,t_i)-u_i^{obs}\big)^2,\quad w_i = 1+4\,u_i^{obs}\;(\text{고밀도 셀 가중})$$

$$L_{pde}=\frac1{N_c}\sum_{j\in\text{육지}} R_\theta(x_j,y_j,t_j)^2,\qquad
L_{bc}=\frac1{N_b}\sum_{k\in\text{해안}} \big(D\,\nabla u_\theta\cdot \hat n_k\big)^2$$

- $\theta$: **모든 파라미터** (망 가중치/바이어스 + `raw_D, raw_r`).
- 고밀도(감염) 셀에 4배까지 가중 → 청정지 편향 방지.
- **$L_{pde}$ 콜로케이션은 육지 내부로 한정**, **$L_{bc}$ 는 해안 무흐름(확산 플럭스 $D\nabla u\cdot\hat n=0$)** → §8.5.

### 8.1 1단계 — Adam 워밍업 (전역 탐색)

```python
ADAM_STEPS, ADAM_RAMP = 3000, 2500
opt   = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = StepLR(opt, step_size=2000, gamma=0.5)
```
- **미니배치**: 매 step 데이터 4096점, **육지 콜로케이션** 4096점, 해안 경계 2048점을 *새로 샘플*.
- **PDE 가중 커리큘럼**: $w_{pde} = 0.1 \to 0.3$ 로 2500 step에 걸쳐 선형 증가(육지 한정 콜로케이션은 물리가 강하므로 박스 시절 1.5보다 낮춤).
  $$w_{pde}(\text{step}) = 0.1 + 0.2\cdot\min\!\big(1,\ \text{step}/2500\big)$$
  → 초반엔 **데이터가 장(field)에 의미 있는 시·공간 구조를 형성**하게 하고, 점차 물리 잔차를 강화.
- 역할: random 초기화에서 좋은 분지(basin)로 이동 + $\tilde D,\tilde r$ 를 식별 가능한 영역으로.

### 8.2 2단계 — L-BFGS 정제 (2차 수렴)

```python
gen = torch.Generator(device=DEVICE).manual_seed(SEED)
cxL,cyL,ctL          = sample_interior(8192, gen)   # 고정 육지 콜로케이션
bxL,byL,btL,nxL,nyL  = sample_boundary(4096, gen)   # 고정 해안 콜로케이션(+법선)
optL = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50,
                         line_search_fn="strong_wolfe",
                         tolerance_grad=1e-10, tolerance_change=1e-12)
def closure():
    optL.zero_grad()
    loss = (w_full*(model(dX,dY,dT)-dU)**2).mean() \
         + 0.3*(pde_residual(model,cxL,cyL,ctL)**2).mean() \
         + 0.3*(model.D*bc_residual(model,bxL,byL,btL,nxL,nyL)).pow(2).mean()
    loss.backward(); return loss
for o in range(1500//20):     # 75회 .step() × 20 = 1500 iteration
    optL.step(closure)
```

L-BFGS가 **결정론적 목적함수**를 전제하므로 1단계와 달리:
- **전배치 데이터** (`dX,dY,dT` 전체 25,263점),
- **고정 콜로케이션** (육지·해안 점을 SEED로 한 번만 샘플),
- $w_{pde}=W_{MAX}=0.3$ **고정** (커리큘럼은 1단계에서 종료).

#### L-BFGS 수식 (준뉴턴 + 2-루프 재귀)

각 iteration $k$ ($g_k=\nabla L(\theta_k)$):

$$p_k = -B_k\,g_k,\qquad \theta_{k+1} = \theta_k + \alpha_k\,p_k$$

$B_k\approx[\nabla^2 L]^{-1}$ 는 역헤시안 근사. 파라미터 수 $n\approx 12{,}800$ 이라 $n\times n$ 저장 불가 → 최근 **$m=50$**(`history_size`) 개 쌍
$s_i=\theta_{i+1}-\theta_i,\ y_i=g_{i+1}-g_i$ 만 저장해 $B_k g_k$ 의 *작용*을 **2-루프 재귀**로 $O(mn)$ 계산:

```
q ← g_k
for i = k−1 … k−m:    a_i ← ρ_i (s_iᵀq);   q ← q − a_i y_i        # ρ_i = 1/(y_iᵀs_i)
γ_k ← (s_{k−1}ᵀy_{k−1})/(y_{k−1}ᵀy_{k−1});   z ← γ_k q            # 초기 스케일 H⁰=γ_k·I
for i = k−m … k−1:    b ← ρ_i (y_iᵀz);   z ← z + (a_i − b) s_i
p_k ← −z
```

**보폭 $\alpha_k$ — strong-Wolfe 라인서치** ($\phi(\alpha)=L(\theta_k+\alpha p_k)$, $c_1=10^{-4}, c_2=0.9$):

$$\phi(\alpha)\le\phi(0)+c_1\alpha\,g_k^\top p_k,\qquad |g(\theta_k+\alpha p_k)^\top p_k|\le c_2\,|g_k^\top p_k|$$

- 후보 $\alpha$ 마다 `closure()` 재호출 → 한 iteration이 closure를 여러 번 부른다.
- 그래서 **L이 호출 간 불변**이어야 함(전배치·고정 콜로케이션의 이유). 미니배치 재샘플 시 라인서치가 깨져 **발산**.
- 강곡률 조건이 $y_k^\top s_k>0$ 보장 → $B_k$ 양정부호 → $p_k$ 하강방향.

| 코드 인자 | 수식 역할 |
|---|---|
| `max_iter=20` | `.step()` 당 $k$-반복 수 (총 1500) |
| `history_size=50` | 이력 쌍 개수 $m$ |
| `line_search_fn="strong_wolfe"` | 위 보폭 규칙 |
| `lr=1.0` | 라인서치 초기 보폭 배율(준뉴턴 자연보폭=1) |
| `tolerance_grad=1e-10` | 종료: $\lVert g_k\rVert_\infty\le$ 이 값 |
| `tolerance_change=1e-12` | 종료: $|\Delta L|$ 또는 보폭 변화 $\le$ 이 값 |

### 8.3 왜 2단계인가 — 순수 L-BFGS 붕괴

random 초기화에서 **순수 L-BFGS**를 쓰면 $\tilde D,\tilde r$ 가 즉시 **0으로 붕괴**(물리 역추정 파괴, 검증 악화).
원인(측정으로 확인):
1. 의미 없는 초기 장에서 유연한 망 + 풀강도 물리로 결합 최적화 시, 잔차를 줄이는 *가장 쉬운 길*은
   $\tilde D,\tilde r\to0$ 으로 공간·반응항을 **통째로 제거**하고 $u$ 를 준정적으로 만드는 **퇴화 분지**.
2. `softplus` 고정: L-BFGS가 `raw_D`를 매우 작은 값(예: $-14$)으로 끌어내리면 기울기 $\sigma(\text{raw}_D)\approx 0$ → **0에 비가역 고정**.
3. Adam 워밍업(약한 물리 커리큘럼)은 *의미 있는 장*을 먼저 만들어 $\tilde D,\tilde r$ 을 식별 가능 상태로 유지 → 이 함정 회피.

옵티마이저 비교(모두 **박스 콜로케이션**, BC 없음 — 도메인/BC 개선은 §8.5):

| 방법 | RMSE | 학습 공간상관 | 2023 검증 | D |
|---|---|---|---|---|
| Adam 단독(6000) | 0.074 | 0.79 | 0.47 | 15.5* |
| 순수 L-BFGS | 0.062 | 0.83 | 0.40 | **0.0 붕괴** |
| **Adam→L-BFGS** | **0.035** | **0.93** | **0.62** | 0.6 |

\* Adam 단독의 $D{=}15.5$ 는 미수렴 아티팩트에 가깝다(L-BFGS가 충분히 수렴하면 $r{\approx}0$ 국면에서 $D$ 는 매우 작아짐).

### 8.4 물리 단위 환산 (셀 13 말미)
학습 후 `D_phys, r_phys, speed` 계산(§6 식). 다운스트림(SIR·요약)에서 재사용되는 전역 변수.

### 8.5 도메인 한정 + 해안 무흐름 경계조건 ★

위 옵티마이저 비교는 콜로케이션을 **사각형 전체**(바다 61% 포함)에 뿌렸다 — 바다 위 무의미한 PDE 강제 + 해안 간 바다확산 결합이라는 결함이 있었다. 이를 바로잡았다:

1. **콜로케이션 육지 한정** — 셀 11에서 육지 내부 점 풀(8만)과 해안 경계점(법선 포함, 6,977점)을 한 번 구축해 매 step 부분추출.
2. **해안 무흐름(Neumann) 경계** — 물리적 무흐름은 *확산 플럭스* $D\nabla u\cdot\hat n=0$. 손실에 `(model.D * bc_residual)²` 로 부과(셀 11 `bc_residual` = $\nabla u\cdot\hat n$).

**왜 flux형 $D\nabla u\cdot\hat n$ 인가(중요):** 학습된 $D\approx0$ 이라 flux는 법선미분과 무관하게 자동으로 ≈0 → **무흐름 BC가 거의 비활성**이고, 그게 *물리적으로 올바른* 결론(확산이 없으면 막을 플럭스도 없음). 큰 $D$ 는 BC 페널티가 커져 자기억제된다. 반면 법선미분 $\nabla u\cdot\hat n=0$ 을 **직접** 강제하면 `|∇u·n|`은 13배 줄지만(0.85→0.06) 남동 해안 핫스팟과 충돌해 검증이 하락(0.62→0.585)하므로 채택하지 않았다.

도메인/BC 적용 후(채택 최종): RMSE 0.040, 학습 공간상관 평균 **0.92**, **2023 검증 0.61**, $D\approx3.9\ \text{km}^2/\text{년}$ — 박스 대비 적합은 동등하게 유지하면서 **물리적으로 올바른 도메인·경계**를 갖춘다.

> 발견: 박스의 좋은 적합은 부분적으로 *약한 물리*(바다 희석)의 부산물이었다. 물리를 제대로 강제하면 Fisher-KPP(특히 $r\approx0$)가 해안 집중·쇠퇴 패턴을 완벽히 설명하진 못함이 드러난다.

---

## 9. 손실·파라미터 수렴 시각화 (셀 14)

- 왼쪽: `total/data/pde/bc` 손실(반로그). 오른쪽: 정규화 $\tilde D,\tilde r$ 수렴.
- `hist["x"]` 는 누적 iteration(Adam 0–3000 + L-BFGS 3000–4500), **Adam→L-BFGS 핸드오프**를 점선(`axvline`)으로 표시.
- `03_training_loss.png`. 핸드오프 직후 손실이 한 자릿수 급락 → 분지가 좋았고 L-BFGS가 제대로 수렴했다는 방증.

---

## 10. 예측·검증 (셀 16)

- **고해상도 160×160 격자**에서 신경망 평가. `predict_year(y)`: 해당 연도 $u$ 장을 반환(육지 외 NaN).
- `model_on_coarse(y)`: 96격자 육지셀 예측값(지표 계산용).
- **지표**:
  - 학습연도(2016–2022): 셀별 **RMSE** 와 **공간상관**(관측 vs 재현).
  - **2023 검증**: `CORR23 = corr(예측, 2023부분관측)` — 2023은 1~4월 부분이라 절대수준이 아닌 **공간패턴** 비교.
- 현재 결과: 학습 공간상관 **0.82–0.97(평균 0.92)**, **2023 r = 0.61**.

---

## 11. 결과 시각화 (셀 18–22)

모두 `plot_field`(셀 9)를 재사용해 지도에 매핑.

| 셀 | 그림 | 내용 |
|---|---|---|
| 18 | `04_pinn_fit_vs_observed.png` | 선택 연도(2017/2019/2022) 관측 vs PINN 재현 |
| 19 | `05_validation_2023.png` | 2023 실제(부분) vs PINN, 제목에 `CORR23` |
| 20 | `06_forecast_future.png` | 2024–2026 예측 피해분포 |
| 21 | `07_spread_timeline.png` | 2016→2026 타임라인(재현/검증/예측) |
| 22 | `08_hotspot_change.png` | $\Delta u = $ 예측(2026) − 관측(2022), 발산형 컬러맵 |

---

## 12. 소진 고려 공간 SIR (셀 23–25)

기본 Fisher-KPP는 포화지가 *계속 감염*으로 남지만, 실제론 감염목이 고사·제거되면 그 자리의
**감수성 기주(소나무)가 사라져 재감염되지 않고** 전선이 인접지로 이동한다. 이를 공간 **SIR**로 확장:

$$\frac{\partial I}{\partial t}=D\nabla^2 I+\beta SI-\gamma I,\quad
\frac{\partial S}{\partial t}=-\beta SI,\quad
\frac{\partial R}{\partial t}=\gamma I$$

**셀 24 — 수치적분** (`run_sir`):
- `lap_masked`: **육지 한정 라플라시안**(해안=무흐름 경계, `np.roll` 5점 스텐실).
- 감수성 기주 `HOST`: 과거(2016–2022) 최대 발생범위를 평활·정규화해 근사.
- 파라미터: **$D$ 는 PINN이 학습한 값(`D_phys`)**, $\beta=1.2,\ \gamma=0.5$ 는 가정($R_0=\beta/\gamma\approx2.4$, 평균 감염기간 ~2년).
- 초기조건: $I_0 = $ 2022 관측, $S_0 = \mathrm{clip}(HOST-I_0)$. **명시적 오일러** `dt=0.1`년으로 2022→2030 적분.
- 출력: `sir_I[y]`(활성 감염), `sir_A[y] = clip(HOST−S)`(누적 피해, 소진 포함).

> $D$ 의 역할: 확산항 $D\nabla^2 I$ 가 전선을 인접 미감염지로 퍼뜨린다. 주 동력은 반응항 $\beta SI$ 이며,
> $D$ 가 작으면(현재 ≈3.9) 확산은 보수적이지만 전선 전파는 유지된다(퇴화하지 않음).

**셀 25** — 2022 초기 / 2026·2030 활성 감염 / 2030 누적 피해 4분할. `09_sir_forecast_2030.png`.

---

## 13. 시도별·추이 분석 (셀 27–28)

**셀 27 — 시도별 순위**:
- 96격자 육지셀을 `gpd.sjoin`(`within`)으로 17개 시도에 공간 배정(`PROV`).
- `prov_sum`: 셀 값을 시도별 합산 → 2022 관측 vs 2026 예측 상위 10개 시도 막대그래프. `10_province_damage.png`.

**셀 28 — 피해 추이 곡선** (전국 평균 밀도, 2016–2030):
- 관측(2016–2022) → 2023 부분관측 → PINN 예측(2024–2026) → SIR 활성/누적(~2030)을 한 장에. `11_damage_trajectory.png`.

---

## 14. 요약 출력 (셀 30)

데이터 건수, 학습/검증/예측 구성, 학습된 $D,r$, 전파속도, 저장된 11개 이미지 목록을 콘솔에 출력.

---

## 15. 주요 결과·한계

**결과 (현재: 2단계 최적화 + 육지 콜로케이션 + 무흐름 BC):**
- 학습연도 공간상관 평균 **0.92**(0.82–0.97), **2023 검증 r ≈ 0.61**.
- 학습된 $D \approx 3.9\ \text{km}^2/\text{년}$, $r \approx 0.002\ /\text{년}$ → 파면속도 $2\sqrt{Dr}\approx0.2$.
- 피해는 남동부(경북>경남>울산) 집중 + 수도권·제주 부차 발생지.

**한계 / 주의:**
- **식별성**: $r\approx0$(쇠퇴 국면)이면 Fisher 파면속도가 $D$ 에 거의 무관($2\sqrt{Dr}\approx0$)해져 **$D$ 가 약하게만 식별**된다. 학습된 $D$ 절대값은 신뢰구간이 넓고, 무흐름 BC도 거의 비활성이 된다(§8.5).
- **표본편향**: 데이터는 조사·방제 정책에 의존하며, 2017년 이후 총량 감소는 방제 효과를 포함 → **정량 수치보다 공간 추세** 해석 권장.
- **SIR의 $\beta,\gamma$ 는 가정값**이고 장기 외삽이므로 2030 결과는 단기 시사점으로 해석.
- 2023은 부분관측이라 검증은 절대수준이 아닌 **공간패턴** 기준.

---

### 부록 — 핵심 전역 변수
| 변수 | 의미 |
|---|---|
| `PX, PY, PYR` | 감염목 점 좌표(m)·연도 |
| `CX, CY, LAND` | 96격자 중심·육지 마스크 |
| `U_OBS` | 정규화 관측밀도 `(7,96,96)` |
| `CAP` | 정규화 상수(99퍼센타일 셀밀도) |
| `nx, ny, nt` | 좌표/시간 정규화 함수 |
| `dX,dY,dT,dU` | 학습 데이터 텐서(육지셀×7년) |
| `model` | 학습된 PINN(+ `D,r`) |
| `D_phys, r_phys, speed` | 물리단위 환산값 |
| `predict_year`, `model_on_coarse` | 예측 헬퍼 |
| `sir_I, sir_A` | SIR 활성/누적 결과 |
