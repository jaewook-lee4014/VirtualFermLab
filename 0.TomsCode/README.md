# VirtualFermLab

폐기물-단백질 전환 발효 공정의 모델링, 시뮬레이션, 최적화를 위한 연구 플랫폼.

실험 데이터 기반 파라미터 추정부터 논문 자동 탐색(LLM 기반 파라미터 추출), Monte Carlo 불확실성 분석,
가상 실험(DOE)까지 하나의 통합 파이프라인으로 제공합니다.

---

## 프로젝트 구조

```
VirtualFermLab/
├── src/virtualfermlab/          # Python 패키지 (pip install -e .)
│   ├── models/                  # ODE 기반 성장 모델
│   │   ├── kinetics.py          #   Monod, Contois 성장 속도식
│   │   ├── enzyme_regulation.py #   효소 유도/Kompala cybernetic 모델
│   │   ├── ph_model.py          #   Cardinal pH 모델
│   │   ├── ode_systems.py       #   통합 ODE 시스템 (ModelConfig)
│   │   └── analytical.py        #   해석적 모델 (exponential cap 등)
│   │
│   ├── simulator/               # ODE 적분 엔진
│   │   ├── integrator.py        #   simulate() — odeint/solve_ivp 래퍼
│   │   └── steady_state.py      #   정상상태 분석
│   │
│   ├── fitting/                 # 파라미터 추정
│   │   ├── calibrate.py         #   Calibration 루프
│   │   ├── validate.py          #   Validation 루프
│   │   ├── objectives.py        #   목적함수 정의
│   │   ├── metrics.py           #   MAE, MSE, BIC, R2 등
│   │   ├── bounds.py            #   파라미터 범위
│   │   ├── fitters.py           #   least_squares / differential_evolution
│   │   └── growth_rate.py       #   모델-프리 성장률 추정
│   │
│   ├── parameters/              # 균주 파라미터 관리
│   │   ├── schema.py            #   StrainProfile, SubstrateParams (Pydantic)
│   │   ├── library.py           #   YAML 기반 균주 라이브러리 로더
│   │   ├── distributions.py     #   분포 기반 파라미터 샘플링
│   │   └── defaults/            #   내장 균주 프로파일 (YAML)
│   │       └── F_venenatum_A35.yaml
│   │
│   ├── discovery/               # 자동 논문 탐색 + LLM 파라미터 추출
│   │   ├── pipeline.py          #   5단계 discovery 오케스트레이터
│   │   ├── paper_search.py      #   PubMed/Semantic Scholar 크롤링 (병렬)
│   │   ├── llm_extraction.py    #   vLLM 기반 파라미터 추출
│   │   ├── db.py                #   SQLite 저장층 (thread-safe)
│   │   ├── name_resolver.py     #   균주명 정규화 (약어 → 전체명)
│   │   ├── taxonomy.py          #   NCBI Taxonomy 기반 유사 균주 매칭
│   │   └── prompts.py           #   LLM 프롬프트 템플릿
│   │
│   ├── experiments/             # 가상 실험 설계
│   │   ├── doe.py               #   Latin Hypercube DOE 조건 생성
│   │   ├── monte_carlo.py       #   Monte Carlo 시뮬레이션
│   │   └── analysis.py          #   Ranking, Heatmap, Pareto front
│   │
│   ├── data/                    # 데이터 전처리
│   │   ├── loaders.py           #   CSV 로더
│   │   ├── transforms.py        #   OD600 → 바이오매스 변환
│   │   └── resampling.py        #   Bootstrap/Jackknife 리샘플링
│   │
│   ├── viz/                     # 시각화 (matplotlib)
│   │   ├── trajectories.py      #   시계열 궤적 플롯
│   │   ├── heatmaps.py          #   DOE 히트맵
│   │   └── ph_analysis.py       #   pH 분석 플롯
│   │
│   ├── io/                      # 입출력
│   │   └── export.py            #   CSV/JSON 결과 내보내기
│   │
│   └── web/                     # Flask 웹 UI
│       ├── app.py               #   Flask 앱 + 모든 API 엔드포인트
│       ├── plotly_charts.py     #   Plotly 차트 생성
│       ├── templates/           #   HTML 템플릿 (v1/v2 테마)
│       └── static/              #   CSS, JavaScript
│
├── tests/                       # pytest 테스트 스위트
│   ├── test_models/
│   ├── test_simulator/
│   ├── test_fitting/
│   ├── test_parameters/
│   ├── test_experiments/
│   └── test_discovery/
│
├── scripts/                     # 서버 실행 스크립트
│   ├── start_vllm.sh            #   vLLM 모델 서버 시작
│   ├── start_ui.sh              #   Flask UI 서버 시작
│   └── switch_ui.sh             #   UI 테마 전환 (v1/v2)
│
├── 0.TomsCode/                  # 베이스라인 노트북 (원본)
│   ├── 0.Data/                  #   실험 CSV 데이터
│   └── 1.Code/                  #   Jupyter 노트북
│
└── pyproject.toml               # 패키지 설정
```

---

## 설치 및 실행

### 설치

```bash
pip install -e ".[web,dev]"
```

### 테스트

```bash
python -m pytest tests/ -v
```

### 서버 실행

이 프로젝트는 **2개의 독립 프로세스**로 구성됩니다:

```
┌─────────────────────────────────────────────────────────┐
│                    Flask UI (:51665)                     │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐ │
│  │Simulation│ │Monte     │ │Virtual    │ │Discovery  │ │
│  │API       │ │Carlo API │ │Experiment │ │Pipeline   │ │
│  └──────────┘ └──────────┘ └───────────┘ └─────┬─────┘ │
│                                                │       │
│       자체 처리 (CPU)                           │       │
└────────────────────────────────────────────────┼───────┘
                                                 │ HTTP
                 ┌───────────────────────────────┼───────┐
                 │         외부 API 호출          │       │
                 │                               ▼       │
                 │  ┌──────────┐  ┌──────────────────┐   │
                 │  │PubMed    │  │vLLM Server (:8000)│   │
                 │  │Semantic  │  │Qwen2.5-32B       │   │
                 │  │Scholar   │  │(GPU 노드)         │   │
                 │  └──────────┘  └──────────────────┘   │
                 └───────────────────────────────────────┘
```

#### 1. vLLM 모델 서버 (GPU 노드)

```bash
srun --jobid=<JOB_ID> --overlap bash scripts/start_vllm.sh
```

- Qwen/Qwen2.5-32B-Instruct 모델을 OpenAI-compatible API로 서빙
- 포트: `8000` (환경변수 `VLLM_PORT`로 변경 가능)
- Discovery pipeline의 LLM 파라미터 추출 단계에서만 사용
- 이 서버가 없어도 나머지 기능(시뮬레이션, MC, DOE)은 정상 동작

#### 2. Flask UI 서버 (CPU 노드)

```bash
srun --jobid=<JOB_ID> --overlap bash scripts/start_ui.sh
```

- 포트: `51665` (환경변수 `UI_PORT`로 변경 가능)
- vLLM 서버 주소: 환경변수 `VLLM_BASE_URL` (기본값 `http://localhost:8000/v1`)

#### UI 테마 전환

```bash
./scripts/switch_ui.sh v1   # Bootstrap dark 테마
./scripts/switch_ui.sh v2   # Preprint-v1 gradient 테마
```

---

## Web API 엔드포인트

### 균주 정보

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/strains` | 사용 가능한 균주 목록 |
| GET | `/api/strain/<name>` | 균주 프로파일 상세 조회 |

### 시뮬레이션

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/simulate` | 단일 ODE 시뮬레이션 실행 |

### Monte Carlo

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/monte-carlo/start` | MC 시뮬레이션 시작 (비동기) |
| GET | `/api/monte-carlo/status/<id>` | 진행 상태 폴링 |
| GET | `/api/monte-carlo/result/<id>` | 결과 + Plotly 차트 |

### Virtual Experiment (DOE + MC)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/virtual-experiment/start` | DOE 조건 생성 + MC 실행 (비동기) |
| GET | `/api/virtual-experiment/status/<id>` | 진행 상태 폴링 |
| GET | `/api/virtual-experiment/result/<id>` | Ranking, Heatmap, Pareto |

### Strain Discovery

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/discovery/start` | 자동 탐색 파이프라인 시작 (비동기) |
| GET | `/api/discovery/status/<id>` | 5단계 진행 상태 |
| GET | `/api/discovery/result/<id>` | 탐색 결과 + StrainProfile |

### 결과 내보내기

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/export/csv/<id>` | 결과를 CSV로 다운로드 |
| GET | `/api/export/json/<id>` | 결과를 JSON으로 다운로드 |

---

## Discovery Pipeline

미지의 균주에 대해 **논문 자동 탐색 → LLM 파라미터 추출 → 유사 균주 매칭 → StrainProfile 자동 생성**을 수행하는 5단계 파이프라인:

```
Stage 1: Paper Search
  PubMed + Semantic Scholar API 병렬 크롤링 (ThreadPoolExecutor)
  DOI 기반 중복 제거 후 Queue에 push (producer)
      │
      ▼ (producer-consumer 패턴, 논문이 도착하는 대로 즉시 처리)
Stage 2: LLM Extraction
  Queue에서 논문을 꺼내 vLLM으로 파라미터 추출 (consumer)
  μ_max, Ks, Yxs, K_I, pH_opt/min/max, lag_time
  값 범위 검증 후 SQLite 저장
      │
Stage 3: DB Storage
  papers + extracted_params 테이블에 영속화
      │
Stage 4: Taxonomy Match
  NCBI Taxonomy lineage 기반으로 기존 라이브러리 균주와 유사도 계산
      │
Stage 5: Build StrainProfile
  우선순위: 문헌 추출값 > 유사 균주 값 > 생물학적 기본값
  Pydantic StrainProfile 객체 생성 + 캐싱
```

---

## ODE 성장 모델

### 지원 모델

| 성장 모델 | 효소 유도 모드 | 핵심 수식 |
|-----------|---------------|-----------|
| **Monod** | 직접 억제 (direct) | `μ = μ_max · S/(S+K_s)`, 자일로스에 `1/(1+S1/K_I)` 적용 |
| **Contois** | 직접 억제 (direct) | `μ = μ_max · S/(S+K_s·X)` |
| **Monod** | 효소 유도 (enzyme) | 효소 Z가 별도 ODE, `enzyme_factor = Z/(K_Z_S+Z)` |
| **Contois** | 효소 유도 (enzyme) | 위 + Contois 성장식 |
| **Monod** | Kompala cybernetic (kompala) | 2개 효소 Z1,Z2 + matching/proportional law |

### 상태변수

- **Direct**: `[X, S1, S2, totalOutput]`
- **Enzyme**: `[X, S1, S2, Z, totalOutput]`
- **Kompala**: `[X, S1, S2, Z1, Z2, totalOutput]`

### 추가 기능

- **Cardinal pH**: pH_min, pH_opt, pH_max 기반 성장률 보정
- **Lag phase**: 지연 시간 모델링
- **연속 발효**: 희석률(D), 유입 기질 농도(S_in) 지원

---

## 파라미터 체계

`StrainProfile` (Pydantic 모델)을 통해 모든 파라미터를 구조화:

```yaml
# 예시: parameters/defaults/F_venenatum_A35.yaml
name: F_venenatum_A35
substrates:
  glucose:
    mu_max: {type: normal, value: 0.12, std: 0.02, confidence: A}
    Ks:     {type: fixed, value: 0.18, confidence: A}
    Yxs:    {type: uniform, low: 0.28, high: 0.36, confidence: B}
  xylose:
    mu_max: {type: normal, value: 0.06, std: 0.01, confidence: B}
    ...
cardinal_pH:
  pH_min: {type: fixed, value: 3.5}
  pH_opt: {type: fixed, value: 6.0}
  pH_max: {type: fixed, value: 7.5}
```

각 파라미터는 `DistributionSpec`으로 표현되어 Monte Carlo 샘플링에 직접 사용됩니다:
- `fixed`: 고정값
- `normal`: 정규분포 (`value`, `std`)
- `uniform`: 균등분포 (`low`, `high`)
- `confidence`: 데이터 신뢰도 (A: 직접 측정, B: 문헌, C: 추정/기본값)

---

## 베이스라인 노트북

`0.TomsCode/` 폴더에 원본 실험 분석 노트북이 있습니다.

| 노트북 | 내용 |
|--------|------|
| `ESCAPE25PEwRandomSampling.ipynb` | 이중기질(glucose+xylose) 발효 모델 선택, 파라미터 추정, 불확실성 분석 |
| `MPRpHExp18112025Analysis.ipynb` | pH 4.0~6.5 범위 성장 kinetics 분석, Monod vs exponential cap 비교 |

### CSV 데이터

| 파일 | 설명 |
|------|------|
| `1-to-1-GlucoseXyloseMicroplateGrowth.csv` | Glucose:Xylose 1:1 calibration 데이터 |
| `2-to-1-GlucoseXyloseMicroplateGrowth.csv` | Glucose:Xylose 2:1 validation 데이터 |
| `MPR18112025CSV - Sheet1.csv` | 6개 pH 조건(4.0~6.5) 성장 데이터 |

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| numpy, scipy | ODE 적분, 최적화 |
| pandas | 데이터 처리 |
| pydantic | 파라미터 스키마 검증 |
| matplotlib | 시각화 (노트북/내보내기) |
| requests | 외부 API 호출 (PubMed, S2, vLLM) |
| flask | 웹 UI 서버 |
| plotly | 인터랙티브 웹 차트 |
| pyyaml | 균주 프로파일 로딩 |
