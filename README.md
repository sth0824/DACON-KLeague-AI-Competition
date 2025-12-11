# K리그 최종 패스 좌표 예측 AI 모델

K리그 경기 내 플레이 시퀀스의 마지막 패스 도착 좌표(X, Y)를 예측하는 AI 모델입니다.

## 프로젝트 구조

```
.
├── config.py              # 설정 파일
├── data_preprocessing.py  # 데이터 전처리 및 특징 엔지니어링
├── model.py              # 모델 정의 (LSTM 기반)
├── train.py              # 모델 학습 스크립트
├── inference.py           # 추론 코드 (대회 규칙: 별도 파일로 분리)
├── requirements.txt       # 필요한 패키지 목록 (버전 명시)
├── README.md             # 프로젝트 설명
└── models/               # 학습된 모델 저장 디렉토리
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 모델 학습

```bash
python train.py
```

- `train.csv`를 사용하여 모델을 학습합니다
- 학습된 모델은 `models/best_model.pth`에 저장됩니다
- Early stopping을 사용하여 과적합을 방지합니다

### 2. 예측 및 제출 파일 생성 (추론)

```bash
python inference.py
```

- `test/` 폴더의 테스트 에피소드들을 예측합니다
- 결과는 `submission.csv`로 저장됩니다 (UTF-8 인코딩)
- **대회 규칙**: 추론 코드는 별도 파일(`inference.py`)로 분리

## 모델 구조

- **기본 아키텍처**: LSTM + GRU 앙상블 (양방향)
- **입력**: 시계열 액션 데이터 (시퀀스 길이: 50, 특징 26개)
- **출력**: 마지막 패스의 도착 좌표 (end_x, end_y)

### 특징 (Features) - 26개

- 좌표 정보: start_x, start_y, end_x, end_y
- 정규화 좌표: start_x_norm, start_y_norm, end_x_norm, end_y_norm
- 시간 정보: time_seconds, time_diff
- 액션 정보: action_type, result
- 계산된 특징: dx, dy, distance, angle, velocity
- 패턴 특징: forward_pass, pass_length_category
- 누적 통계: cumsum_distance, cumsum_forward
- 이동 평균: ma3_dx, ma3_dy
- 필드 구역: zone_x, zone_y
- 팀 정보: is_home

## 하이퍼파라미터

설정은 `config.py`에서 변경할 수 있습니다:

- `SEQUENCE_LENGTH`: 50 (시퀀스 길이)
- `HIDDEN_DIM`: 512 (LSTM/GRU hidden dimension)
- `NUM_LAYERS`: 4 (레이어 수)
- `BATCH_SIZE`: 64
- `LEARNING_RATE`: 0.0003
- `NUM_EPOCHS`: 40
- 손실 함수: SmoothL1Loss (Huber Loss)
- 옵티마이저: AdamW

## 평가 방식

- **평가 산식**: 유클리드 거리 (Euclidean Distance)
- **Public Score**: 전체 테스트 데이터 중 사전 샘플링된 30%
- **Private Score**: 전체 테스트 데이터 100%
- **선발 방식**: Private Score 상위 20팀 선발 → 코드 검증 → 최종 상위 15팀 수상

학습 시 유클리드 거리를 함께 모니터링하여 실제 평가 지표를 추적합니다.

## 대회 규칙 준수

### 데이터 규칙
- ✅ 각 game_id-episode 단위로 독립적으로 예측
- ✅ 제공된 데이터만 사용 (외부 데이터 사용 안 함)
- ✅ 로컬에서 실행 가능한 모델 (API 사용 안 함)
- ✅ 오픈소스 라이선스 모델만 사용 (2025.11.23 이전 공개, MIT/Apache 2.0 등)
- ✅ Data Leakage 방지: 각 에피소드는 독립적으로 처리

### 코드 제출 규칙
- ✅ 데이터 입/출력 경로: 상대 경로 사용
- ✅ 코드 인코딩: UTF-8
- ✅ 라이브러리 버전: `requirements.txt`에 명시
- ✅ 학습/추론 코드 분리: `train.py` / `inference.py`
- ✅ 재현 가능성: 시드 고정 (seed=42)
- ✅ CSV 인코딩: UTF-8

## 환경 설정 및 재현 가능성

### 환경 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (선택사항, CPU도 가능)
- 모든 패키지는 `requirements.txt`에 명시된 버전 사용

### 재현 가능성 보장
- 시드 고정: `train.py`와 `inference.py` 모두 시드 42로 고정
- 결정적 연산: `torch.backends.cudnn.deterministic = True`
- 동일한 환경에서 동일한 결과 보장

### 실행 순서
1. 환경 설정: `pip install -r requirements.txt`
2. 모델 학습: `python train.py` → `models/best_model.pth` 생성
3. 추론 실행: `python inference.py` → `submission.csv` 생성

## 주의사항

- GPU가 있으면 자동으로 사용하며, 없으면 CPU를 사용합니다
- 학습 시간은 데이터 크기와 하드웨어에 따라 다릅니다
- 추론 시 모델 파일(`models/best_model.pth`)이 필요합니다
- 코드 검증을 위해 재현 가능성을 보장합니다 (시드 고정)
- 학습 중 유클리드 거리(Euclidean Distance)를 함께 출력하여 실제 평가 지표를 모니터링합니다
- 모든 CSV 파일은 UTF-8 인코딩으로 처리됩니다

