# K리그 최종 패스 좌표 예측 AI 모델

K리그 경기 내 플레이 시퀀스의 마지막 패스 도착 좌표(X, Y)를 예측하는 AI 모델입니다.

## 프로젝트 구조

```
.
├── config.py              # 설정 파일
├── data_preprocessing.py  # 데이터 전처리 및 특징 엔지니어링
├── model.py              # 모델 정의 (LSTM 기반)
├── train.py              # 모델 학습 스크립트
├── predict.py            # 예측 및 제출 파일 생성
├── requirements.txt       # 필요한 패키지 목록
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

### 2. 예측 및 제출 파일 생성

```bash
python predict.py
```

- `test/` 폴더의 테스트 에피소드들을 예측합니다
- 결과는 `submission.csv`로 저장됩니다

## 모델 구조

- **기본 아키텍처**: LSTM (Long Short-Term Memory)
- **입력**: 시계열 액션 데이터 (시퀀스 길이: 50)
- **출력**: 마지막 패스의 도착 좌표 (end_x, end_y)

### 특징 (Features)

- 좌표 정보: start_x, start_y, end_x, end_y
- 시간 정보: time_seconds, time_diff
- 액션 정보: action_type, result
- 계산된 특징: dx, dy, distance, angle
- 팀 정보: is_home

## 하이퍼파라미터

설정은 `config.py`에서 변경할 수 있습니다:

- `SEQUENCE_LENGTH`: 50 (시퀀스 길이)
- `HIDDEN_DIM`: 128 (LSTM hidden dimension)
- `NUM_LAYERS`: 2 (LSTM 레이어 수)
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 0.001
- `NUM_EPOCHS`: 20

## 데이터 규칙 준수

- ✅ 각 game_id-episode 단위로 독립적으로 예측
- ✅ 제공된 데이터만 사용 (외부 데이터 사용 안 함)
- ✅ 로컬에서 실행 가능한 모델 (API 사용 안 함)
- ✅ 오픈소스 라이선스 모델만 사용

## 주의사항

- GPU가 있으면 자동으로 사용하며, 없으면 CPU를 사용합니다
- 학습 시간은 데이터 크기와 하드웨어에 따라 다릅니다
- 예측 시 모델 파일(`models/best_model.pth`)이 필요합니다

