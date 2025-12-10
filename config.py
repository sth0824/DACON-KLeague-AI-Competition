"""
설정 파일
"""
import os

# 데이터 경로
DATA_DIR = "."
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
MATCH_INFO_CSV = os.path.join(DATA_DIR, "match_info.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 모델 설정
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 하이퍼파라미터 (대폭 개선 설정)
SEQUENCE_LENGTH = 50  # 시퀀스 길이 (패딩/트렁케이션)
HIDDEN_DIM = 512  # 대폭 증가: 더 큰 표현력
NUM_LAYERS = 4  # 더 깊은 네트워크
DROPOUT = 0.4  # 과적합 방지 강화
LEARNING_RATE = 0.0003  # 더 작은 학습률
BATCH_SIZE = 64  # 배치 크기 증가 (더 안정적)
NUM_EPOCHS = 40  # 1시간 내 완료를 위해 조정
EARLY_STOPPING_PATIENCE = 7  # 적절한 기회

# 특징 설정
FEATURE_COLS = [
    'start_x', 'start_y', 'end_x', 'end_y',
    'time_seconds', 'is_home',
    'action_type_encoded', 'result_encoded',
    'dx', 'dy', 'distance', 'angle',
    'time_diff'
]

# 좌표 범위 (FIFA 표준: 105 x 68)
FIELD_LENGTH = 105
FIELD_WIDTH = 68

