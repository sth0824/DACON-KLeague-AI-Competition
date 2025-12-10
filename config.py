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

# 하이퍼파라미터
SEQUENCE_LENGTH = 50  # 시퀀스 길이 (패딩/트렁케이션)
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

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

