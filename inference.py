"""
추론(Inference) 코드 - 테스트 데이터 예측 및 제출 파일 생성
대회 규칙: 추론 코드는 반드시 별도의 코드 파일로 작성해야 함
"""
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import random

from model import PassPredictor
from data_preprocessing import load_test_data, load_test_episodes, create_features
from config import *

# 재현 가능성을 위한 시드 고정
def set_seed(seed=42):
    """재현 가능성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inference():
    """
    테스트 데이터 예측 및 제출 파일 생성
    
    대회 규칙 준수:
    - 각 game_id-episode 단위로 독립적으로 예측
    - 해당 에피소드 내부의 시퀀스 데이터만 사용
    - 다른 에피소드의 데이터 활용 금지
    """
    # 재현 가능성을 위한 시드 고정
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 로드
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model not found: {checkpoint_path}. Please train the model first using train.py"
        )
    
    # PyTorch 2.6+ 호환성: StandardScaler를 포함한 checkpoint 로드를 위해 weights_only=False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_type_map = checkpoint['action_type_map']
    scaler = checkpoint.get('scaler', None)  # scaler가 없을 수도 있음 (이전 모델 호환)
    
    # 테스트 데이터 로드
    print("\n=== Loading Test Data ===")
    test_df = load_test_data()
    
    # 테스트 에피소드 로드 및 전처리
    # 각 에피소드는 독립적으로 처리됨 (Data Leakage 방지)
    print("\n=== Processing Test Episodes ===")
    sequences, episode_ids = load_test_episodes(test_df, action_type_map, scaler)
    
    # 모델 초기화
    input_dim = sequences.shape[2]
    print(f"Input dimension: {input_dim}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Sequences stats: min={sequences.min():.4f}, max={sequences.max():.4f}, mean={sequences.mean():.4f}, NaN count={np.isnan(sequences).sum()}")
    
    model = PassPredictor(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 예측
    print("\n=== Making Predictions ===")
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i+batch_size]
            batch_sequences = torch.FloatTensor(batch_sequences).to(device)
            
            # 입력 확인
            if i == 0:
                print(f"First batch shape: {batch_sequences.shape}")
                print(f"First batch stats: min={batch_sequences.min():.4f}, max={batch_sequences.max():.4f}, mean={batch_sequences.mean():.4f}")
                print(f"First batch NaN count: {torch.isnan(batch_sequences).sum().item()}")
            
            outputs = model(batch_sequences)
            
            # 출력 확인
            if i == 0:
                print(f"First output shape: {outputs.shape}")
                print(f"First output: {outputs[0].cpu().numpy()}")
                print(f"Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
                print(f"Output NaN count: {torch.isnan(outputs).sum().item()}")
            
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    
    # 모델 출력 확인
    print(f"\n=== Model Output Statistics ===")
    print(f"Predictions shape: {predictions.shape}")
    print(f"end_x range: [{predictions[:, 0].min():.4f}, {predictions[:, 0].max():.4f}], mean: {predictions[:, 0].mean():.4f}")
    print(f"end_y range: [{predictions[:, 1].min():.4f}, {predictions[:, 1].max():.4f}], mean: {predictions[:, 1].mean():.4f}")
    print(f"Unique end_x values: {len(np.unique(predictions[:, 0]))}")
    print(f"Unique end_y values: {len(np.unique(predictions[:, 1]))}")
    
    # 제출 파일 생성
    print("\n=== Creating Submission File ===")
    submission = pd.DataFrame({
        'game_episode': episode_ids,
        'end_x': predictions[:, 0],
        'end_y': predictions[:, 1]
    })
    
    # 좌표 범위 제한 (0 ~ FIELD_LENGTH, 0 ~ FIELD_WIDTH)
    submission['end_x'] = submission['end_x'].clip(0, FIELD_LENGTH)
    submission['end_y'] = submission['end_y'].clip(0, FIELD_WIDTH)
    
    # sample_submission.csv와 동일한 순서로 정렬
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV, encoding='utf-8')
    
    # merge를 사용하여 안전하게 정렬 (없는 에피소드는 기본값으로 채움)
    submission = sample_submission[['game_episode']].merge(
        submission, 
        on='game_episode', 
        how='left'
    )
    
    # NaN 값 처리 (예측되지 않은 에피소드의 경우 필드 중앙값 사용)
    submission['end_x'] = submission['end_x'].fillna(FIELD_LENGTH / 2)
    submission['end_y'] = submission['end_y'].fillna(FIELD_WIDTH / 2)
    
    # 저장 (UTF-8 인코딩, 대회 규칙 준수)
    output_path = 'submission.csv'
    submission.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Submission file saved: {output_path}")
    print(f"\nPredictions summary:")
    print(f"  Total episodes: {len(submission)}")
    print(f"  end_x range: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
    print(f"  end_y range: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")
    
    return submission

if __name__ == "__main__":
    inference()

