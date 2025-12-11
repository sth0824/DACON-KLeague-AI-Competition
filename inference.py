"""
개선된 추론 코드
"""
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import random

from model import ImprovedPassPredictor
from data_preprocessing import load_test_data, load_test_episodes
from config import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inference():
    """개선된 모델로 추론"""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 로드
    checkpoint_path = os.path.join(MODEL_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_type_map = checkpoint['action_type_map']
    scaler = checkpoint.get('scaler', None)
    
    # 테스트 데이터 로드
    print("\n=== Loading Test Data ===")
    test_df = load_test_data()
    
    # 테스트 에피소드 로드 및 전처리
    print("\n=== Processing Test Episodes ===")
    sequences, episode_ids = load_test_episodes(test_df, action_type_map, scaler)
    
    # 모델 초기화
    input_dim = sequences.shape[2]
    print(f"Input dimension: {input_dim}")
    model = ImprovedPassPredictor(input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 예측
    print("\n=== Making Predictions ===")
    predictions = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i+batch_size]
            batch_sequences = torch.FloatTensor(batch_sequences).to(device)
            
            outputs = model(batch_sequences)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    
    print(f"\nPredictions stats:")
    print(f"  end_x: [{predictions[:, 0].min():.2f}, {predictions[:, 0].max():.2f}], mean: {predictions[:, 0].mean():.2f}")
    print(f"  end_y: [{predictions[:, 1].min():.2f}, {predictions[:, 1].max():.2f}], mean: {predictions[:, 1].mean():.2f}")
    
    # 제출 파일 생성
    print("\n=== Creating Submission File ===")
    submission = pd.DataFrame({
        'game_episode': episode_ids,
        'end_x': predictions[:, 0],
        'end_y': predictions[:, 1]
    })
    
    # 좌표 범위 제한
    submission['end_x'] = submission['end_x'].clip(0, FIELD_LENGTH)
    submission['end_y'] = submission['end_y'].clip(0, FIELD_WIDTH)
    
    # sample_submission.csv와 동일한 순서로 정렬
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV, encoding='utf-8')
    submission = sample_submission[['game_episode']].merge(
        submission, 
        on='game_episode', 
        how='left'
    )
    
    # NaN 값 처리
    submission['end_x'] = submission['end_x'].fillna(FIELD_LENGTH / 2)
    submission['end_y'] = submission['end_y'].fillna(FIELD_WIDTH / 2)
    
    # 저장
    output_path = 'submission.csv'
    submission.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Submission file saved: {output_path}")
    print(f"\nFinal summary:")
    print(f"  Total episodes: {len(submission)}")
    print(f"  end_x range: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
    print(f"  end_y range: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")
    
    return submission

if __name__ == "__main__":
    inference()

