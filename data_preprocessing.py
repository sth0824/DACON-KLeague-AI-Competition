"""
데이터 전처리 및 특징 엔지니어링
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from config import *

def load_train_data() -> pd.DataFrame:
    """훈련 데이터 로드 (UTF-8 인코딩, 대회 규칙 준수)"""
    print("Loading train data...")
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8')
    print(f"Loaded {len(df)} rows")
    return df

def load_test_data() -> pd.DataFrame:
    """테스트 데이터 로드 (UTF-8 인코딩, 대회 규칙 준수)"""
    print("Loading test data...")
    df = pd.read_csv(TEST_CSV, encoding='utf-8')
    print(f"Loaded {len(df)} test episodes")
    return df

def create_features(df: pd.DataFrame, action_type_map: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """특징 엔지니어링"""
    df = df.copy()
    
    # 액션 타입 인코딩
    if action_type_map is None:
        # 모든 가능한 액션 타입 수집 (train 데이터 기준)
        train_df = load_train_data()
        all_action_types = train_df['type_name'].unique()
        action_type_map = {action: idx for idx, action in enumerate(all_action_types)}
    
    # 매핑 (없는 액션은 0으로 처리)
    df['action_type_encoded'] = df['type_name'].map(action_type_map).fillna(0).astype(int)
    
    # 결과 인코딩
    result_map = {'Successful': 1, 'Unsuccessful': 0, np.nan: 0.5}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(0.5)
    
    # 좌표 차이 계산
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    
    # 거리 계산
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # 각도 계산 (라디안)
    df['angle'] = np.arctan2(df['dy'], df['dx'])
    
    # 시간 차이 계산 (에피소드 내에서)
    df = df.sort_values(['game_episode', 'time_seconds'])
    df['time_diff'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0)
    
    # is_home을 숫자로 변환
    df['is_home'] = df['is_home'].astype(int)
    
    # NaN 처리 (end_x, end_y가 없는 경우 이전 값으로 채우기)
    df['end_x'] = df.groupby('game_episode')['end_x'].ffill()
    df['end_y'] = df.groupby('game_episode')['end_y'].ffill()
    
    # 여전히 NaN이 있으면 start 좌표로 채우기
    df['end_x'] = df['end_x'].fillna(df['start_x'])
    df['end_y'] = df['end_y'].fillna(df['start_y'])
    
    return df, action_type_map

def prepare_sequences(df: pd.DataFrame, action_type_map: Dict, 
                     is_train: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    시퀀스 데이터 준비
    각 에피소드를 독립적인 시퀀스로 변환
    """
    sequences = []
    targets = []
    episode_ids = []
    
    # 에피소드별로 그룹화
    for episode_id, group in df.groupby('game_episode'):
        group = group.sort_values('time_seconds').reset_index(drop=True)
        
        # 특징 선택
        feature_cols = ['start_x', 'start_y', 'end_x', 'end_y',
                       'time_seconds', 'is_home',
                       'action_type_encoded', 'result_encoded',
                       'dx', 'dy', 'distance', 'angle', 'time_diff']
        
        features = group[feature_cols].values
        
        # 타겟: 마지막 액션의 end_x, end_y
        if is_train:
            # 마지막 행의 end_x, end_y가 타겟
            if len(features) > 0:
                target = features[-1, 2:4]  # end_x, end_y
                # 마지막 행을 제외한 시퀀스
                sequence = features[:-1]
            else:
                continue
        else:
            # 테스트: 전체 시퀀스 사용
            sequence = features
            target = np.array([0.0, 0.0])  # 더미 타겟
        
        # 패딩 또는 트렁케이션
        if len(sequence) < SEQUENCE_LENGTH:
            # 패딩: 0으로 채우기
            padding = np.zeros((SEQUENCE_LENGTH - len(sequence), len(feature_cols)))
            sequence = np.vstack([sequence, padding])
        else:
            # 트렁케이션: 마지막 SEQUENCE_LENGTH개만 사용
            sequence = sequence[-SEQUENCE_LENGTH:]
        
        sequences.append(sequence)
        targets.append(target)
        episode_ids.append(episode_id)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f"Prepared {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    
    return sequences, targets, episode_ids

def load_test_episodes(test_df: pd.DataFrame, action_type_map: Dict = None) -> Tuple[np.ndarray, List[str]]:
    """테스트 에피소드 로드 및 전처리"""
    sequences = []
    episode_ids = []
    
    # action_type_map이 없으면 train 데이터에서 생성
    if action_type_map is None:
        train_df = load_train_data()
        _, action_type_map = create_features(train_df)
    
    for idx, row in test_df.iterrows():
        episode_id = row['game_episode']
        file_path = row['path']
        
        # 경로 수정 (./test/... -> test/...)
        if file_path.startswith('./'):
            file_path = file_path[2:]
        
        full_path = os.path.join(DATA_DIR, file_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found")
            continue
        
        # CSV 파일 로드 (UTF-8 인코딩, 대회 규칙 준수)
        episode_df = pd.read_csv(full_path, encoding='utf-8')
        
        # 특징 생성 (action_type_map 전달)
        episode_df, _ = create_features(episode_df, action_type_map)
        
        # 시퀀스 준비
        feature_cols = ['start_x', 'start_y', 'end_x', 'end_y',
                       'time_seconds', 'is_home',
                       'action_type_encoded', 'result_encoded',
                       'dx', 'dy', 'distance', 'angle', 'time_diff']
        
        features = episode_df[feature_cols].values
        
        # 패딩 또는 트렁케이션
        if len(features) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(features), len(feature_cols)))
            features = np.vstack([features, padding])
        else:
            features = features[-SEQUENCE_LENGTH:]
        
        sequences.append(features)
        episode_ids.append(episode_id)
    
    sequences = np.array(sequences)
    print(f"Loaded {len(sequences)} test sequences")
    
    return sequences, episode_ids

