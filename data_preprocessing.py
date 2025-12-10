"""
데이터 전처리 및 특징 엔지니어링
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from sklearn.preprocessing import StandardScaler
import pickle
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
    # 각도 NaN 처리 (dx, dy가 모두 0인 경우)
    df['angle'] = df['angle'].fillna(0)
    
    # 시간 차이 계산 (에피소드 내에서)
    df = df.sort_values(['game_episode', 'time_seconds'])
    df['time_diff'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0)
    
    # 추가 특징: 속도 (거리/시간)
    df['velocity'] = df['distance'] / (df['time_diff'] + 1e-6)  # 0으로 나누기 방지
    df['velocity'] = df['velocity'].clip(-100, 100)  # 이상치 제거
    
    # 추가 특징: 상대적 위치 (필드 크기 대비)
    df['start_x_norm'] = df['start_x'] / FIELD_LENGTH
    df['start_y_norm'] = df['start_y'] / FIELD_WIDTH
    df['end_x_norm'] = df['end_x'] / FIELD_LENGTH
    df['end_y_norm'] = df['end_y'] / FIELD_WIDTH
    
    # 추가 특징: 공격 방향 (전방 패스 여부)
    df['forward_pass'] = (df['dx'] > 0).astype(int)
    
    # 추가 특징: 누적 통계 (에피소드 내)
    df['cumsum_distance'] = df.groupby('game_episode')['distance'].cumsum()
    df['cumsum_forward'] = df.groupby('game_episode')['forward_pass'].cumsum()
    
    # 추가 특징: 이동 평균 (최근 3개 액션)
    df['ma3_dx'] = df.groupby('game_episode')['dx'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df['ma3_dy'] = df.groupby('game_episode')['dy'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    
    # 추가 특징: 필드 구역 (3x3 그리드)
    df['zone_x'] = pd.cut(df['start_x'], bins=3, labels=[0, 1, 2]).astype(int)
    df['zone_y'] = pd.cut(df['start_y'], bins=3, labels=[0, 1, 2]).astype(int)
    
    # 추가 특징: 패스 길이 카테고리 (짧은/중간/긴 패스)
    # NaN을 먼저 처리한 후 카테고리 생성
    distance_filled = df['distance'].fillna(0)
    df['pass_length_category'] = pd.cut(
        distance_filled, 
        bins=[-1, 10, 25, float('inf')], 
        labels=[0, 1, 2],
        include_lowest=True
    )
    # NaN 처리 (cut 결과가 NaN인 경우 0으로 설정)
    df['pass_length_category'] = df['pass_length_category'].fillna(0).astype(int)
    
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
                     scaler: Optional[StandardScaler] = None,
                     is_train: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
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
        
        # 특징 선택 (대폭 확장된 특징 세트)
        feature_cols = [
            'start_x', 'start_y', 'end_x', 'end_y',
            'start_x_norm', 'start_y_norm', 'end_x_norm', 'end_y_norm',
            'time_seconds', 'is_home',
            'action_type_encoded', 'result_encoded',
            'dx', 'dy', 'distance', 'angle', 'time_diff',
            'velocity', 'forward_pass', 'pass_length_category',
            'cumsum_distance', 'cumsum_forward',
            'ma3_dx', 'ma3_dy',
            'zone_x', 'zone_y'
        ]
        
        # 없는 컬럼은 스킵
        available_cols = [col for col in feature_cols if col in group.columns]
        features = group[available_cols].values
        
        # 누락된 특징은 0으로 채우기
        if len(available_cols) < len(feature_cols):
            missing_cols = len(feature_cols) - len(available_cols)
            padding = np.zeros((features.shape[0], missing_cols))
            features = np.hstack([features, padding])
        
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
    
    # NaN 처리 (정규화 전에)
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 데이터 정규화 (타겟 제외)
    if scaler is None and is_train:
        # 학습 데이터: 새로운 scaler 생성
        n_samples, n_timesteps, n_features = sequences.shape
        sequences_2d = sequences.reshape(-1, n_features)
        scaler = StandardScaler()
        sequences_scaled = scaler.fit_transform(sequences_2d)
        sequences = sequences_scaled.reshape(n_samples, n_timesteps, n_features)
    elif scaler is not None:
        # 테스트 데이터: 학습 시 사용한 scaler 적용
        n_samples, n_timesteps, n_features = sequences.shape
        sequences_2d = sequences.reshape(-1, n_features)
        # NaN 처리
        sequences_2d = np.nan_to_num(sequences_2d, nan=0.0, posinf=0.0, neginf=0.0)
        sequences_scaled = scaler.transform(sequences_2d)
        sequences = sequences_scaled.reshape(n_samples, n_timesteps, n_features)
    
    print(f"Prepared {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    
    return sequences, targets, episode_ids, scaler

def load_test_episodes(test_df: pd.DataFrame, action_type_map: Dict = None, 
                      scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, List[str]]:
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
        
        # 시퀀스 준비 (학습과 동일한 확장된 특징 세트)
        feature_cols = [
            'start_x', 'start_y', 'end_x', 'end_y',
            'start_x_norm', 'start_y_norm', 'end_x_norm', 'end_y_norm',
            'time_seconds', 'is_home',
            'action_type_encoded', 'result_encoded',
            'dx', 'dy', 'distance', 'angle', 'time_diff',
            'velocity', 'forward_pass', 'pass_length_category',
            'cumsum_distance', 'cumsum_forward',
            'ma3_dx', 'ma3_dy',
            'zone_x', 'zone_y'
        ]
        
        # 없는 컬럼은 스킵
        available_cols = [col for col in feature_cols if col in episode_df.columns]
        features = episode_df[available_cols].values
        
        # 누락된 특징은 0으로 채우기
        if len(available_cols) < len(feature_cols):
            missing_cols = len(feature_cols) - len(available_cols)
            padding = np.zeros((features.shape[0], missing_cols))
            features = np.hstack([features, padding])
        
        # 패딩 또는 트렁케이션
        if len(features) < SEQUENCE_LENGTH:
            padding = np.zeros((SEQUENCE_LENGTH - len(features), len(feature_cols)))
            features = np.vstack([features, padding])
        else:
            features = features[-SEQUENCE_LENGTH:]
        
        sequences.append(features)
        episode_ids.append(episode_id)
    
    sequences = np.array(sequences)
    
    # NaN 처리
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 정규화 적용 (학습 시 사용한 scaler)
    if scaler is not None:
        n_samples, n_timesteps, n_features = sequences.shape
        sequences_2d = sequences.reshape(-1, n_features)
        # NaN 처리 (추가 안전장치)
        sequences_2d = np.nan_to_num(sequences_2d, nan=0.0, posinf=0.0, neginf=0.0)
        sequences_scaled = scaler.transform(sequences_2d)
        sequences = sequences_scaled.reshape(n_samples, n_timesteps, n_features)
    
    print(f"Loaded {len(sequences)} test sequences")
    
    return sequences, episode_ids

