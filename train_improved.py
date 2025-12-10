"""
개선된 모델 학습 스크립트 - Huber Loss, 더 많은 에포크
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import random

from model_improved import ImprovedPassPredictor
from data_preprocessing import load_train_data, create_features, prepare_sequences
from config import *

# 재현 가능성을 위한 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_euclidean_distance(pred, target):
    """유클리드 거리 계산 (대회 평가 지표)"""
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean().item()

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def train_improved_model():
    """개선된 모델 학습"""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로드 및 전처리
    print("\n=== Data Loading ===")
    df = load_train_data()
    df, action_type_map = create_features(df)
    
    # 시퀀스 준비 (정규화 포함)
    sequences, targets, episode_ids, scaler = prepare_sequences(df, action_type_map, is_train=True)
    
    # Train/Validation 분할
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.15, random_state=42  # 15%로 줄여 더 많은 학습 데이터
    )
    
    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 모델 초기화
    input_dim = sequences.shape[2]
    print(f"Input dimension: {input_dim}")
    model = ImprovedPassPredictor(input_dim=input_dim).to(device)
    
    # 손실 함수: Huber Loss (이상치에 강함)
    criterion = nn.SmoothL1Loss()  # Huber Loss
    
    # 옵티마이저: AdamW (weight decay 개선)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 학습률 스케줄러: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 학습
    best_val_loss = float('inf')
    best_val_euclidean = float('inf')
    patience_counter = 0
    
    print("\n=== Training Improved Model ===")
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for sequences_batch, targets_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            sequences_batch = sequences_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences_batch)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_euclidean = 0.0
        with torch.no_grad():
            for sequences_batch, targets_batch in val_loader:
                sequences_batch = sequences_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                outputs = model(sequences_batch)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item()
                
                # 유클리드 거리 계산
                euclidean_dist = calculate_euclidean_distance(outputs, targets_batch)
                val_euclidean += euclidean_dist
        
        val_loss /= len(val_loader)
        val_euclidean /= len(val_loader)
        
        # 학습률 스케줄러 업데이트
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Euclidean: {val_euclidean:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping (유클리드 거리 기준)
        if val_euclidean < best_val_euclidean:
            best_val_loss = val_loss
            best_val_euclidean = val_euclidean
            patience_counter = 0
            # 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_euclidean': val_euclidean,
                'action_type_map': action_type_map,
                'scaler': scaler,
            }, os.path.join(MODEL_DIR, 'best_model_improved.pth'))
            print(f"  -> Model saved! (Val Euclidean: {val_euclidean:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation MSE Loss: {best_val_loss:.4f}")
    print(f"Best validation Euclidean Distance: {best_val_euclidean:.4f}")

if __name__ == "__main__":
    train_improved_model()

