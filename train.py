"""
모델 학습 스크립트
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

from model import PassPredictor, SimpleLSTMPredictor
from data_preprocessing import load_train_data, create_features, prepare_sequences
from config import *

# 재현 가능성을 위한 시드 고정 (코드 검증을 위해 중요)
def set_seed(seed=42):
    """재현 가능성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_euclidean_distance(pred, target):
    """
    유클리드 거리 계산 (대회 평가 지표)
    pred: (batch_size, 2) - 예측된 [end_x, end_y]
    target: (batch_size, 2) - 실제 [end_x, end_y]
    """
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean().item()

class SequenceDataset(Dataset):
    """시계열 데이터셋"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def train_model():
    """모델 학습"""
    # 재현 가능성을 위한 시드 고정
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 로드 및 전처리
    print("\n=== Data Loading ===")
    df = load_train_data()
    df, action_type_map = create_features(df)
    
    # 시퀀스 준비
    sequences, targets, episode_ids = prepare_sequences(df, action_type_map, is_train=True)
    
    # Train/Validation 분할
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화
    input_dim = sequences.shape[2]
    model = PassPredictor(input_dim=input_dim).to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3, verbose=True)
    
    # 학습
    best_val_loss = float('inf')
    best_val_euclidean = float('inf')
    patience_counter = 0
    
    print("\n=== Training ===")
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
                
                # 유클리드 거리 계산 (실제 평가 지표)
                euclidean_dist = calculate_euclidean_distance(outputs, targets_batch)
                val_euclidean += euclidean_dist
        
        val_loss /= len(val_loader)
        val_euclidean /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Euclidean: {val_euclidean:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
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
            }, os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f"  -> Model saved! (Val Loss: {val_loss:.4f}, Val Euclidean: {val_euclidean:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation MSE Loss: {best_val_loss:.4f}")
    print(f"Best validation Euclidean Distance: {best_val_euclidean:.4f}")

if __name__ == "__main__":
    train_model()

