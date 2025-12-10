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

from model import PassPredictor, SimpleLSTMPredictor
from data_preprocessing import load_train_data, create_features, prepare_sequences
from config import *

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
        with torch.no_grad():
            for sequences_batch, targets_batch in val_loader:
                sequences_batch = sequences_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                outputs = model(sequences_batch)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'action_type_map': action_type_map,
            }, os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f"  -> Model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()

