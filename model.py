"""
시계열 예측 모델 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class PassPredictor(nn.Module):
    """패스 좌표 예측 모델 (LSTM 기반)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, 
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super(PassPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention 메커니즘 (선택적)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 출력 레이어
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, 2)  # end_x, end_y
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        
        # Attention 적용
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 마지막 타임스텝 사용
        last_hidden = attn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, 2)
        
        return out

class SimpleLSTMPredictor(nn.Module):
    """간단한 LSTM 모델 (대안)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, 
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super(SimpleLSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 마지막 레이어의 마지막 타임스텝 사용
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        out = self.fc(last_hidden)
        return out

