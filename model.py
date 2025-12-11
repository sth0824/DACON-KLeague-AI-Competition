"""
개선된 모델 정의 - 앙상블 및 고급 기법
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class ImprovedPassPredictor(nn.Module):
    """개선된 패스 좌표 예측 모델 (LSTM + GRU 앙상블)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, 
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super(ImprovedPassPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 브랜치
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # 양방향
        )
        
        # GRU 브랜치 (다른 관점 학습)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 양방향이므로 2배
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 양 브랜치 결합
        combined_dim = hidden_dim * 4  # LSTM + GRU, 양방향 = 4배
        
        # 출력 레이어 (더 깊은 네트워크)
        self.fc1 = nn.Linear(combined_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(hidden_dim // 2, 2)  # end_x, end_y
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM 브랜치
        lstm_out, _ = self.lstm(x)
        
        # GRU 브랜치
        gru_out, _ = self.gru(x)
        
        # Attention (LSTM 출력에 적용)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 마지막 타임스텝 결합
        lstm_last = attn_out[:, -1, :]
        gru_last = gru_out[:, -1, :]
        combined = torch.cat([lstm_last, gru_last], dim=1)
        
        # 깊은 FC 네트워크
        out = self.fc1(combined)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        
        return out


class ResidualBlock(nn.Module):
    """Residual Block for better gradient flow"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

