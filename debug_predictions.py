import torch
import numpy as np
from inference import inference

# 추론 실행
submission = inference()

# 실제 예측값 확인
print("\n=== Detailed Prediction Analysis ===")
print(f"Unique end_x values: {submission['end_x'].nunique()}")
print(f"Unique end_y values: {submission['end_y'].nunique()}")
print(f"\nFirst 20 predictions:")
print(submission.head(20)[['game_episode', 'end_x', 'end_y']])

# 예측값 분포 확인
print(f"\nend_x value counts (top 10):")
print(submission['end_x'].value_counts().head(10))
print(f"\nend_y value counts (top 10):")
print(submission['end_y'].value_counts().head(10))

