import pandas as pd

df = pd.read_csv('submission.csv', encoding='utf-8')
print(f'Total rows: {len(df)}')
print(f'\nend_x stats:')
print(f'  min: {df["end_x"].min():.2f}')
print(f'  max: {df["end_x"].max():.2f}')
print(f'  mean: {df["end_x"].mean():.2f}')
print(f'  unique values: {df["end_x"].nunique()}')
print(f'\nend_y stats:')
print(f'  min: {df["end_y"].min():.2f}')
print(f'  max: {df["end_y"].max():.2f}')
print(f'  mean: {df["end_y"].mean():.2f}')
print(f'  unique values: {df["end_y"].nunique()}')
print(f'\nNaN count: end_x={df["end_x"].isna().sum()}, end_y={df["end_y"].isna().sum()}')
print(f'\nFirst 10 rows:')
print(df.head(10))

