# Colab GPU 연동 가이드

## 📋 준비 사항

1. Google Colab Extension 설치 완료 ✅
2. Google 계정 로그인
3. 프로젝트 파일 준비

## 🚀 Colab 연동 방법

### 방법 1: Colab Extension 사용 (추천)

1. **VS Code에서 Colab Extension 사용**
   - 프로젝트 폴더에서 `DACON_KLeague_Training.ipynb` 파일 열기
   - Colab Extension 아이콘 클릭
   - "Open in Colab" 선택

2. **자동으로 Colab에서 열림**
   - GPU 자동 설정
   - 파일 자동 업로드

### 방법 2: Google Drive 업로드

1. **프로젝트 압축**
   ```bash
   # Windows PowerShell
   Compress-Archive -Path .\* -DestinationPath DACON-KLeague.zip
   ```

2. **Google Drive에 업로드**
   - Google Drive 접속
   - 압축 파일 업로드

3. **Colab에서 실행**
   - 새 Colab 노트북 생성
   - `DACON_KLeague_Training.ipynb` 내용 복사
   - Drive 마운트 후 압축 해제

### 방법 3: GitHub 연동 (가장 편리)

1. **GitHub에 프로젝트 업로드**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin [your-repo-url]
   git push -u origin main
   ```

2. **Colab에서 클론**
   ```python
   !git clone https://github.com/[your-username]/DACON-KLeague-AI-Competition.git
   %cd DACON-KLeague-AI-Competition
   ```

## ⚙️ Colab 노트북 설정

### GPU 활성화
1. Colab 메뉴: `런타임` → `런타임 유형 변경`
2. 하드웨어 가속기: **GPU** 선택
3. 저장

### 필수 파일 업로드
- `config.py`
- `data_preprocessing.py`
- `model_improved.py`
- `train_improved.py`
- `inference_improved.py`
- `train.csv`
- `test.csv`
- `test/` 폴더 전체
- `sample_submission.csv`

## 📝 Colab 노트북 실행 순서

1. **GPU 확인**
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. **패키지 설치**
   ```python
   !pip install torch numpy pandas scikit-learn tqdm
   ```

3. **데이터 업로드** (Google Drive 또는 직접 업로드)

4. **학습 실행**
   ```python
   !python train_improved.py
   ```

5. **추론 실행**
   ```python
   !python inference_improved.py
   ```

6. **결과 다운로드**
   ```python
   from google.colab import files
   files.download('submission_improved.csv')
   ```

## 💡 팁

- **세션 시간**: Colab 무료 버전은 12시간 후 자동 종료
- **GPU 할당**: 무료 버전은 T4 GPU (제한적)
- **데이터 저장**: Google Drive에 모델 저장 권장
- **재현성**: 시드 고정으로 동일한 결과 보장

## 🔧 문제 해결

### GPU가 인식되지 않을 때
- 런타임 재시작
- 런타임 유형 다시 확인

### 메모리 부족 시
- 배치 크기 줄이기 (`config.py`에서 `BATCH_SIZE` 조정)
- 모델 크기 줄이기 (`HIDDEN_DIM` 조정)

### 파일 경로 문제
- Colab에서는 `/content/`가 루트 디렉토리
- `config.py`의 경로 확인 필요

