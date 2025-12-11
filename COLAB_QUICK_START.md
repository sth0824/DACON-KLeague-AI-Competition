# 🚀 Colab GPU 연동 빠른 시작 가이드

## 📌 Colab Extension 사용 방법

### 1단계: VS Code에서 노트북 열기
1. VS Code에서 `DACON_KLeague_Training.ipynb` 파일 열기
2. 상단에 "Open in Colab" 버튼이 보이면 클릭
3. 또는 우클릭 → "Open in Colab"

### 2단계: Colab에서 설정
1. **GPU 활성화 (필수!)**
   - `런타임` → `런타임 유형 변경`
   - 하드웨어 가속기: **GPU** 선택
   - 저장

2. **첫 번째 셀 실행**
   - GPU 확인 셀 실행
   - GPU가 인식되는지 확인

### 3단계: 데이터 업로드

**방법 A: 직접 업로드 (간단)**
1. Colab 왼쪽 사이드바의 📁 아이콘 클릭
2. "파일 업로드" 클릭
3. 다음 파일들 업로드:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`
   - `config.py`
   - `data_preprocessing.py`
   - `model_improved.py`
   - `train_improved.py`
   - `inference_improved.py`
   - `test/` 폴더 전체 (압축 후 업로드 가능)

**방법 B: Google Drive 사용 (대용량 권장)**
1. Google Drive에 프로젝트 폴더 업로드
2. 노트북의 "3-1. Google Drive 마운트" 셀 실행
3. 인증 후 경로 설정

**방법 C: GitHub 클론**
1. GitHub에 프로젝트 업로드
2. 노트북의 "3-2. GitHub 클론" 셀 수정 후 실행

### 4단계: 학습 실행
1. 모든 셀을 순서대로 실행
2. 학습 진행 상황 확인
3. GPU 사용 시 훨씬 빠름!

### 5단계: 결과 다운로드
- 학습 완료 후 `submission_improved.csv` 자동 다운로드

## ⚡ 빠른 체크리스트

- [ ] GPU 활성화 (런타임 → 런타임 유형 변경 → GPU)
- [ ] 필수 파일 업로드 완료
- [ ] 필수 파일 확인 셀에서 모든 파일 ✓ 확인
- [ ] 학습 실행
- [ ] 결과 다운로드

## 💡 팁

1. **GPU 할당**: 무료 버전은 T4 GPU (제한적)
2. **세션 시간**: 12시간 후 자동 종료 (무료 버전)
3. **모델 저장**: Google Drive에 저장 권장
4. **재현성**: 시드 고정으로 동일한 결과 보장

## 🔧 문제 해결

### GPU가 인식되지 않을 때
- 런타임 재시작
- 런타임 유형 다시 확인

### 파일을 찾을 수 없을 때
- 현재 디렉토리 확인: `!pwd`
- 파일 목록 확인: `!ls`
- 경로 수정 필요 시 `config.py` 확인

### 메모리 부족 시
- `config.py`에서 `BATCH_SIZE` 줄이기
- `HIDDEN_DIM` 줄이기

