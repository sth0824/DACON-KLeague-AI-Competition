"""
Colab에서 빠르게 시작하기 위한 설정 스크립트
이 파일을 Colab에서 실행하면 자동으로 환경을 설정합니다.
"""
import os
import sys

def setup_colab_environment():
    """Colab 환경 설정"""
    print("=== Colab 환경 설정 ===")
    
    # Colab인지 확인
    try:
        import google.colab
        IN_COLAB = True
        print("✓ Google Colab 환경 감지됨")
    except ImportError:
        IN_COLAB = False
        print("⚠ 로컬 환경입니다")
        return
    
    # GPU 확인
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU를 사용할 수 없습니다. 런타임 → 런타임 유형 변경 → GPU 선택")
    
    # 현재 디렉토리 확인
    current_dir = os.getcwd()
    print(f"현재 작업 디렉토리: {current_dir}")
    
    # 데이터 디렉토리 설정
    if '/content' in current_dir:
        # Colab 환경
        data_dir = current_dir
        print(f"데이터 디렉토리: {data_dir}")
        
        # 필수 파일 확인
        required_files = ['train.csv', 'test.csv', 'config.py', 'data_preprocessing.py']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"\n⚠ 누락된 파일: {', '.join(missing_files)}")
            print("다음 방법으로 파일을 업로드하세요:")
            print("1. Google Drive에 업로드 후 마운트")
            print("2. Colab 파일 메뉴에서 직접 업로드")
            print("3. GitHub에서 클론")
        else:
            print("✓ 필수 파일 확인 완료")
    
    print("\n=== 설정 완료 ===")
    print("이제 train_improved.py를 실행할 수 있습니다!")

if __name__ == "__main__":
    setup_colab_environment()

