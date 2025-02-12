import numpy as np

def check_input_file():
    print("=== input.npy 파일 분석 ===")
    
    # mmap_mode='r'로 메모리에 전체를 로드하지 않고 읽기
    input_data = np.load('data/input.npy', mmap_mode='r')
    
    # 기본 정보만 출력
    print(f"데이터 형태 (shape): {input_data.shape}")
    print(f"데이터 타입 (dtype): {input_data.dtype}")
    print(f"차원 수: {input_data.ndim}")
    
    # 메모리 사용량 계산
    size_bytes = input_data.size * input_data.itemsize
    size_gb = size_bytes / (1024**3)
    print(f"파일 크기: {size_gb:.2f} GB")
    
    # 첫 번째 데이터의 형태만 확인
    print("\n=== 첫 번째 데이터 형태 ===")
    print(f"First sample shape: {input_data[0].shape}")

if __name__ == "__main__":
    check_input_file()