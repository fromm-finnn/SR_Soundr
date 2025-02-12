import numpy as np

# 데이터 로드
data = np.load('data/test_sample.npy')

# 기본 정보 출력
print("Shape:", data.shape)
print("Data type:", data.dtype)

# 처음 몇 개의 샘플 출력
print("\n처음 5개 프레임:")
for i in range(min(5, len(data))):
    print(f"\nFrame {i}:")
    print(data[i])