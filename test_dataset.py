from params import config
from dataset import AudioDataset
import numpy as np

def test_dataset():
    # 1. 데이터셋 생성 테스트
    print("\n=== 데이터셋 생성 테스트 ===")
    train_dataset = AudioDataset(
        input_path="data/input.npy",
        output_path="data/output.npy",
        starts_path="data/starts.npy",
        config=config,
        mode='train'
    )
    
    val_dataset = AudioDataset(
        input_path="data/input.npy",
        output_path="data/output.npy",
        starts_path="data/starts.npy",
        config=config,
        mode='val'
    )
    
    test_dataset = AudioDataset(
        input_path="data/input.npy",
        output_path="data/output.npy",
        starts_path="data/starts.npy",
        config=config,
        mode='test'
    )

    # 2. 데이터셋 크기 확인
    print("\n=== 데이터셋 크기 확인 ===")
    print(f"Train 데이터셋 크기: {len(train_dataset)}")
    print(f"Validation 데이터셋 크기: {len(val_dataset)}")
    print(f"Test 데이터셋 크기: {len(test_dataset)}")

    # 3. 세션 중복 확인
    print("\n=== 세션 중복 확인 ===")
    train_sessions = set(train_dataset.session_indices)
    val_sessions = set(val_dataset.session_indices)
    test_sessions = set(test_dataset.session_indices)

    print("Train-Val 중복:", len(train_sessions & val_sessions))
    print("Val-Test 중복:", len(val_sessions & test_sessions))
    print("Train-Test 중복:", len(train_sessions & test_sessions))

    # 4. 데이터 형태 확인
    print("\n=== 데이터 형태 확인 ===")
    x, y = train_dataset[0]
    print(f"입력 데이터 형태: {x.shape}")
    print(f"출력 데이터 형태: {y.shape}")

    # 5. 정규화 확인
    print("\n=== 정규화 확인 ===")
    print(f"Position 범위: [{y[:3].min():.3f}, {y[:3].max():.3f}]")
    print(f"Quaternion 범위: [{y[3:].min():.3f}, {y[3:].max():.3f}]")

if __name__ == "__main__":
    test_dataset()