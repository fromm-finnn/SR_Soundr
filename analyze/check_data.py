import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from dataset import AudioDataset
from torch.utils.data import DataLoader
from params import config

def analyze_angle_distributions(train_loader, val_loader):
    """Train과 Val 세션의 각도 분포 분석 및 시각화"""
    
    def extract_angles(loader, mode='train'):
        angles = []
        offsets = []  # 시간 오프셋 저장
        
        print(f"\n=== {mode.upper()} 세션 분석 ===")
        
        for sess_id in loader.dataset.session_indices:
            session = loader.dataset.starts[sess_id]
            start_idx = session['start']
            end_idx = session['end']
            
            # 쿼터니온 데이터 추출
            quaternions = loader.dataset.outputs[start_idx:end_idx, 3:7]
            
            # 쿼터니온을 오일러 각도로 변환
            rot = R.from_quat(quaternions)
            euler_angles = rot.as_euler('xyz', degrees=True)  # roll, pitch, yaw
            
            # 세션별 통계
            print(f"\n세션 {sess_id} 통계:")
            print(f"데이터 포인트 수: {len(euler_angles)}")
            print(f"Yaw 범위: {euler_angles[:,2].min():.1f}° ~ {euler_angles[:,2].max():.1f}°")
            print(f"시작 인덱스: {start_idx}, 종료 인덱스: {end_idx}")
            
            angles.extend(euler_angles)
            offsets.append(session.get('offset', 0))  # offset 정보 저장
            
        return np.array(angles), offsets

    # Train과 Val 데이터의 각도 추출
    train_angles, train_offsets = extract_angles(train_loader, 'train')
    val_angles, val_offsets = extract_angles(val_loader, 'val')
    
    # 결과 저장 디렉토리 생성
    save_dir = Path("analysis_results")
    save_dir.mkdir(exist_ok=True)
    
    # 그래프 설정
    plt.figure(figsize=(15, 10))
    
    # Yaw 각도 분포 비교
    plt.subplot(2, 1, 1)
    sns.histplot(data=train_angles[:,2], label='Train', bins=50, alpha=0.5)
    sns.histplot(data=val_angles[:,2], label='Val', bins=50, alpha=0.5)
    plt.title('Yaw Angle Distribution')
    plt.xlabel('Yaw Angle (degrees)')
    plt.ylabel('Count')
    plt.legend()
    
    # Pitch 각도 분포 비교
    plt.subplot(2, 1, 2)
    sns.histplot(data=train_angles[:,1], label='Train', bins=50, alpha=0.5)
    sns.histplot(data=val_angles[:,1], label='Val', bins=50, alpha=0.5)
    plt.title('Pitch Angle Distribution')
    plt.xlabel('Pitch Angle (degrees)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'angle_distributions.png')
    plt.close()
    
    # 오프셋 분석
    print("\n=== 오프셋 분석 ===")
    print("Train 세션 오프셋:", train_offsets)
    print("Val 세션 오프셋:", val_offsets)
    
    return train_angles, val_angles, train_offsets, val_offsets

def main():
    # 데이터 경로 설정
    data_dir = Path("data")
    
    # 데이터셋 생성
    train_dataset = AudioDataset(
        input_path=data_dir / "input.npy",
        output_path=data_dir / "output.npy",
        starts_path=data_dir / "starts.npy",
        config=config,
        mode='train'
    )
    
    val_dataset = AudioDataset(
        input_path=data_dir / "input.npy",
        output_path=data_dir / "output.npy",
        starts_path=data_dir / "starts.npy",
        config=config,
        mode='val'
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 분석을 위해 셔플 비활성화
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # 분석 실행
    train_angles, val_angles, train_offsets, val_offsets = analyze_angle_distributions(train_loader, val_loader)

if __name__ == "__main__":
    main()