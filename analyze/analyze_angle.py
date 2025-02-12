# analyze_angles.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from dataset import AudioDataset  # dataset.py에서 AudioDataset 임포트

class SimpleConfig:
    """간단한 설정 클래스"""
    def __init__(self):
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.random_seed = 42
        self.environment_type = 'same_user_same_space'
        self.audio_mean = 0.0
        self.audio_std = 0.0008
        self.pos_min = [-3.70873404, 0.81547654, -13.88833714]
        self.pos_max = [1.81399226, 2.48111653, 4.08869743]
        self.use_augmentation = False
        self.sequence_length = 30
        self.batch_size = 32
        
        # 증강 관련 설정
        self.noise_smoothing_kernel = 5
        self.impulse_response_length = 32
        
    def get_augmentation_params(self):
        return {
            'noise_prob': 0.5,
            'noise_level_range': (0.001, 0.002),
            'noise_correlation': 0.7
        }

def quaternion_to_euler(quaternions):
    """쿼터니온을 오일러 각도로 변환 (도 단위)"""
    r = R.from_quat(quaternions)
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles

def analyze_angles(dataset, save_dir="angle_analysis_results"):
    """데이터셋의 각도 분포 분석"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 모든 각도 데이터 수집
    all_angles = []
    all_positions = []
    
    print("각도 데이터 수집 중...")
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"처리 중: {i}/{len(dataset)}")
            
        _, (pos, quat) = dataset[i]
        all_positions.append(pos.numpy())
        
        # 쿼터니온을 오일러 각도로 변환
        euler = quaternion_to_euler(quat.numpy())
        all_angles.append(euler)
    
    all_angles = np.array(all_angles)
    all_positions = np.array(all_positions)
    
    # 각 축별 분석
    axes = ['X', 'Y', 'Z']
    plt.figure(figsize=(15, 15))
    
    for i, axis in enumerate(axes):
        plt.subplot(3, 1, i+1)
        
        # 히스토그램
        sns.histplot(all_angles[:, i], bins=36, stat='count')
        plt.title(f"{axis}-axis Rotation Distribution")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Count")
        
        # 통계 정보
        mean_angle = np.mean(all_angles[:, i])
        std_angle = np.std(all_angles[:, i])
        median_angle = np.median(all_angles[:, i])
        
        plt.axvline(mean_angle, color='r', linestyle='dashed', label=f'Mean: {mean_angle:.1f}°')
        plt.text(0.02, 0.95, 
                f'Mean: {mean_angle:.1f}°\n'
                f'Std: {std_angle:.1f}°\n'
                f'Median: {median_angle:.1f}°', 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(save_dir / f'angle_distribution_{timestamp}.png')
    plt.close()
    
    # 극좌표 시각화
    plt.figure(figsize=(15, 5))
    for i, axis in enumerate(axes):
        plt.subplot(1, 3, i+1, projection='polar')
        hist, bins = np.histogram(all_angles[:, i], bins=36, range=(-180, 180))
        center = (bins[:-1] + bins[1:]) / 2
        center_rad = np.radians(center)
        plt.bar(center_rad, hist, width=np.radians(10))
        plt.title(f"{axis}-axis Polar Distribution")
    
    plt.tight_layout()
    plt.savefig(save_dir / f'angle_polar_{timestamp}.png')
    plt.close()
    
    # 통계 정보 저장
    stats = {
        "timestamp": timestamp,
        "total_samples": len(all_angles),
        "axes": {}
    }
    
    for i, axis in enumerate(axes):
        stats["axes"][axis] = {
            "mean": float(np.mean(all_angles[:, i])),
            "std": float(np.std(all_angles[:, i])),
            "median": float(np.median(all_angles[:, i])),
            "min": float(np.min(all_angles[:, i])),
            "max": float(np.max(all_angles[:, i]))
        }
    
    with open(save_dir / f'angle_stats_{timestamp}.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("\n=== 각도 분석 결과 ===")
    for axis in axes:
        print(f"\n{axis}-축 통계:")
        print(f"평균: {stats['axes'][axis]['mean']:.1f}°")
        print(f"표준편차: {stats['axes'][axis]['std']:.1f}°")
        print(f"범위: {stats['axes'][axis]['min']:.1f}° ~ {stats['axes'][axis]['max']:.1f}°")

def main():
    # 데이터 경로 설정
    data_dir = Path("data")  # 실제 데이터 경로로 수정
    input_path = data_dir / "input.npy"
    output_path = data_dir / "output.npy"
    starts_path = data_dir / "starts.npy"
    
    # 설정 및 데이터셋 초기화
    config = SimpleConfig()
    dataset = AudioDataset(
        input_path=input_path,
        output_path=output_path,
        starts_path=starts_path,
        config=config,
        mode='val'  # 검증 데이터 분석
    )
    
    # 각도 분석 실행
    analyze_angles(dataset)

if __name__ == "__main__":
    main()