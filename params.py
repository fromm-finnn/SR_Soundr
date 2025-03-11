import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch 
import numpy as np

@dataclass
class TrainingConfig:
    # 데이터 관련
    sample_num: int = 2400  # 세그먼트 길이 (0.05초 * 48000Hz)
    microphone_num: int = 4 #16 4 
    output_num: int = 7  #위치(3) + 쿼터니언(4)
    
    # 데이터셋 설정
    # val_size: int = 13288  # 최대 검증 데이터 크기
    
    # 데이터 분할 설정
    train_ratio: float = 0.7  # 학습용 세션 비율
    val_ratio: float = 0.15   # 검증용 세션 비율
    test_ratio: float = 0.15  # 테스트용 세션 비율
    random_seed: int = 42     # 데이터 분할용 시드
    
    # 환경 설정
    environment_type: str = 'same_user_same_space'  # 기본 환경 타입
    
    # 데이터셋 저장 설정
    save_split_info: bool = True  # 데이터 분할 정보 저장 여부
    split_info_dir: str = "split_info"  # 분할 정보 저장 디렉토리
    
    # 테스트 관련 설정
    test_batch_size: int = 1  # 테스트시 배치 크기
    test_num_workers: int = 2  # 테스트시 워커 수
    
    # 평가 메트릭 설정
    position_error_threshold: float = 0.5  # 위치 오차 허용 임계값 (미터)
    angle_error_threshold: float = 30.0    # 각도 오차 허용 임계값 (도)
    
    # 데이터 로더 설정
    batch_size: int = 256
    num_workers: int = 16
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    
    # 학습 하이퍼파라미터
    learning_rate: float = 5e-5 # 5e-5 
    weight_decay: float = 1e-4
    num_epochs: int = 200
    
    # 학습 안정성
    warmup_steps: int = 1000
    gradient_clip: float = 2.0
    dropout_rate: float = 0.2
    early_stopping_patience: int = 25
    
    # 옵티마이저 설정
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    
    # Mixed Precision 설정
    use_amp: bool = True
    scaler_growth_interval: int = 100
    
    # 점진적 학습 관련 설정
    warmup_epochs: int = 5  # 위치 학습에만 집중할 epoch 수
    rotation_ramp_epochs: int = 5  # 회전 손실을 점진적으로 증가시킬 epoch 수
    initial_rotation_weight: float = 0.0  # 초기 회전 손실 가중치
    final_rotation_weight: float = 4.0  # 최종 회전 손실 가중치
    position_loss_weight: float = 1.0  # 위치 손실 가중치
    rotation_loss_weight: float = 3.0  # 회전 손실 가중치 (final_rotation_weight와 동일하게 설정)

    # 체크포인트 설정
    save_freq: int = 10
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 5
    
    # 로깅 설정
    log_freq: int = 100
    use_wandb: bool = False
    project_name: Optional[str] = "soundr-dl"
    experiment_name: Optional[str] = None
    
    # 데이터 정규화 상수
    audio_mean: float = 0.0
    audio_std: float = 0.0008
    pos_min: Tuple[float, float, float] = (-3.70873404, 0.81547654, -13.88833714)
    pos_max: Tuple[float, float, float] = (1.81399226, 2.48111653, 4.08869743)
    
    # 데이터 증강 설정 (수정된 부분)
    use_augmentation: bool = True
    angle_aug_prob: float = 0.5    # 각도 증강을 적용할 확률
    max_angle_offset: float = 30.0 # 최대 각도 변화 (도)

    # 마이크 배열 설정 (각도 증강에 필요)
    mic_positions: list = field(default_factory=lambda: [
        [0, 0, 0],      # 첫 번째 마이크 (기준점)
        [0.1, 0, 0],    # 두 번째 마이크
        [0, 0.1, 0],    # 세 번째 마이크
        [0.1, 0.1, 0]   # 네 번째 마이크
    ])
    speed_of_sound: float = 343.0  # 음속 (m/s)
    sample_rate: int = 48000       # 샘플링 레이트 (Hz)

    # 노이즈 관련 설정만 남기고 나머지 제거
    noise_prob: float = 0.5     # temporal_noise_prob에서 이름 변경
    noise_level_range: Tuple[float, float] = (0.0001, 0.001)
    noise_smoothing_kernel: int = 5
    noise_correlation: float = 0.7
    
    # 데이터 필터링 설정
    min_distance: float = 0.1
    max_distance: float = 15.0
    min_angle: float = -180.0
    max_angle: float = 180.0
    
    # 세션 관련 설정
    min_session_length: int = 100
    max_session_length: int = 10000
    
    # 데이터 캐싱 설정
    use_cache: bool = True
    cache_dir: str = "cache"
    
    # 시퀀스 관련 설정
    sequence_length: int = 20
    stride: int = 1
    
    # LSTM 관련 설정
    lstm_hidden_size: int = 512
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True

    def __post_init__(self):
        """초기화 후 처리"""
        if torch.cuda.is_available():
            # 기존 디렉토리 생성
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)
            
            # 분할 정보 저장 디렉토리 생성
            if self.save_split_info:
                os.makedirs(self.split_info_dir, exist_ok=True)
            
            # 실험 이름이 없으면 자동 생성
            if self.experiment_name is None:
                self.experiment_name = ("dov_soundr_4ch(0,4,8,12)_lr5e5")
        
        # 데이터 분할 비율 검증
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(f"분할 비율의 합이 1이 되어야 합니다. (현재: {total_ratio})")

    def get_optimizer_params(self):
        """옵티마이저 파라미터 반환"""
        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }

    def get_scheduler_params(self):
        """스케줄러 파라미터 반환"""
        return {
            "warmup_steps": self.warmup_steps,
            "num_training_steps": self.num_epochs * 100,
            "num_cycles": 0.5,
        }

    def get_session_params(self):
        """세션 관련 파라미터 반환"""
        return {
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "min_session_length": self.min_session_length,
            "max_session_length": self.max_session_length,
            "random_seed": self.random_seed
        }
    
    def get_filter_params(self):
        """데이터 필터링 파라미터 반환"""
        return {
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "min_angle": self.min_angle,
            "max_angle": self.max_angle
        }

    def get_dataset_params(self):
        """데이터셋 관련 파라미터 반환"""
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "environment_type": self.environment_type,
            "save_split_info": self.save_split_info,
            "split_info_dir": self.split_info_dir
        }

    def get_test_params(self):
        """테스트 관련 파라미터 반환"""
        return {
            "batch_size": self.test_batch_size,
            "num_workers": self.test_num_workers,
            "position_error_threshold": self.position_error_threshold,
            "angle_error_threshold": self.angle_error_threshold
        }

    def get_evaluation_params(self):
        """평가 관련 파라미터 반환"""
        return {
            "position_error_threshold": self.position_error_threshold,
            "angle_error_threshold": self.angle_error_threshold,
            "environment_type": self.environment_type
        }


    def get_augmentation_params(self):
        """데이터 증강 파라미터 반환"""
        return {
            "use_augmentation": self.use_augmentation,
            "angle_aug_prob": self.angle_aug_prob,
            "max_angle_offset": self.max_angle_offset,
            "mic_positions": self.mic_positions,
            "speed_of_sound": self.speed_of_sound,
            "sample_rate": self.sample_rate
        }
# 기본 설정 인스턴스 생성
config = TrainingConfig()