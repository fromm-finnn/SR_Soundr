import sys
import random
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Subset
import torch.cuda.amp as amp
from pathlib import Path
from tqdm.auto import tqdm  # 수정된 import

from dataset import AudioDataset
from network import AudioNet
from trainer import AudioTrainer
from params import *

def set_seeds(seed=24):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_device(gpu_numbers):
    """여러 GPU 설정"""
    if torch.cuda.is_available():
        # 쉼표로 구분된 GPU 번호 처리
        if isinstance(gpu_numbers, str) and ',' in gpu_numbers:
            gpu_list = [int(x) for x in gpu_numbers.split(',')]
            device = torch.device("cuda")
        else:
            gpu_list = [int(gpu_numbers)]
            device = torch.device(f"cuda:{gpu_numbers}")
            
        print("\n=== CUDA 정보 ===")
        print(f"사용할 GPU 개수: {len(gpu_list)}")
        for gpu_id in gpu_list:
            print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            print(f"가용 메모리: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        print("================")
    else:
        device = torch.device("cpu")
        print("GPU를 찾을 수 없어 CPU를 사용합니다.")
    
    return device

def create_dataloaders(config):
    """데이터로더 생성"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        config.batch_size = (config.batch_size // n_gpus) * n_gpus
    
    # 데이터 경로 설정
    data_dir = Path("data")
    
    # 학습/검증/테스트용 데이터셋 생성
    train_dataset = AudioDataset(
        input_path=data_dir / "input.npy",
        output_path=data_dir / "output.npy",
        starts_path=data_dir / "starts.npy",
        config=config,
        mode='train',
        transform=None,
        env_type=config.environment_type  # config에서 환경 타입 가져오기
    )
    
    val_dataset = AudioDataset(
        input_path=data_dir / "input.npy",
        output_path=data_dir / "output.npy",
        starts_path=data_dir / "starts.npy",
        config=config,
        mode='val',
        transform=None,
        env_type=config.environment_type
    )
    
    test_dataset = AudioDataset(  # 테스트 데이터셋 추가
        input_path=data_dir / "input.npy",
        output_path=data_dir / "output.npy",
        starts_path=data_dir / "starts.npy",
        config=config,
        mode='test',
        transform=None,
        env_type=config.environment_type
    )
    
    print(f"학습 데이터 크기: {len(train_dataset)}")
    print(f"검증 데이터 크기: {len(val_dataset)}")
    print(f"테스트 데이터 크기: {len(test_dataset)}")  # 테스트 크기 출력 추가
    
    # DataLoader 생성
    # DataLoader 생성 (collate_fn 추가)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=True,
        collate_fn=AudioDataset.collate_fn  # 추가
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=True,
        collate_fn=AudioDataset.collate_fn  # 추가
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.test_num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=False,
        collate_fn=AudioDataset.collate_fn  # 추가
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python main.py [train/test] [gpu_numbers] [checkpoint_path]")
        sys.exit(1)

    mode = sys.argv[1]
    
    # GPU 번호 처리 수정
    if len(sys.argv) > 2:
        gpu_numbers = sys.argv[2]  # 문자열 그대로 유지
    else:
        gpu_numbers = "0"
        
    checkpoint_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 초기 설정
    set_seeds()
    device = setup_device(gpu_numbers)  # 수정된 setup_device 함수 사용
    
    # 데이터 경로 설정
    data_dir = Path("data")
    input_path = data_dir / "input.npy"
    output_path = data_dir / "output.npy"
    starts_path = data_dir / "starts.npy"

    if mode == "train":
        try:
            # 데이터로더 생성
            train_loader, val_loader, test_loader = create_dataloaders(config)  # test_loader 추가
            print("데이터로더 생성 완료!")
            
            # 샘플 데이터 확인
            sample_input, (sample_pos, sample_rot) = next(iter(train_loader))
            print(f"입력 형태: {sample_input.shape}")
            print(f"위치 출력 형태: {sample_pos.shape}")
            print(f"회전 출력 형태: {sample_rot.shape}")
            
            # AudioNet 초기화 파라미터 확인 추가
            print("\n=== AudioNet 초기화 파라미터 ===")
            print(f"config.sample_num: {config.sample_num}")
            print(f"config.microphone_num: {config.microphone_num}")
            print(f"config.output_num: {config.output_num}")
            print("============================\n")
            
            # 모델 초기화
            model = AudioNet(
                sample_num=config.sample_num,
                microphone_num=config.microphone_num,
                output_num=config.output_num,
                config=config
            )
            
            # DataParallel로 감싸기
            if torch.cuda.device_count() > 1:
                print(f"\n{torch.cuda.device_count()}개의 GPU를 사용한 병렬 처리를 시작합니다.")
                model = nn.DataParallel(model)
            
            model = model.to(device)

            # Trainer 생성
            trainer = AudioTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=config
            )

            if checkpoint_path:
                print(f"체크포인트 로딩: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 모델 가중치 로드
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 옵티마이저 상태 로드 및 수정
                optimizer_state = checkpoint['optimizer_state_dict']
                for param_group in optimizer_state['param_groups']:
                    param_group['lr'] = config.learning_rate
                    param_group['weight_decay'] = config.weight_decay
                trainer.optimizer.load_state_dict(optimizer_state)
                
                # 시작 에포크 설정
                trainer.start_epoch = checkpoint['epoch'] + 1
                
                # 스케줄러 설정 및 상태 복원
                if config.scheduler == "cosine":
                    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        trainer.optimizer,
                        T_max=config.num_epochs,
                        eta_min=config.learning_rate * 0.01
                    )
                    # 현재 에포크까지 스케줄러 진행
                    for _ in range(trainer.start_epoch - 1):
                        trainer.scheduler.step()
                
                # 학습 히스토리 복원
                if 'history' in checkpoint:
                    trainer.history = checkpoint['history']
                    trainer.n_iter = checkpoint['n_iter']
                    trainer.best_angle_acc = checkpoint['best_angle_acc']
                    trainer.best_composite_score = checkpoint['best_composite_score']
                
                print(f"체크포인트 로딩 완료 (에포크 {trainer.start_epoch}부터 시작)")

            # 학습 시작
            trainer.train()
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            raise e
    elif mode == "test":
        try:
            # 채널 직접 설정
            selected_channels = [0, 8]  # 4채널 고정
            config.microphone_num = 2
            config.batch_size = 32
            
            print(f"선택된 채널: {selected_channels}")
            
            # 데이터로더 생성 (테스트용)
            _, _, test_loader = create_dataloaders(config)
            
            # 데이터셋의 채널 설정 업데이트
            test_loader.dataset.selected_channels = selected_channels
            
            # 모델 초기화
            model = AudioNet(
                sample_num=config.sample_num,
                microphone_num=config.microphone_num,
                output_num=config.output_num,
                config=config
            ).to(device)
            
            if checkpoint_path:
                print(f"체크포인트 로딩: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # 평가 실행
            from evaluate import evaluate_model, print_evaluation_results
            results = evaluate_model(model, test_loader, device, config)
            print_evaluation_results(results, config)
            
        except Exception as e:
            print(f"평가 중 오류 발생: {str(e)}")
            raise e
    else:
        print("사용법: python main.py [train/test] [gpu_number]")