import sys
import random
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Subset
import torch.cuda.amp as amp
from pathlib import Path
from tqdm.auto import tqdm  
import argparse  
import os

from dataset import AudioDataset
from network import AudioNetV3  
from trainer import AudioTrainer
from params import *
from evaluate import evaluate_model, print_evaluation_results

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
    
    print("\n=== 데이터셋 초기화 중... ===")
    
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
    
    test_dataset = AudioDataset( 
        input_path=data_dir / "input.npy",
        output_path=data_dir / "output.npy",
        starts_path=data_dir / "starts.npy",
        config=config,
        mode='test',
        transform=None,
        env_type=config.environment_type
    )
    
    # 데이터셋 크기 요약 출력
    print(f"\n=== 데이터셋 크기 요약 ===")
    print(f"학습: {len(train_dataset):,} 샘플")
    print(f"검증: {len(val_dataset):,} 샘플")
    print(f"테스트: {len(test_dataset):,} 샘플")
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=True,
        collate_fn=AudioDataset.collate_fn  
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
        collate_fn=AudioDataset.collate_fn  
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
        collate_fn=AudioDataset.collate_fn  
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 처리
    parser = argparse.ArgumentParser(description='SoundR-DL 학습 및 평가')
    
    # 필수 인자
    parser.add_argument('mode', type=str, choices=['train', 'test'], 
                        help='실행 모드 (train 또는 test)')
    
    # 선택적 인자
    parser.add_argument('--gpu', type=str, default='0',
                        help='사용할 GPU 번호 (쉼표로 구분, 기본값: 0)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='로드할 체크포인트 경로')
    parser.add_argument('--resume_training', action='store_true',
                        help='체크포인트에서 학습 상태까지 복원 (기본값: False)')
    
    # 마이크 및 시퀀스 관련 인자
    parser.add_argument('--mic_num', type=int, default=4,
                        help='마이크 채널 수 (기본값: 4)')
    parser.add_argument('--mic_array', type=str, default='0,4,8,12',
                        help='사용할 마이크 인덱스 (쉼표로 구분, 기본값: 0,4,8,12)')
    parser.add_argument('--sequence_length', type=int, default=20,
                        help='시퀀스 길이 (기본값: 20)')
    parser.add_argument('--optimize_two_channel', action='store_true',
                        help='2채널 최적화 활성화 (기본값: False)')
    
    # 학습 관련 인자 
    parser.add_argument('--batch_size', type=int, default=None,
                        help='배치 크기 (기본값: params.py에서 설정)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='학습률 (기본값: params.py에서 설정)')
    
    # 테스트 관련 인자 
    parser.add_argument('--run_snr_test', action='store_true',
                        help='SNR 테스트 실행 여부 (기본값: False)')
    
    # 검증 관련 인자
    parser.add_argument('--measure_latency', action='store_true',
                        help='검증 시 latency 측정 여부 (기본값: False)')
    
    # 체크포인트 관련 인자 
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='실험 이름 (체크포인트 저장 폴더명, 기본값: 자동 생성)')
    
    # 모델 경량화 관련 인자
    parser.add_argument('--use_depthwise_separable', dest='use_depthwise_separable', action='store_true',
                        help='Depthwise Separable Convolution 사용 활성화 (기본값: True)')
    parser.add_argument('--no_depthwise_separable', dest='use_depthwise_separable', action='store_false',
                        help='Depthwise Separable Convolution 사용 비활성화')
    parser.add_argument('--depthwise_for_all_blocks', action='store_true',
                        help='모든 CNN 블록에 분리형 컨볼루션 적용 (기본값: False)')
    
    # LSTM 경량화 관련 인자 추가
    parser.add_argument('--use_lightweight_lstm', dest='use_lightweight_lstm', action='store_true',
                        help='경량화된 LSTM 사용 활성화 (기본값: True)')
    parser.add_argument('--no_lightweight_lstm', dest='use_lightweight_lstm', action='store_false',
                        help='경량화된 LSTM 사용 비활성화')
    parser.add_argument('--lightweight_lstm_hidden_size', type=int, default=384,
                        help='경량화된 LSTM의 은닉층 크기 (기본값: 384)')
    parser.add_argument('--lightweight_lstm_bidirectional', action='store_true',
                        help='경량화된 LSTM에서 양방향 사용 (기본값: False)')
    parser.add_argument('--lightweight_lstm_num_layers', type=int, default=1,
                        help='경량화된 LSTM의 레이어 수 (기본값: 1)')
    
    # Context Module 경량화 관련 인자 추가
    parser.add_argument('--use_lightweight_context', dest='use_lightweight_context', action='store_true',
                        help='경량화된 Context Module 사용 활성화 (기본값: True)')
    parser.add_argument('--no_lightweight_context', dest='use_lightweight_context', action='store_false',
                        help='경량화된 Context Module 사용 비활성화')
    parser.add_argument('--lightweight_context_hidden_size_ratio', type=float, default=0.5,
                        help='경량화된 Context Module의 은닉층 크기 비율 (기본값: 0.5)')
    
    parser.set_defaults(use_depthwise_separable=True, use_lightweight_lstm=True, use_lightweight_context=True)
    
    args = parser.parse_args()
    
    # 인자 처리
    mode = args.mode
    gpu_numbers = args.gpu
    checkpoint_path = args.checkpoint
    
    # 마이크 및 시퀀스 설정
    config.microphone_num = args.mic_num
    print(f"마이크 채널 수 설정: {config.microphone_num}")
    
    # 마이크 배열 설정
    mic_array = [int(idx) for idx in args.mic_array.split(',')]
    # 명시적으로 config.selected_channels 설정
    config.selected_channels = mic_array
    print(f"선택된 마이크 채널: {config.selected_channels}")
    
    # 2채널 최적화 설정
    if args.optimize_two_channel and args.mic_num == 2:
        config.optimize_for_two_channel = True
        config.two_channel_indices = tuple(mic_array)
        print(f"2채널 최적화 활성화: {config.optimize_for_two_channel}")
        print(f"2채널 인덱스: {config.two_channel_indices}")
    else:
        config.optimize_for_two_channel = False
    
    # 경량화 관련 설정
    config.use_depthwise_separable = args.use_depthwise_separable
    config.depthwise_for_all_blocks = args.depthwise_for_all_blocks
    print(f"분리형 컨볼루션 사용: {config.use_depthwise_separable} (전체 블록: {config.depthwise_for_all_blocks})")
    
    # LSTM 경량화 설정 적용
    config.use_lightweight_lstm = args.use_lightweight_lstm
    config.lightweight_lstm_hidden_size = args.lightweight_lstm_hidden_size
    config.lightweight_lstm_bidirectional = args.lightweight_lstm_bidirectional
    config.lightweight_lstm_num_layers = args.lightweight_lstm_num_layers
    
    # Context Module 경량화 설정 적용
    config.use_lightweight_context = args.use_lightweight_context
    config.lightweight_context_hidden_size_ratio = args.lightweight_context_hidden_size_ratio
    
    if config.use_lightweight_lstm or config.use_lightweight_context:
        print("\n=== 경량화 설정 ===")
        if config.use_lightweight_lstm:
            print(f"경량화 LSTM: hidden_size={config.lightweight_lstm_hidden_size}, bidirectional={config.lightweight_lstm_bidirectional}, num_layers={config.lightweight_lstm_num_layers}")
        if config.use_lightweight_context:
            print(f"경량화 Context Module: hidden_size_ratio={config.lightweight_context_hidden_size_ratio}")
        print("=====================\n")
    
    # 시퀀스 길이 설정
    config.sequence_length = args.sequence_length
    print(f"시퀀스 길이 설정: {config.sequence_length}")
    
    # 배치 크기 설정 (명령줄에서 지정된 경우에만 변경)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
        config.test_batch_size = args.batch_size  # 테스트 배치 크기도 동일하게 설정
    print(f"배치 크기 설정: {config.batch_size}")
    
    # 학습률 설정 (명령줄에서 지정된 경우에만 변경)
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    print(f"학습률 설정: {config.learning_rate}")
    
    # latency 측정 여부 설정
    config.measure_latency = args.measure_latency
    print(f"Latency 측정 여부: {config.measure_latency}")
    
    # 실험 이름 설정 (체크포인트 저장 폴더명)
    if args.experiment_name:
        config.experiment_name = args.experiment_name
        print(f"실험 이름 설정: {config.experiment_name}")
    else:
        pass
    
    # 초기 설정
    set_seeds()
    device = setup_device(gpu_numbers)  
    
    # 데이터 경로 설정
    data_dir = Path("data")
    input_path = data_dir / "input.npy"
    output_path = data_dir / "output.npy"
    starts_path = data_dir / "starts.npy"

    if mode == "train":
        try:
            # 모델 초기화 - AudioNetV3 모델 사용
            print("AudioNetV3 모델 사용")
            model = AudioNetV3(
                sample_num=config.sample_num,
                microphone_num=config.microphone_num,
                output_num=config.output_num,
                config=config
            )
            
            # 모델을 GPU로 이동
            model = model.to(device)
            print(f"모델을 {device}로 이동했습니다.")
            
            # 데이터로더 생성
            train_loader, val_loader, test_loader = create_dataloaders(config)
            print("데이터로더 생성 완료!")
            
            # 체크포인트 로딩
            if checkpoint_path:
                try:
                    if args.resume_training:
                        print(f"체크포인트에서 학습 상태 전체 복원 시도: {checkpoint_path}")
                        # Trainer 객체 먼저 생성
                        trainer = AudioTrainer(model, train_loader, val_loader, device, config)
                        # trainer의 load_checkpoint 메서드로 전체 학습 상태 복원
                        success = trainer.load_checkpoint(checkpoint_path)
                        if success:
                            print(f"체크포인트에서 학습 상태 복원 완료: {checkpoint_path}")
                            print(f"에포크 {trainer.start_epoch}부터 학습을 재개합니다.")
                            # 복원된 에포크부터 시작
                            trainer.train()
                        else:
                            print("학습 상태 복원 실패. 처음부터 학습을 시작합니다.")
                            trainer = AudioTrainer(model, train_loader, val_loader, device, config)
                            trainer.train()
                    else:
                        # 기존 방식 - 모델 가중치만 로드
                        print(f"모델 가중치만 로드: {checkpoint_path}")
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"체크포인트 로딩 완료: {checkpoint_path}")
                        
                        # Trainer 초기화 및 학습 시작
                        try:
                            trainer = AudioTrainer(model, train_loader, val_loader, device, config)
                            # 학습 시작
                            trainer.train()
                        except Exception as e:
                            print(f"Trainer 생성 중 오류 발생: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            raise e
                except Exception as e:
                    print(f"체크포인트 로딩 중 오류 발생: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                # 체크포인트 없이 처음부터 학습
                try:
                    trainer = AudioTrainer(model, train_loader, val_loader, device, config)
                    # 학습 시작
                    trainer.train()
                except Exception as e:
                    print(f"Trainer 생성 중 오류 발생: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise e

        except Exception as e:
            print(f"오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    elif mode == "test":
        try:
            # 채널 설정 - config에서 가져오기
            print(f"선택된 채널: {config.selected_channels}")
            print(f"마이크 채널 수: {config.microphone_num}")
            
            # 데이터로더 생성 (테스트용)
            _, _, test_loader = create_dataloaders(config)
            
            # 모델 초기화 - AudioNetV3만 사용
            print("AudioNetV3 모델 사용")
            model = AudioNetV3(
                sample_num=config.sample_num,
                microphone_num=config.microphone_num,
                output_num=config.output_num,
                config=config
            )
            
            # 모델을 GPU로 이동 
            model = model.to(device)
            print(f"모델을 {device}로 이동했습니다.")
            
            if checkpoint_path:
                print(f"체크포인트 로딩: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("체크포인트 로딩 완료")
            else:
                print("경고: 체크포인트가 제공되지 않았습니다. 초기화된 모델로 평가합니다.")
            
            # SNR 테스트 실행 여부 확인
            if args.run_snr_test:
                # 노이즈 강건성 테스트 실행
                print("\n=== 노이즈 강건성 테스트 시작 ===")
                results_by_noise = evaluate_model(model, test_loader, device, config)
                print_evaluation_results(results_by_noise, config)
                
                # 결과 저장 디렉토리 생성
                os.makedirs("./results", exist_ok=True)
                
                # 결과 저장 (선택적)
                if checkpoint_path:
                    save_path = f"./results/noise_test_results_{checkpoint_path.split('/')[-1].split('.')[0]}.npy"
                else:
                    save_path = f"./results/noise_test_results_initialized_model.npy"
                    
                np.save(save_path, {
                    str(k): {
                        env: {
                            'distance_errors': np.array(v[env]['distance_errors']),
                            'angle_errors': np.array(v[env]['angle_errors']),
                            'samples': v[env]['samples']
                        } for env in v
                    } for k, v in results_by_noise.items()
                })
                print(f"\n결과가 {save_path}에 저장되었습니다.")
            else:
                # 기본 평가만 실행 (SNR 테스트 없이)
                print("\n=== 기본 모델 평가 시작 ===")
                # evaluate_model 함수를 수정하여 SNR 테스트 없이 기본 평가만 수행하는 함수 호출
                from evaluate import evaluate_model_basic
                results = evaluate_model_basic(model, test_loader, device, config)
                
                # 결과 저장 디렉토리 생성
                os.makedirs("./results", exist_ok=True)
                
                # 결과 저장
                if checkpoint_path:
                    save_path = f"./results/basic_test_results_{checkpoint_path.split('/')[-1].split('.')[0]}.npy"
                else:
                    save_path = f"./results/basic_test_results_initialized_model.npy"
                
                np.save(save_path, {
                    env: {
                        'distance_errors': np.array(results[env]['distance_errors']),
                        'angle_errors': np.array(results[env]['angle_errors']),
                        'samples': results[env]['samples']
                    } for env in results
                })
                
                # 결과 출력
                print("\n=== 평가 결과 ===")
                for env, metrics in results.items():
                    if metrics['samples'] == 0:
                        continue
                        
                    avg_distance = np.mean(metrics['distance_errors'])
                    avg_angle = np.mean(metrics['angle_errors'])
                    
                    print(f"\n{env}:")
                    print(f"├─ 샘플 수: {metrics['samples']}")
                    print(f"├─ 거리 오차: {avg_distance:.3f}m")
                    print(f"└─ 각도 오차: {avg_angle:.1f}°")
                
                print(f"\n결과가 {save_path}에 저장되었습니다.")
            
        except Exception as e:
            print(f"평가 중 오류 발생: {str(e)}")
            raise e
    else:
        parser.print_help()