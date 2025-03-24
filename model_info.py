import os
import torch
import numpy as np
import argparse
from network import AudioNetV3
from params import TrainingConfig

def count_parameters(model):
    """모델의 파라미터 수를 계산합니다."""
    # 총 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    
    # 학습 가능한 파라미터 수
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 레이어별 파라미터 수
    print("\n=== 레이어별 파라미터 수 ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")
    
    # 모듈별 파라미터 수 (개선된 분석)
    print("\n=== 모듈별 파라미터 수 ===")
    module_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 더 상세한 모듈 분류
            if 'cnn_blocks' in name:
                module_name = f"CNN_Block_{name.split('.')[1]}"
            elif 'channel_attention' in name:
                module_name = "Channel_Attention"
            elif 'self_attention' in name:
                module_name = "Self_Attention"
            elif 'context_module' in name:
                module_name = "Context_Module"
            elif 'lstm' in name:
                module_name = "LSTM"
            elif 'fc_position' in name:
                module_name = "Position_Head"
            elif 'fc_rotation' in name:
                module_name = "Rotation_Head"
            else:
                module_name = name.split('.')[0]
                
            if module_name not in module_params:
                module_params[module_name] = 0
            module_params[module_name] += param.numel()
    
    # 모듈별 파라미터 수와 비율을 정렬하여 출력
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    for module_name, param_count in sorted_modules:
        print(f"{module_name}: {param_count:,} ({param_count/trainable_params*100:.2f}%)")
    
    # 모델 크기 계산 (MB)
    model_size_bytes = 0
    for param in model.parameters():
        model_size_bytes += param.numel() * param.element_size()
    
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
        "module_params": module_params
    }

def analyze_model(mic_num, mic_array, optimize_two_channel=False, use_depthwise_separable=True, depthwise_for_all_blocks=False, use_lightweight_lstm=True, lightweight_lstm_hidden_size=384, lightweight_lstm_bidirectional=False, lightweight_lstm_num_layers=1, use_lightweight_context=True, lightweight_context_hidden_size_ratio=0.5):
    """지정된 설정으로 모델을 분석합니다."""
    # 설정 로드
    config = TrainingConfig()
    
    # 설정 업데이트
    config.microphone_num = mic_num
    config.selected_channels = mic_array
    
    if optimize_two_channel and mic_num == 2:
        config.optimize_for_two_channel = True
        config.two_channel_indices = tuple(mic_array)
    else:
        config.optimize_for_two_channel = False
    
    # 경량화 설정 추가
    config.use_depthwise_separable = use_depthwise_separable
    config.depthwise_for_all_blocks = depthwise_for_all_blocks
    
    # LSTM 경량화 설정 적용
    config.use_lightweight_lstm = use_lightweight_lstm
    config.lightweight_lstm_hidden_size = lightweight_lstm_hidden_size
    config.lightweight_lstm_bidirectional = lightweight_lstm_bidirectional
    config.lightweight_lstm_num_layers = lightweight_lstm_num_layers
    
    # Context Module 경량화 설정 적용
    config.use_lightweight_context = use_lightweight_context
    config.lightweight_context_hidden_size_ratio = lightweight_context_hidden_size_ratio
    
    # AudioNetV3 모델 초기화
    model = AudioNetV3(
        sample_num=config.sample_num,
        microphone_num=config.microphone_num,
        output_num=config.output_num,
        config=config
    )
    
    # 파라미터 수 계산
    param_info = count_parameters(model)
    
    # 결과 출력
    print(f"\n=== AudioNetV3 모델 정보 요약 ({mic_num}채널) ===")
    print(f"마이크 채널: {mic_array}")
    print(f"2채널 최적화: {config.optimize_for_two_channel}")
    print(f"총 파라미터 수: {param_info['total_params']:,}")
    print(f"학습 가능한 파라미터 수: {param_info['trainable_params']:,}")
    print(f"모델 크기: {param_info['model_size_mb']:.2f} MB")
    
    # 모델 구조 출력
    print("\n=== 모델 구조 ===")
    print(model)
    
    # 모델 설정 정보 출력
    print("\n=== 모델 설정 정보 ===")
    print(f"Channel Attention Reduction Ratio: {config.channel_attention_reduction_ratio}")
    print(f"Self Attention Dropout Rate: {config.self_attention_dropout}")
    print(f"Kernel Sizes: {config.kernel_sizes}")
    print(f"Context Module 사용: {config.use_context_module}")
    print(f"Position-Rotation 연결 사용: {config.use_position_for_rotation}")
    print(f"2채널 최적화: {config.optimize_for_two_channel}")
    print(f"분리형 컨볼루션 사용: {config.use_depthwise_separable} (전체 블록: {config.depthwise_for_all_blocks})")
    
    # 경량화 정보 출력
    if config.use_lightweight_lstm or config.use_lightweight_context:
        print("\n=== 경량화 설정 ===")
        if config.use_lightweight_lstm:
            print(f"경량화 LSTM: hidden_size={config.lightweight_lstm_hidden_size}, bidirectional={config.lightweight_lstm_bidirectional}, num_layers={config.lightweight_lstm_num_layers}")
        if config.use_lightweight_context:
            print(f"경량화 Context Module: hidden_size_ratio={config.lightweight_context_hidden_size_ratio}")
    
    # FLOPS 계산
    batch_size = 1
    sequence_length = 20
    sample_num = 2400
    input_tensor = torch.randn(batch_size, config.microphone_num, sample_num, sequence_length)
    
    import time
    from thop import profile
    
    flops, params = profile(model, inputs=(input_tensor,))
    
    # GFLOPS로 변환하여 출력
    print(f"\n모델 FLOPS: {flops / 1e9:.2f} GFLOPS")
    
    return param_info

def compare_models(models_info):
    """여러 모델 설정을 비교합니다."""
    print("\n" + "="*80)
    print("모델 비교 요약".center(80))
    print("="*80)
    
    headers = ["설정", "파라미터 수", "모델 크기(MB)"]
    row_format = "{:<25} {:<20} {:<15}"
    
    print(row_format.format(*headers))
    print("-"*80)
    
    for name, info in models_info.items():
        print(row_format.format(
            name,
            f"{info['trainable_params']:,}",
            f"{info['model_size_mb']:.2f}"
        ))
    
    print("\n" + "="*80)
    print("주요 모듈별 파라미터 비교".center(80))
    print("="*80)
    
    # 모든 모듈 이름 수집
    all_modules = set()
    for info in models_info.values():
        all_modules.update(info['module_params'].keys())
    
    # 주요 모듈만 선택 (예: CNN 블록, LSTM, 헤드 등)
    main_modules = [m for m in all_modules if any(
        keyword in m for keyword in ['CNN_Block', 'LSTM', 'Position_Head', 'Rotation_Head', 'Self_Attention', 'Channel_Attention']
    )]
    main_modules.sort()
    
    # 헤더 출력
    headers = ["모듈"] + list(models_info.keys())
    row_format = "{:<25}" + " {:<20}" * len(models_info)
    
    print(row_format.format(*headers))
    print("-"*80)
    
    # 각 모듈별 파라미터 수 비교
    for module in main_modules:
        row = [module]
        for name, info in models_info.items():
            param_count = info['module_params'].get(module, 0)
            row.append(f"{param_count:,}")
        print(row_format.format(*row))

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='AudioNetV3 모델 정보 분석')
    parser.add_argument('--compare', action='store_true', help='2채널과 4채널 모델 비교')
    parser.add_argument('--mic_num', type=int, default=4, help='마이크 채널 수 (기본값: 4)')
    parser.add_argument('--mic_array', type=str, default='0,4,8,12', help='마이크 인덱스 (쉼표로 구분)')
    parser.add_argument('--optimize_two_channel', action='store_true', help='2채널 최적화 활성화')
    parser.add_argument('--checkpoint', type=str, default=None, help='체크포인트 파일 경로 (지정 시 모델 가중치 로드)')
    parser.add_argument('--use_depthwise_separable', dest='use_depthwise_separable', action='store_true', help='분리형 컨볼루션 사용')
    parser.add_argument('--no_depthwise_separable', dest='use_depthwise_separable', action='store_false', help='일반 컨볼루션 사용')
    parser.add_argument('--depthwise_for_all_blocks', action='store_true', help='모든 블록에 분리형 컨볼루션 적용')
    parser.add_argument('--use_lightweight_lstm', action='store_true', help='경량화 LSTM 사용')
    parser.add_argument('--lightweight_lstm_hidden_size', type=int, default=384, help='경량화 LSTM hidden_size')
    parser.add_argument('--lightweight_lstm_bidirectional', action='store_true', help='경량화 LSTM bidirectional')
    parser.add_argument('--lightweight_lstm_num_layers', type=int, default=1, help='경량화 LSTM num_layers')
    parser.add_argument('--use_lightweight_context', action='store_true', help='경량화 Context Module 사용')
    parser.add_argument('--lightweight_context_hidden_size_ratio', type=float, default=0.5, help='경량화 Context Module hidden_size_ratio')
    parser.add_argument('--no_lightweight_lstm', dest='use_lightweight_lstm', action='store_false', help='경량화 LSTM 비활성화')
    parser.add_argument('--no_lightweight_context', dest='use_lightweight_context', action='store_false', help='경량화 Context Module 비활성화')
    parser.set_defaults(use_depthwise_separable=True, use_lightweight_lstm=True, use_lightweight_context=True)
    
    args = parser.parse_args()
    
    # 마이크 배열 파싱
    mic_array = [int(idx) for idx in args.mic_array.split(',')]
    
    if args.compare:
        # 여러 모델 설정 비교
        models_info = {}
        
        # 4채널 기본 모델
        print("\n=== 4채널 기본 모델 분석 중... ===")
        models_info["4채널 기본"] = analyze_model(4, [0, 4, 8, 12], False, args.use_depthwise_separable, args.depthwise_for_all_blocks, args.use_lightweight_lstm, args.lightweight_lstm_hidden_size, args.lightweight_lstm_bidirectional, args.lightweight_lstm_num_layers, args.use_lightweight_context, args.lightweight_context_hidden_size_ratio)
        
        # 2채널 기본 모델 (최적화 없음)
        print("\n=== 2채널 기본 모델 분석 중... ===")
        models_info["2채널 기본"] = analyze_model(2, [0, 8], False, args.use_depthwise_separable, args.depthwise_for_all_blocks, args.use_lightweight_lstm, args.lightweight_lstm_hidden_size, args.lightweight_lstm_bidirectional, args.lightweight_lstm_num_layers, args.use_lightweight_context, args.lightweight_context_hidden_size_ratio)
        
        # 2채널 최적화 모델
        print("\n=== 2채널 최적화 모델 분석 중... ===")
        models_info["2채널 최적화"] = analyze_model(2, [0, 8], True, args.use_depthwise_separable, args.depthwise_for_all_blocks, args.use_lightweight_lstm, args.lightweight_lstm_hidden_size, args.lightweight_lstm_bidirectional, args.lightweight_lstm_num_layers, args.use_lightweight_context, args.lightweight_context_hidden_size_ratio)
        
        # 모델 비교
        compare_models(models_info)
    else:
        # 단일 모델 분석
        if args.checkpoint:
            # 체크포인트 있을 경우 로드 후 분석
            print(f"\n체크포인트를 로드합니다: {args.checkpoint}")
            try:
                # 설정 생성
                config = TrainingConfig()
                config.microphone_num = args.mic_num
                config.selected_channels = mic_array
                config.use_depthwise_separable = args.use_depthwise_separable
                config.depthwise_for_all_blocks = args.depthwise_for_all_blocks
                config.use_lightweight_lstm = args.use_lightweight_lstm
                config.lightweight_lstm_hidden_size = args.lightweight_lstm_hidden_size
                config.lightweight_lstm_bidirectional = args.lightweight_lstm_bidirectional
                config.lightweight_lstm_num_layers = args.lightweight_lstm_num_layers
                config.use_lightweight_context = args.use_lightweight_context
                config.lightweight_context_hidden_size_ratio = args.lightweight_context_hidden_size_ratio
                
                if args.optimize_two_channel and args.mic_num == 2:
                    config.optimize_for_two_channel = True
                    config.two_channel_indices = tuple(mic_array)
                
                # 모델 초기화
                model = AudioNetV3(
                    sample_num=config.sample_num,
                    microphone_num=config.microphone_num,
                    output_num=config.output_num,
                    config=config
                )
                
                # 체크포인트 로드
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                print("체크포인트 로드 완료")
                
                # 파라미터 분석
                count_parameters(model)
                
                # 결과 출력
                print(f"\n=== AudioNetV3 모델 정보 요약 ({args.mic_num}채널) ===")
                print(f"마이크 채널: {mic_array}")
                print(f"2채널 최적화: {config.optimize_for_two_channel}")
                print(f"분리형 컨볼루션: {config.use_depthwise_separable} (전체 블록: {config.depthwise_for_all_blocks})")
                print(f"체크포인트: {args.checkpoint}")
                print(f"모델 크기: {os.path.getsize(args.checkpoint) / (1024 * 1024):.2f} MB (파일 기준)")
                
                # 모델 구조 출력
                print("\n=== 모델 구조 ===")
                print(model)
            except Exception as e:
                print(f"체크포인트 로드 실패: {str(e)}")
        else:
            # 체크포인트 없이 분석
            analyze_model(args.mic_num, mic_array, args.optimize_two_channel, args.use_depthwise_separable, args.depthwise_for_all_blocks, args.use_lightweight_lstm, args.lightweight_lstm_hidden_size, args.lightweight_lstm_bidirectional, args.lightweight_lstm_num_layers, args.use_lightweight_context, args.lightweight_context_hidden_size_ratio)

if __name__ == "__main__":
    main() 