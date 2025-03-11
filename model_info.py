import os
import torch
import numpy as np
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
        "model_size_mb": model_size_mb
    }

def main():
    # 설정 로드
    config = TrainingConfig()
    
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
    print("\n=== AudioNetV3 모델 정보 요약 ===")
    print(f"총 파라미터 수: {param_info['total_params']:,}")
    print(f"학습 가능한 파라미터 수: {param_info['trainable_params']:,}")
    print(f"모델 크기: {param_info['model_size_mb']:.2f} MB")
    
    # 모델 구조 출력
    print("\n=== 모델 구조 ===")
    print(model)
    
    # 모델 설정 정보 출력
    print("\n=== 모델 설정 정보 ===")
    print(f"Channel Attention Reduction Ratio: {model.channel_attention_reduction_ratio}")
    print(f"Self Attention Dropout Rate: {model.self_attention_dropout}")
    print(f"Kernel Sizes: {model.kernel_sizes}")
    print(f"Context Module 사용: {model.use_context_module}")
    print(f"Position-Rotation 연결 사용: {model.use_position_for_rotation}")
    print(f"2채널 최적화: {model.optimize_for_two_channel}")
    
    # FLOPS 계산 (선택적)
    try:
        from thop import profile
        # 입력 텐서 형태: (batch_size, channels, samples)
        dummy_input = torch.randn(1, config.microphone_num, config.sample_num)
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            print(f"\n모델 FLOPS: {flops/1e9:.2f} GFLOPS")
    except Exception as e:
        print(f"\nFLOPS 계산 중 오류 발생: {str(e)}")
        print("FLOPS 계산을 위해 thop 패키지를 설치하세요: pip install thop")

if __name__ == "__main__":
    main() 