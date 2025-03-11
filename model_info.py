import os
import torch
import numpy as np
from network import AudioNet
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
    
    # 모듈별 파라미터 수
    print("\n=== 모듈별 파라미터 수 ===")
    module_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            module_name = name.split('.')[0]
            if module_name not in module_params:
                module_params[module_name] = 0
            module_params[module_name] += param.numel()
    
    for module_name, param_count in module_params.items():
        print(f"{module_name}: {param_count:,} ({param_count/trainable_params*100:.2f}%)")
    
    # 모델 크기 계산 (MB)
    model_size_bytes = 0
    for param in model.parameters():
        # 파라미터 크기 (bytes) = 원소 수 * 원소당 바이트 수
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
    
    # 모델 초기화
    model = AudioNet(
        sample_num=config.sample_num,
        microphone_num=config.microphone_num,
        output_num=config.output_num,
        config=config
    )
    
    # 파라미터 수 계산
    param_info = count_parameters(model)
    
    # 결과 출력
    print("\n=== 모델 정보 요약 ===")
    print(f"총 파라미터 수: {param_info['total_params']:,}")
    print(f"학습 가능한 파라미터 수: {param_info['trainable_params']:,}")
    print(f"모델 크기: {param_info['model_size_mb']:.2f} MB")
    
    # 모델 구조 출력
    print("\n=== 모델 구조 ===")
    print(model)
    
    # FLOPS 계산 (선택적)
    try:
        from thop import profile
        dummy_input = torch.randn(1, config.microphone_num, config.sample_num, 20)
        flops, _ = profile(model, inputs=(dummy_input,))
        print(f"\n모델 FLOPS: {flops/1e9:.2f} GFLOPS")
    except ImportError:
        print("\nFLOPS 계산을 위해 thop 패키지를 설치하세요: pip install thop")

if __name__ == "__main__":
    main() 