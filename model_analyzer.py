import torch
import os
import numpy as np
from network import AudioNet

def format_size(size_bytes):
    """바이트 크기를 사람이 읽기 쉬운 형태로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

class Config:
    """AudioNet 설정을 위한 임시 클래스"""
    def __init__(self):
        self.use_amp = True
        self.sequence_length = 1
        self.dropout_rate = 0.5
        self.warmup_epochs = 5
        self.rotation_ramp_epochs = 10
        self.position_loss_weight = 1.0
        self.initial_rotation_weight = 0.0
        self.final_rotation_weight = 1.0

def analyze_model(model_path):
    """모델 분석"""
    print(f"\n=== 모델 분석: {os.path.basename(model_path)} ===\n")
    
    # 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()
    
    model = AudioNet(
        sample_num=1200,  # 25ms * 48000Hz
        microphone_num=2,
        output_num=7,     # position(3) + quaternion(4)
        config=config
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 1. 전체 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("파라미터 수:")
    print(f"- 전체: {total_params:,}")
    print(f"- 학습 가능: {trainable_params:,}")
    print(f"- 고정: {total_params - trainable_params:,}")
    
    # 2. 모델 크기
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    print("\n모델 크기:")
    print(f"- 파라미터: {format_size(param_size)}")
    print(f"- 버퍼: {format_size(buffer_size)}")
    print(f"- 전체: {format_size(total_size)}")
    
    # 3. 레이어별 분석
    print("\n레이어별 분석:")
    print(f"{'레이어':50} {'파라미터 수':>15} {'크기':>10}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_size = param.nelement() * param.element_size()
        print(f"{name:50} {param_count:15,} {format_size(param_size):>10}")
    
    # 4. 레이어 타입별 통계
    layer_types = {}
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        if layer_type not in ['Sequential', 'AudioNet']:  # 컨테이너 제외
            if layer_type not in layer_types:
                layer_types[layer_type] = {
                    'count': 0,
                    'params': 0
                }
            layer_types[layer_type]['count'] += 1
            layer_types[layer_type]['params'] += sum(p.numel() for p in module.parameters())
    
    print("\n레이어 타입별 통계:")
    print(f"{'타입':20} {'개수':>10} {'파라미터 수':>15} {'비율':>10}")
    print("-" * 60)
    
    for layer_type, info in layer_types.items():
        ratio = info['params'] / total_params * 100
        print(f"{layer_type:20} {info['count']:10d} {info['params']:15,} {ratio:9.2f}%")

def main():
    model_path = "./checkpoints/dov_soundr_2ch_lr5e5/model_epoch_100_best_composite.pth"
    analyze_model(model_path)

if __name__ == "__main__":
    main()