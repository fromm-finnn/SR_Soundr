import torch
from network import AudioNet
import os

def count_parameters(model):
    """학습 가능한 파라미터 수를 계산"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def get_model_size(model):
    """모델 크기를 MB 단위로 계산"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def analyze_model(checkpoint_path):
    print(f"\n체크포인트 파일 크기: {os.path.getsize(checkpoint_path) / 1024**2:.2f} MB")
    
    # 모델 생성 및 체크포인트 로드
    model = AudioNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 파라미터 수 계산
    total_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print("\n=== 모델 파라미터 분석 ===")
    print(f"총 파라미터 수: {total_params:,}")
    print(f"모델 메모리 사용량: {model_size:.2f} MB")
    
    print("\n=== 레이어별 파라미터 수 ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/20241219T085405/model_epoch_36_best_angle.pth"  # 체크포인트 경로
    if not os.path.exists(checkpoint_path):
        print(f"Error: 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        exit(1)
        
    analyze_model(checkpoint_path)