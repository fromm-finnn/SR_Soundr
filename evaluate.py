import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import math

def quaternion_angle_difference(quat_pred, quat_true):
    """쿼터니언 각도 차이 계산 (양방향 고려)"""
    eps = 1e-6
    quat_pred = F.normalize(quat_pred, p=2, dim=1)
    quat_true = F.normalize(quat_true, p=2, dim=1)
    
    # 양방향 모두 고려
    dot_product_pos = torch.sum(quat_pred * quat_true, dim=1)
    dot_product_neg = torch.sum(quat_pred * (-quat_true), dim=1)
    
    # 더 작은 각도를 주는 방향 선택
    dot_product = torch.where(
        torch.abs(dot_product_pos) > torch.abs(dot_product_neg),
        dot_product_pos,
        dot_product_neg
    )
    
    # 안전한 범위로 클램핑
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
    
    # 각도 계산 (라디안 -> 도)
    angle_difference = 2 * torch.acos(torch.abs(dot_product))
    return torch.rad2deg(angle_difference)

def evaluate_model(model, test_loader, device, config):
    """모델 평가 함수"""
    model.eval()
    
    # 환경별 메트릭 저장용 딕셔너리
    env_metrics = {
        'same_user_same_space': {
            'distance_errors': [],
            'angle_errors': [],
            'loss': 0,
            'samples': 0
        },
        'diff_user_same_space': {
            'distance_errors': [],
            'angle_errors': [],
            'loss': 0,
            'samples': 0
        },
        'diff_user_diff_space': {
            'distance_errors': [],
            'angle_errors': [],
            'loss': 0,
            'samples': 0
        }
    }
    
    print("\n테스트 진행 중...")
    test_loader_tqdm = tqdm(enumerate(test_loader), desc='Testing', total=len(test_loader), unit='batch')
    
    total_processed = 0
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            for batch_idx, (inputs, (pos_target, rot_target)) in test_loader_tqdm:
                # 데이터를 디바이스로 이동
                inputs = inputs.to(device)
                pos_target = pos_target.to(device)
                rot_target = rot_target.to(device)
                
                # 모델 추론
                pos_pred, rot_pred = model(inputs)
                
                # 실제 배치 크기 계산
                batch_size = pos_pred.size(0)
                
                # 전역 인덱스 계산으로 세션 찾기
                session_ids = []
                for i in range(batch_size):
                    global_idx = total_processed + i
                    if global_idx >= len(test_loader.dataset):
                        break
                        
                    real_idx = test_loader.dataset.indices[global_idx]
                    # 해당 real_idx가 속한 세션 찾기
                    for sess_id, session in enumerate(test_loader.dataset.starts):
                        if session['start'] <= real_idx < session['end']:
                            session_ids.append(sess_id)
                            break
                
                total_processed += batch_size
                
                # 각 샘플별로 환경 분류하여 메트릭 계산
                for i in range(len(session_ids)):  # session_ids의 길이만큼만 처리
                    sess_id = session_ids[i]
                    
                    # 환경 확인
                    if sess_id in test_loader.dataset.environment_sessions['same_user_same_space']:
                        env = 'same_user_same_space'
                    elif sess_id in test_loader.dataset.environment_sessions['diff_user_same_space']:
                        env = 'diff_user_same_space'
                    else:
                        env = 'diff_user_diff_space'
                    
                    # 개별 샘플의 오차 계산
                    distance_error = torch.norm(pos_pred[i] - pos_target[i]).item()
                    
                    # 각도 오차
                    rot_pred_i = F.normalize(rot_pred[i:i+1], p=2, dim=1)
                    rot_target_i = F.normalize(rot_target[i:i+1], p=2, dim=1)
                    angle_error = quaternion_angle_difference(rot_pred_i, rot_target_i).item()
                    
                    # 환경별 메트릭 저장
                    env_metrics[env]['distance_errors'].append(distance_error)
                    env_metrics[env]['angle_errors'].append(angle_error)
                    env_metrics[env]['samples'] += 1
                
                # 진행 바 업데이트
                mean_dist = np.mean([err for env in env_metrics.values() for err in env['distance_errors']])
                mean_ang = np.mean([err for env in env_metrics.values() for err in env['angle_errors']])
                test_loader_tqdm.set_postfix(
                    mae=f"{mean_dist:.3f}m",
                    angle=f"{mean_ang:.2f}°"
                )
    
    return env_metrics
    
def print_evaluation_results(results, config):
    """평가 결과 출력"""
    paper_metrics = {
        'same_user_same_space': {'distance': 0.31, 'angle': 34.3},
        'diff_user_same_space': {'distance': 0.33, 'angle': 35.1},
        'diff_user_diff_space': {'distance': 0.35, 'angle': 36.2}
    }
    
    target_thresholds = {'distance': 0.35, 'angle': 25.0}
    
    print("\n=== 환경별 테스트 결과 ===")
    for env, metrics in results.items():
        if metrics['samples'] == 0:
            continue
        
        avg_distance = np.mean(metrics['distance_errors'])
        avg_angle = np.mean(metrics['angle_errors'])
        rmse_distance = np.sqrt(np.mean(np.array(metrics['distance_errors']) ** 2))
        
        within_dist = sum(d <= target_thresholds['distance'] for d in metrics['distance_errors'])
        within_angle = sum(a <= target_thresholds['angle'] for a in metrics['angle_errors'])
        
        print(f"\n{env}:")
        print(f"├─ 샘플 수: {metrics['samples']}")
        print(f"├─ 거리 오차:")
        print(f"│  ├─ MAE: {avg_distance:.3f}m (논문: {paper_metrics[env]['distance']}m)")
        print(f"│  ├─ RMSE: {rmse_distance:.3f}m")
        print(f"│  └─ 목표 달성률 (<{target_thresholds['distance']}m): {within_dist/metrics['samples']:.2%}")
        print(f"└─ 각도 오차:")
        print(f"   ├─ 평균: {avg_angle:.2f}° (논문: {paper_metrics[env]['angle']}°)")
        print(f"   └─ 목표 달성률 (<{target_thresholds['angle']}°): {within_angle/metrics['samples']:.2%}")