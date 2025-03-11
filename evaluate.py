import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import math
import time
import os

def add_noise_to_batch(inputs, noise_level=0.001):
    """테스트용 노이즈 추가 함수"""
    # 입력 신호의 RMS 계산
    rms = torch.sqrt(torch.mean(inputs ** 2))
    
    # SNR 기반 노이즈 레벨 조정
    noise = torch.randn_like(inputs)
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    scaling_factor = rms * noise_level / noise_rms
    
    # 스케일된 노이즈 추가
    noisy_inputs = inputs + (noise * scaling_factor)
    
    # SNR 계산 (진행 바에 추가)
    snr = 20 * torch.log10(rms / (noise_rms * scaling_factor))
    
    return noisy_inputs, snr.item()

def quaternion_angle_difference(quat_pred, quat_true):
    """쿼터니언 각도 차이 계산 (양방향 고려)"""
    eps = 1e-6
    quat_pred = F.normalize(quat_pred, p=2, dim=1)
    quat_true = F.normalize(quat_true, p=2, dim=1)
    
    dot_product_pos = torch.sum(quat_pred * quat_true, dim=1)
    dot_product_neg = torch.sum(quat_pred * (-quat_true), dim=1)
    
    dot_product = torch.where(
        torch.abs(dot_product_pos) > torch.abs(dot_product_neg),
        dot_product_pos,
        dot_product_neg
    )
    
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
    angle_difference = 2 * torch.acos(torch.abs(dot_product))
    return torch.rad2deg(angle_difference)

def evaluate_model(model, test_loader, device, config):
    """모델 평가 함수"""
    model.eval()
    
    # SNR 직접 지정 (dB 단위)
    target_snrs = [
        40,     # 거의 원본 수준 (매우 조용한 환경)
        20,     # 조용한 사무실 정도
        10,     # 일반적인 실내 환경
        0,      # 시끄러운 환경
        -10,     # 매우 시끄러운 환경
        -20
    ]
    
    # SNR을 노이즈 레벨로 변환
    noise_levels = [10 ** (-snr/20) for snr in target_snrs]
    results_by_noise = {}
    
    # Latency 측정을 위한 변수 초기화
    total_latency = 0.0
    total_samples = 0
    latencies = []  # 각 샘플별 latency를 저장할 리스트
    
    for snr, noise_level in zip(target_snrs, noise_levels):
        print(f"\n=== 목표 SNR {snr}dB 테스트 ===")
        
        env_metrics = {
            'same_user_same_space': {
                'distance_errors': [],
                'angle_errors': [],
                'loss': 0,
                'samples': 0,
                'latencies': []  # 환경별 latency 저장
            },
            'diff_user_same_space': {
                'distance_errors': [],
                'angle_errors': [],
                'loss': 0,
                'samples': 0,
                'latencies': []  # 환경별 latency 저장
            },
            'diff_user_diff_space': {
                'distance_errors': [],
                'angle_errors': [],
                'loss': 0,
                'samples': 0,
                'latencies': []  # 환경별 latency 저장
            }
        }
        
        test_loader_tqdm = tqdm(enumerate(test_loader), 
                               desc=f'SNR {snr}dB', 
                               total=len(test_loader), 
                               unit='batch')
        
        total_processed = 0
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=config.use_amp):
                for batch_idx, (inputs, (pos_target, rot_target)) in test_loader_tqdm:
                    batch_size = inputs.size(0)
                    batch_latencies = []  # 현재 배치의 각 샘플별 latency를 저장
                    
                    # 노이즈 추가
                    if noise_level > 0:
                        inputs, current_snr = add_noise_to_batch(inputs, noise_level)
                    
                    # 데이터를 디바이스로 이동
                    inputs = inputs.to(device)
                    pos_target = pos_target.to(device)
                    rot_target = rot_target.to(device)
                    
                    # 각 샘플별로 latency 측정
                    for i in range(batch_size):
                        single_input = inputs[i:i+1]  # 단일 샘플 선택
                        
                        # 추론 시작 시간 기록
                        start_time = time.time()
                        
                        # 모델 추론
                        pos_pred_single, rot_pred_single = model(single_input)
                        
                        # GPU 연산 완료 대기
                        torch.cuda.synchronize()
                        
                        # 추론 종료 시간 기록
                        end_time = time.time()
                        
                        # Latency 계산 (밀리초 단위)
                        latency = (end_time - start_time) * 1000
                        batch_latencies.append(latency)
                        latencies.append(latency)
                        
                        total_latency += latency
                        total_samples += 1
                    
                    # 전체 배치에 대한 모델 추론 (메트릭 계산용)
                    pos_pred, rot_pred = model(inputs)
                    
                    # 세션 ID 찾기
                    session_ids = []
                    for i in range(batch_size):
                        global_idx = total_processed + i
                        if global_idx >= len(test_loader.dataset):
                            break
                            
                        real_idx = test_loader.dataset.indices[global_idx]
                        for sess_id, session in enumerate(test_loader.dataset.starts):
                            if session['start'] <= real_idx < session['end']:
                                session_ids.append(sess_id)
                                break
                    
                    total_processed += batch_size
                    
                    # 각 샘플별로 환경 분류하여 메트릭 계산
                    for i in range(len(session_ids)):
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
                        env_metrics[env]['latencies'].append(batch_latencies[i])  # 환경별 latency 저장
                    
                    # 진행 바 업데이트
                    mean_dist = np.mean([err for env in env_metrics.values() for err in env['distance_errors']])
                    mean_ang = np.mean([err for env in env_metrics.values() for err in env['angle_errors']])
                    mean_latency = np.mean(batch_latencies)
                    test_loader_tqdm.set_postfix(
                        mae=f"{mean_dist:.3f}m",
                        angle=f"{mean_ang:.2f}°",
                        snr=f"{current_snr:.1f}dB" if noise_level > 0 else "Clean",
                        latency=f"{mean_latency:.1f}ms"
                    )
        
        results_by_noise[noise_level] = env_metrics
    
    # 전체 latency 통계 계산
    avg_latency = total_latency / total_samples
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print("\n=== Latency 통계 ===")
    print(f"평균 추론 시간: {avg_latency:.2f}ms")
    print(f"표준 편차: {std_latency:.2f}ms")
    print(f"최소 추론 시간: {min_latency:.2f}ms")
    print(f"최대 추론 시간: {max_latency:.2f}ms")
    print(f"95퍼센타일: {p95_latency:.2f}ms")
    print(f"99퍼센타일: {p99_latency:.2f}ms")
    
    # latency 정보를 결과에 추가
    results_by_noise['latency_stats'] = {
        'avg': avg_latency,
        'std': std_latency,
        'min': min_latency,
        'max': max_latency,
        'p95': p95_latency,
        'p99': p99_latency,
        'all_latencies': latencies
    }
    
    return results_by_noise

def print_evaluation_results(results_by_noise, config):
    """평가 결과 출력"""
    paper_metrics = {
        'same_user_same_space': {'distance': 0.31, 'angle': 34.3},
        'diff_user_same_space': {'distance': 0.33, 'angle': 35.1},
        'diff_user_diff_space': {'distance': 0.35, 'angle': 36.2}
    }
    
    target_thresholds = {'distance': 0.35, 'angle': 25.0}
    
    # Latency 통계 출력
    if 'latency_stats' in results_by_noise:
        stats = results_by_noise.pop('latency_stats')  # latency_stats를 제거하고 나머지 결과만 처리
        print("\n=== 전체 Latency 통계 ===")
        print(f"├─ 평균 추론 시간: {stats['avg']:.2f}ms")
        print(f"├─ 표준 편차: {stats['std']:.2f}ms")
        print(f"├─ 최소 추론 시간: {stats['min']:.2f}ms")
        print(f"├─ 최대 추론 시간: {stats['max']:.2f}ms")
        print(f"├─ 95퍼센타일: {stats['p95']:.2f}ms")
        print(f"└─ 99퍼센타일: {stats['p99']:.2f}ms")
    
    for noise_level, results in results_by_noise.items():
        print(f"\n=== 노이즈 레벨 {noise_level} 결과 ===")
        for env, metrics in results.items():
            if metrics['samples'] == 0:
                continue
            
            avg_distance = np.mean(metrics['distance_errors'])
            avg_angle = np.mean(metrics['angle_errors'])
            rmse_distance = np.sqrt(np.mean(np.array(metrics['distance_errors']) ** 2))
            
            within_dist = sum(d <= target_thresholds['distance'] for d in metrics['distance_errors'])
            within_angle = sum(a <= target_thresholds['angle'] for a in metrics['angle_errors'])
            
            # 환경별 latency 통계 계산
            if 'latencies' in metrics and metrics['latencies']:
                env_latencies = metrics['latencies']
                avg_latency = np.mean(env_latencies)
                std_latency = np.std(env_latencies)
                p95_latency = np.percentile(env_latencies, 95)
            
            print(f"\n{env}:")
            print(f"├─ 샘플 수: {metrics['samples']}")
            print(f"├─ 거리 오차:")
            print(f"│  ├─ MAE: {avg_distance:.3f}m (논문: {paper_metrics[env]['distance']}m)")
            print(f"│  ├─ RMSE: {rmse_distance:.3f}m")
            print(f"│  └─ 목표 달성률 (<{target_thresholds['distance']}m): {within_dist/metrics['samples']:.2%}")
            print(f"├─ 각도 오차:")
            print(f"│  ├─ 평균: {avg_angle:.2f}° (논문: {paper_metrics[env]['angle']}°)")
            print(f"│  └─ 목표 달성률 (<{target_thresholds['angle']}°): {within_angle/metrics['samples']:.2%}")
            if 'latencies' in metrics and metrics['latencies']:
                print(f"└─ Latency:")
                print(f"   ├─ 평균: {avg_latency:.2f}ms")
                print(f"   ├─ 표준편차: {std_latency:.2f}ms")
                print(f"   └─ 95퍼센타일: {p95_latency:.2f}ms")

def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device, criterion_tdoa=None):
    test_filelist = data_generator.get_filelist()
    latency_list = []
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for values in data_generator.generate():
            start_time = time.time()
            if len(values) == 2:
                data, target = values
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
                # 노이즈 강건성 테스트 주석 처리
                # noisy_data, snr = add_noise_to_batch(data, noise_level)
                output = model(data)
            elif len(values) == 3:
                data, vid_feat, target = values
                data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
                # 노이즈 강건성 테스트 주석 처리
                # noisy_data, snr = add_noise_to_batch(data, noise_level)
                output = model(data, vid_feat)
            end_time = time.time()
            latency = end_time - start_time
            latency_list.append(latency * 1000)  # Convert to milliseconds

            loss = criterion(output, target)
            test_loss += loss.item()
            nb_test_batches += 1

            # Write output to file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            data_generator.write_output_format_file(output_file, output_dict)

        test_loss /= nb_test_batches
        print("Average latency: {}ms".format(np.mean(latency_list)), flush=True)
    return test_loss