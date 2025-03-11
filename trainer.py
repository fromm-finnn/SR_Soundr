# trainer.py

import os
import datetime as dt
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import seaborn as sns
import torch.nn.functional as F
import kornia
from network import AudioNet, AudioNetV3  # AudioNetV3 추가

eps = 1e-6

class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.max_epochs = config.num_epochs
        
        # 모델 타입 확인
        self.is_v3_model = isinstance(model, AudioNetV3)
        print(f"모델 타입: {'AudioNetV3' if self.is_v3_model else 'AudioNet'}")

        # Mixed Precision 설정
        self.scaler = torch.amp.GradScaler(
            init_scale=2**10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=100,
            enabled=config.use_amp
        )
        self.use_amp = config.use_amp

        # 손실 함수 및 옵티마이저 설정
        self.pos_criterion = nn.MSELoss()
        self.optimizer = getattr(torch.optim, config.optimizer)(
            self.model.parameters(),
            **config.get_optimizer_params()
        )

        # 학습률 스케줄러
        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01
            )

        # Tensorboard 설정
        self.writer = SummaryWriter(os.path.join(config.checkpoint_dir, "logs"))
        self.n_iter = 0
        self.min_avg_loss = math.inf

        # 최고 성능 지표 초기화 추가
        self.best_angle_acc = 0.0
        self.best_composite_score = 0.0

        # 1. 우리의 목표 임계값
        self.target_thresholds = {
            'distance': 0.35,  # 거리 오차 목표: ≤ 0.35m
            'angle': 25.0      # 각도 오차 목표: ≤ 25도
        }
        
        # 2. 논문의 환경별 평균 오차
        self.paper_metrics = {
            'same_user_same_space': {
                'distance': 0.31,
                'angle': 34.3
            },
            'diff_user_same_space': {
                'distance': 0.33,
                'angle': 40.0
            },
            'diff_user_diff_space': {
                'distance': 0.57,
                'angle': 57.0
            }
        }
        
        self.angle_loss_weight = config.rotation_loss_weight 

        # 점진적 학습 관련 설정 추가
        self.warmup_epochs = config.warmup_epochs
        self.rotation_ramp_epochs = config.rotation_ramp_epochs
        self.initial_rotation_weight = config.initial_rotation_weight
        self.final_rotation_weight = config.final_rotation_weight
        self.position_loss_weight = config.position_loss_weight

    
        self.history = {
            # 기존 메트릭
            'train_loss': [], 
            'train_mae': [], 
            'train_rmse': [], 
            'train_angle_error': [],
            'val_loss': [], 
            'val_mae': [], 
            'val_rmse': [], 
            'val_angle_error': [],
            
            # 추가 메트릭
            'train_pos_loss': [],
            'train_quat_loss': [],
            'val_pos_loss': [],
            'val_quat_loss': [],
            'train_pos_acc': [],
            'train_angle_acc': [],
            'val_pos_acc': [],
            'val_angle_acc': [],
            'composite_score': [],
            
            # 환경별 메트릭 추가
            'env_metrics': {
                'same_user_same_space': {
                    'distance_errors': [],
                    'angle_errors': [],
                    'latencies': []
                },
                'diff_user_same_space': {
                    'distance_errors': [],
                    'angle_errors': [],
                    'latencies': []
                },
                'diff_user_diff_space': {
                    'distance_errors': [],
                    'angle_errors': [],
                    'latencies': []
                }
            }
        }
        
        # 시작 에포크 설정
        self.start_epoch = 1

    # 각도 오차 계산 함수 구현 
    def quaternion_angle_difference(self, q1, q2):
        # 양방향 모두 고려
        dot_product_pos = torch.sum(q1 * q2, dim=1)
        dot_product_neg = torch.sum(q1 * (-q2), dim=1)
        
        # 더 작은 각도를 주는 방향 선택
        dot_product = torch.where(
            torch.abs(dot_product_pos) > torch.abs(dot_product_neg),
            dot_product_pos,
            dot_product_neg
        )
        
        # 안전한 범위로 클램핑
        dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
        
        # 각도 계산
        angle_difference = 2 * torch.acos(torch.abs(dot_product))
        return torch.rad2deg(angle_difference)

    def criterion(self, outputs, references):
        """
        Args:
            outputs: (position_pred, rotation_pred) 튜플
            references: (position_target, rotation_target) 튜플
        """
        # 위치와 회전 분리 (이미 튜플로 받음)
        pos_pred, quat_pred = outputs
        pos_target, quat_target = references
        
        # 위치 손실
        pos_loss = self.pos_criterion(pos_pred, pos_target)
        
        # 쿼터니온 정규화
        quat_pred = F.normalize(quat_pred, p=2, dim=1)
        quat_target = F.normalize(quat_target, p=2, dim=1)
        
        # 양방향 고려
        dot_product_pos = torch.sum(quat_pred * quat_target, dim=1)
        dot_product_neg = torch.sum(quat_pred * (-quat_target), dim=1)
        
        # 더 작은 각도 선택
        dot_product = torch.where(
            torch.abs(dot_product_pos) > torch.abs(dot_product_neg),
            dot_product_pos,
            dot_product_neg
        )
        
        quat_loss = 1 - torch.abs(dot_product).mean()
        
        # 점진적 학습을 위한 손실 가중치 조정
        if hasattr(self, 'current_epoch'):
            if self.current_epoch < self.warmup_epochs:
                # warmup 기간에는 위치 학습에만 집중
                loss = self.position_loss_weight * pos_loss
                current_rotation_weight = 0.0
            else:
                # warmup 이후 회전 손실 점진적 도입
                progress = min(1.0, (self.current_epoch - self.warmup_epochs) / 
                             self.rotation_ramp_epochs)
                current_rotation_weight = (self.final_rotation_weight - 
                                        self.initial_rotation_weight) * progress + \
                                        self.initial_rotation_weight
                
                loss = self.position_loss_weight * pos_loss + \
                       current_rotation_weight * quat_loss
        else:
            # epoch 정보가 없는 경우 (검증 등)
            current_rotation_weight = self.final_rotation_weight
            loss = self.position_loss_weight * pos_loss + \
                   current_rotation_weight * quat_loss
        
        return loss, pos_loss, quat_loss
    
    def train(self):
        # 타임스탬프 디렉토리 설정
        self.timestamp_dir = os.path.join(
            self.config.checkpoint_dir, 
            self.config.experiment_name
        )
        os.makedirs(self.timestamp_dir, exist_ok=True)
        
        # 시작 에포크 설정 (체크포인트에서 복원된 값 사용)
        start_epoch = getattr(self, 'start_epoch', 1)
            
        for epoch in range(start_epoch, self.max_epochs + 1):
            try:
                # self.model.set_epoch(epoch)  # 현재 epoch 정보 모델에 전달
                self.current_epoch = epoch  # 현재 epoch 정보 저장

                epoch_start_time = time.time()
                print(f"\n{'='*100}")
                print(f"Epoch {epoch}/{self.max_epochs}".center(100))
                print('='*100)
                    
                self.model.train()
                running_loss = 0.0

                # 점진적 학습 상태 출력
                print("\nProgressive Learning Status:")
                if epoch < self.warmup_epochs:
                    print(f"Phase: Position only (Epoch {epoch}/{self.warmup_epochs})")
                    current_rotation_weight = 0.0
                else:
                    progress = min(1.0, (epoch - self.warmup_epochs) / self.rotation_ramp_epochs)
                    current_rotation_weight = (self.final_rotation_weight - self.initial_rotation_weight) * progress + \
                                            self.initial_rotation_weight
                    print(f"Phase: Position + Rotation (Progress: {progress:.2%})")
                    print(f"Current rotation weight: {current_rotation_weight:.4f}")
                    
                # 지표 초기화
                total_distance_error = 0.0
                total_distance_error_sq = 0.0
                total_angle_error = 0.0
                num_within_target_distance = 0
                num_within_target_angle = 0
                total_samples = 0

                # tqdm wrapper 생성
                train_loader_tqdm = tqdm(self.train_loader, 
                                        desc='Training', 
                                        unit='batch',
                                        dynamic_ncols=True)

                for inputs, (pos_target, rot_target) in train_loader_tqdm:
                    batch_size = inputs.size(0)
                        
                    # 디바이스로 이동
                    inputs = inputs.to(self.device)
                    pos_target = pos_target.to(self.device)
                    rot_target = rot_target.to(self.device)
                        
                    self.optimizer.zero_grad()
                        
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pos_pred, rot_pred = self.model(inputs)
                        loss, pos_loss, rot_loss = self.criterion(
                            (pos_pred, rot_pred),
                            (pos_target, rot_target)
                        )

                    # Gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    running_loss += loss.item() * batch_size
                    self.n_iter += 1
                    total_samples += batch_size

                    # 배치별 메트릭 계산
                    with torch.no_grad():
                        # 위치 오차 계산
                        distance_errors = torch.norm(pos_pred - pos_target, dim=1)
                        mean_distance_error = distance_errors.mean().item()
                        
                        # 각도 오차 계산
                        angle_diff = self.quaternion_angle_difference(
                            F.normalize(rot_pred, p=2, dim=1),
                            F.normalize(rot_target, p=2, dim=1)
                        )
                        mean_angle_error = angle_diff.mean().item()
                        
                        # 누적 메트릭 업데이트
                        total_distance_error += distance_errors.sum().item()
                        total_distance_error_sq += (distance_errors ** 2).sum().item()
                        total_angle_error += angle_diff.sum().item()
                        
                        # 목표 달성률 계산
                        num_within_target_distance += (distance_errors <= self.target_thresholds['distance']).sum().item()
                        num_within_target_angle += (angle_diff <= self.target_thresholds['angle']).sum().item()

                    # 진행 바 업데이트
                    train_loader_tqdm.set_postfix(
                        loss=loss.item(),
                        pos_loss=pos_loss.item(),
                        rot_loss=rot_loss.item(),
                        mae=f"{mean_distance_error:.2f}m",
                        angle_error=f"{mean_angle_error:.2f}°"
                    )
                
                # 에폭 종료 시 지표 계산
                epoch_loss = running_loss / len(self.train_loader.dataset)
                mae_distance_error = total_distance_error / total_samples
                rmse_distance_error = math.sqrt(total_distance_error_sq / total_samples)
                avg_angle_error = total_angle_error / total_samples
                accuracy_target_distance = num_within_target_distance / total_samples
                accuracy_target_angle = num_within_target_angle / total_samples

                # 메트릭 저장
                train_metrics = {
                    'train_loss': epoch_loss,
                    'train_mae': mae_distance_error,
                    'train_rmse': rmse_distance_error,
                    'train_angle_error': avg_angle_error,
                    'train_target_dist_acc': accuracy_target_distance,
                    'train_target_angle_acc': accuracy_target_angle,
                    'train_paper_dist_ratio': mae_distance_error / self.paper_metrics['same_user_same_space']['distance'],
                    'train_paper_angle_ratio': avg_angle_error / self.paper_metrics['same_user_same_space']['angle'],
                    'current_rotation_weight': current_rotation_weight  # 현재 회전 가중치 추가
                }

                # Training 결과 출력
                print("\n[Training Results]")
                print(f"├─ Loss: {epoch_loss:.4f}")
                print(f"├─ Current Rotation Weight: {current_rotation_weight:.4f}")

                print("\n1. Distance Metrics")
                print(f"├─ Error Measurements")
                print(f"│  ├─ MAE: {mae_distance_error:.4f}m (Paper: {self.paper_metrics['same_user_same_space']['distance']}m)")
                print(f"│  └─ RMSE: {rmse_distance_error:.4f}m")
                print(f"└─ Target Achievement")
                print(f"   └─ Success Rate (<{self.target_thresholds['distance']}m): {accuracy_target_distance:.2%}")

                print("\n2. Angle Metrics")
                print(f"├─ Error Measurements")
                print(f"│  ├─ Mean Error: {avg_angle_error:.2f}° (Paper: {self.paper_metrics['same_user_same_space']['angle']}°)")
                print(f"│  └─ Error Ratio: {(avg_angle_error/self.paper_metrics['same_user_same_space']['angle']):.2%} of paper")
                print(f"└─ Target Achievement")
                print(f"   └─ Success Rate (<{self.target_thresholds['angle']}°): {accuracy_target_angle:.2%}")

                # 검증 수행
                val_metrics = self.validate(epoch)
                
                self.save_metrics(epoch, train_metrics, val_metrics)

                # 학습률 스케줄러 업데이트 - 매 에포크마다 실행
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()

                # 복합 점수 계산 및 체크포인트 저장 - 매 에포크마다 실행
                composite_score = (
                    0.3 * val_metrics['val_target_dist_acc'] +
                    0.7 * val_metrics['val_target_angle_acc']
                )

                # 최고 점수 갱신 여부 확인
                is_best_composite = composite_score > self.best_composite_score
                is_best_angle = val_metrics['val_target_angle_acc'] > self.best_angle_acc

                # 최고 점수 업데이트
                if is_best_composite:
                    self.best_composite_score = composite_score
                if is_best_angle:
                    self.best_angle_acc = val_metrics['val_target_angle_acc']

                # 체크포인트 저장 - 매 에포크마다 실행
                self.save_checkpoint(
                    epoch=epoch,
                    composite_score=composite_score,
                    val_metrics=val_metrics,
                    is_best_composite=is_best_composite,
                    is_best_angle=is_best_angle
                )
                
            except Exception as e:
                print(f"\n[ERROR] 에포크 {epoch}에서 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
                break

        self.writer.close()
        print("Training completed.")

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        latencies = []  # latency 저장용 리스트
            
        # 환경별 메트릭 저장용 딕셔너리
        env_metrics = {
            'same_user_same_space': {
                'distance_errors': [],
                'angle_errors': [],
                'loss': 0,
                'samples': 0,
                'latencies': []
            },
            'diff_user_same_space': {
                'distance_errors': [],
                'angle_errors': [],
                'loss': 0,
                'samples': 0,
                'latencies': []
            },
            'diff_user_diff_space': {
                'distance_errors': [],
                'angle_errors': [],
                'loss': 0,
                'samples': 0,
                'latencies': []
            }
        }

        val_loader_tqdm = tqdm(self.val_loader, desc='Validation', unit='batch')
        
        # Warmup을 위한 더미 추론
        dummy_input = torch.randn(1, self.config.microphone_num, self.config.sample_num, 20, 
                                device=self.device)  # 4차원 텐서로 생성 (batch, channels, samples, seq_len)
        for _ in range(10):  # 10회 워밍업
            with torch.no_grad():
                self.model(dummy_input)
        torch.cuda.synchronize()  # GPU 동기화
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                for batch_idx, (inputs, (pos_target, rot_target)) in enumerate(val_loader_tqdm):
                    batch_size = inputs.size(0)
                    
                    # 데이터를 GPU로 이동
                    inputs = inputs.to(self.device)
                    pos_target = pos_target.to(self.device)
                    rot_target = rot_target.to(self.device)
                    
                    batch_latencies = []  # 배치 내 각 샘플의 latency 저장
                    
                    # 각 샘플별로 개별 추론 및 latency 측정
                    for i in range(batch_size):
                        single_input = inputs[i:i+1]  # 단일 샘플 선택 (이미 4차원 텐서)
                        torch.cuda.synchronize()  # GPU 동기화
                        start_time = time.time()
                        
                        # 단일 샘플 추론
                        single_pos_pred, single_rot_pred = self.model(single_input)
                        
                        torch.cuda.synchronize()  # GPU 동기화
                        end_time = time.time()
                        
                        # Latency 계산 (밀리초 단위)
                        latency = (end_time - start_time) * 1000
                        batch_latencies.append(latency)
                        
                        if i == 0:  # 배치의 첫 번째 샘플로 전체 예측 텐서 초기화
                            pos_pred = torch.zeros(batch_size, single_pos_pred.size(1), 
                                                device=self.device)
                            rot_pred = torch.zeros(batch_size, single_rot_pred.size(1), 
                                                device=self.device)
                        
                        # 예측 결과 저장
                        pos_pred[i] = single_pos_pred.squeeze(0)
                        rot_pred[i] = single_rot_pred.squeeze(0)
                    
                    # 전체 배치에 대한 평균 latency 계산
                    latencies.extend(batch_latencies)
                    
                    # 손실 계산 - 튜플 형태로 전달
                    loss, pos_loss, rot_loss = self.criterion(
                        (pos_pred, rot_pred),
                        (pos_target, rot_target)
                    )
                    
                    running_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    # 배치의 세션 ID 가져오기
                    batch_session_ids = self.val_loader.dataset.get_session_ids(batch_idx, batch_size)
                    
                    # 각 샘플에 대한 메트릭 계산
                    for i in range(batch_size):
                        sess_id = batch_session_ids[i]
                        
                        # 환경 확인
                        if sess_id in self.val_loader.dataset.environment_sessions['same_user_same_space']:
                            env = 'same_user_same_space'
                        elif sess_id in self.val_loader.dataset.environment_sessions['diff_user_same_space']:
                            env = 'diff_user_same_space'
                        else:
                            env = 'diff_user_diff_space'
                        
                        # 거리 오차 계산
                        distance_error = torch.norm(pos_pred[i] - pos_target[i]).item()
                        
                        # 각도 오차 계산
                        rot_pred_i = F.normalize(rot_pred[i:i+1], p=2, dim=1)
                        rot_target_i = F.normalize(rot_target[i:i+1], p=2, dim=1)
                        angle_error = self.quaternion_angle_difference(rot_pred_i, rot_target_i).item()
                        
                        # 환경별 메트릭 저장
                        env_metrics[env]['distance_errors'].append(distance_error)
                        env_metrics[env]['angle_errors'].append(angle_error)
                        env_metrics[env]['loss'] += loss.item()
                        env_metrics[env]['samples'] += 1
                        env_metrics[env]['latencies'].append(batch_latencies[i])  # 개별 샘플의 latency 저장
                    
                    # 진행 바 업데이트
                    val_loader_tqdm.set_postfix(
                        loss=running_loss/total_samples,
                        pos_loss=pos_loss.item(),
                        rot_loss=rot_loss.item(),
                        latency=f"{np.mean(batch_latencies):.1f}ms"
                    )

                # 환경별 결과 출력
                print("\n=== 환경별 검증 결과 ===")
                for env, metrics in env_metrics.items():
                    if metrics['samples'] == 0:
                        continue
                            
                    avg_distance = np.mean(metrics['distance_errors'])
                    avg_angle = np.mean(metrics['angle_errors'])
                    avg_loss = metrics['loss'] / metrics['samples']
                    avg_latency = np.mean(metrics['latencies'])
                    std_latency = np.std(metrics['latencies'])
                        
                    within_dist = sum(d <= self.target_thresholds['distance'] 
                                    for d in metrics['distance_errors'])
                    within_angle = sum(a <= self.target_thresholds['angle'] 
                                    for a in metrics['angle_errors'])
                        
                    print(f"\n{env}:")
                    print(f"├─ 샘플 수: {metrics['samples']}")
                    print(f"├─ 평균 손실: {avg_loss:.4f}")
                    print(f"├─ 거리 오차: {avg_distance:.3f}m (논문: {self.paper_metrics[env]['distance']}m)")
                    print(f"├─ 각도 오차: {avg_angle:.1f}° (논문: {self.paper_metrics[env]['angle']}°)")
                    print(f"├─ 목표 거리 달성률: {within_dist/metrics['samples']:.2%}")
                    print(f"├─ 목표 각도 달성률: {within_angle/metrics['samples']:.2%}")
                    print(f"├─ 평균 latency: {avg_latency:.2f}ms")
                    print(f"└─ Latency 표준편차: {std_latency:.2f}ms")

                # 전체 latency 통계 계산
                avg_latency = np.mean(latencies)
                std_latency = np.std(latencies)
                min_latency = np.min(latencies)
                max_latency = np.max(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)

                print("\n=== 전체 Latency 통계 ===")
                print(f"├─ 평균 추론 시간: {avg_latency:.2f}ms")
                print(f"├─ 표준 편차: {std_latency:.2f}ms")
                print(f"├─ 최소 추론 시간: {min_latency:.2f}ms")
                print(f"├─ 최대 추론 시간: {max_latency:.2f}ms")
                print(f"├─ 95퍼센타일: {p95_latency:.2f}ms")
                print(f"└─ 99퍼센타일: {p99_latency:.2f}ms")

                # 전체 메트릭 계산
                all_distance_errors = []
                all_angle_errors = []
                for env_metric in env_metrics.values():
                    all_distance_errors.extend(env_metric['distance_errors'])
                    all_angle_errors.extend(env_metric['angle_errors'])
                    
                # 환경별 평균 계산 후 히스토리에 저장
                # env_metrics 키가 없으면 초기화
                if 'env_metrics' not in self.history:
                    self.history['env_metrics'] = {
                        'same_user_same_space': {'distance_errors': [], 'angle_errors': [], 'latencies': []},
                        'diff_user_same_space': {'distance_errors': [], 'angle_errors': [], 'latencies': []},
                        'diff_user_diff_space': {'distance_errors': [], 'angle_errors': [], 'latencies': []}
                    }
                
                for env in env_metrics:
                    if env_metrics[env]['samples'] > 0:
                        avg_distance = np.mean(env_metrics[env]['distance_errors'])
                        avg_angle = np.mean(env_metrics[env]['angle_errors'])
                        avg_latency = np.mean(env_metrics[env]['latencies'])
                        
                        # 환경별 키가 없으면 초기화
                        if env not in self.history['env_metrics']:
                            self.history['env_metrics'][env] = {'distance_errors': [], 'angle_errors': [], 'latencies': []}
                            
                        self.history['env_metrics'][env]['distance_errors'].append(avg_distance)
                        self.history['env_metrics'][env]['angle_errors'].append(avg_angle)
                        
                        # latency 히스토리 추가
                        if 'latencies' not in self.history['env_metrics'][env]:
                            self.history['env_metrics'][env]['latencies'] = []
                        self.history['env_metrics'][env]['latencies'].append(avg_latency)
                    
                # 전체 메트릭 반환
                return {
                    'val_loss': running_loss / total_samples,
                    'val_mae': np.mean(all_distance_errors),
                    'val_rmse': np.sqrt(np.mean(np.array(all_distance_errors) ** 2)),
                    'val_angle_error': np.mean(all_angle_errors),
                    'val_target_dist_acc': sum(d <= self.target_thresholds['distance'] 
                                            for d in all_distance_errors) / total_samples,
                    'val_target_angle_acc': sum(a <= self.target_thresholds['angle'] 
                                            for a in all_angle_errors) / total_samples,
                    'val_paper_dist_ratio': np.mean(all_distance_errors) / 
                                        self.paper_metrics['same_user_same_space']['distance'],
                    'val_paper_angle_ratio': np.mean(all_angle_errors) / 
                                            self.paper_metrics['same_user_same_space']['angle'],
                    'val_latency': {
                        'mean': avg_latency,
                        'std': std_latency,
                        'min': min_latency,
                        'max': max_latency,
                        'p95': p95_latency,
                        'p99': p99_latency
                    }
                }
            
    def weights_init(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()

    def save_checkpoint(self, epoch, composite_score=None, val_metrics=None, is_best_composite=False, is_best_angle=False):
        """체크포인트 저장 함수"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict(),
                'best_angle_acc': self.best_angle_acc,
                'best_composite_score': self.best_composite_score,
                'history': self.history,
                'n_iter': self.n_iter,
                # 점진적 학습 관련 정보 추가
                'warmup_epochs': self.warmup_epochs,
                'rotation_ramp_epochs': self.rotation_ramp_epochs,
                'current_rotation_weight': self.final_rotation_weight if epoch >= self.warmup_epochs else 0.0
            }
                
            if composite_score is not None:
                checkpoint['composite_score'] = composite_score
                
            if val_metrics is not None:
                checkpoint['angle_accuracy'] = val_metrics['val_target_angle_acc']
                checkpoint['distance_accuracy'] = val_metrics['val_target_dist_acc']
                
            # 기본 체크포인트 저장
            base_filename = f'model_epoch_{epoch}'
                
            # 최고 복합 점수인 경우
            if is_best_composite:
                filename = f'{base_filename}_best_composite.pth'
                save_path = os.path.join(self.timestamp_dir, filename)
                torch.save(checkpoint, save_path)
                print(f"\n[Checkpoint] New best composite score: {self.best_composite_score:.2%}")
                print(f"           Angle Acc: {val_metrics['val_target_angle_acc']:.2%}, "
                    f"Distance Acc: {val_metrics['val_target_dist_acc']:.2%}")
                    
            # 최고 각도 정확도인 경우
            if is_best_angle:
                filename = f'{base_filename}_best_angle.pth'
                save_path = os.path.join(self.timestamp_dir, filename)
                torch.save(checkpoint, save_path)
                print(f"\n[Checkpoint] New best angle accuracy: {self.best_angle_acc:.2%}")
            
            # 정기 체크포인트 저장 (10 에폭마다 또는 마지막 에폭)
            if epoch % self.config.save_freq == 0 or epoch == self.max_epochs:
                filename = f'{base_filename}.pth'
                save_path = os.path.join(self.timestamp_dir, filename)
                torch.save(checkpoint, save_path)
                print(f"\n[Checkpoint] Regular checkpoint saved at epoch {epoch}")
                
                # 오래된 체크포인트 삭제 (최신 N개만 유지)
                if hasattr(self.config, 'keep_last_n_checkpoints') and self.config.keep_last_n_checkpoints > 0:
                    self._cleanup_old_checkpoints()
            
            return True  # 성공적으로 저장됨을 나타내는 값 반환
            
        except Exception as e:
            print(f"\n[ERROR] 체크포인트 저장 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False  # 저장 실패를 나타내는 값 반환

    def save_metrics(self, epoch, train_metrics, val_metrics):
        """메트릭 저장 함수"""
        # 기존 메트릭 저장
        for key in train_metrics:
            if key in self.history:
                self.history[key].append(train_metrics[key])
        
        for key in val_metrics:
            if key in self.history:
                self.history[key].append(val_metrics[key])
                
        # 점진적 학습 관련 메트릭 추가
        if epoch < self.warmup_epochs:
            current_phase = "warmup"
            rotation_weight = 0.0
        else:
            current_phase = "full"
            progress = min(1.0, (epoch - self.warmup_epochs) / self.rotation_ramp_epochs)
            rotation_weight = self.initial_rotation_weight + \
                            (self.final_rotation_weight - self.initial_rotation_weight) * progress
        
        # Tensorboard에 현재 학습 단계 정보 기록
        self.writer.add_scalar('Training/Phase', 1 if current_phase == "full" else 0, epoch)
        self.writer.add_scalar('Training/RotationWeight', rotation_weight, epoch)

    def close(self):
        self.writer.close()

    def _cleanup_old_checkpoints(self):
        """오래된 정기 체크포인트 삭제 (최고 성능 체크포인트는 유지)"""
        try:
            # 정기 체크포인트 파일 목록 가져오기
            checkpoint_files = []
            for filename in os.listdir(self.timestamp_dir):
                if filename.startswith('model_epoch_') and filename.endswith('.pth'):
                    # 최고 성능 체크포인트는 제외
                    if '_best_' not in filename:
                        checkpoint_files.append(filename)
            
            # 에폭 번호 기준으로 정렬
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
            
            # 오래된 체크포인트 삭제
            if len(checkpoint_files) > self.config.keep_last_n_checkpoints:
                files_to_delete = checkpoint_files[self.config.keep_last_n_checkpoints:]
                
                for filename in files_to_delete:
                    file_path = os.path.join(self.timestamp_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"[Cleanup] Removed old checkpoint: {filename}")
                    except Exception as e:
                        print(f"[Cleanup] Error removing {filename}: {str(e)}")
                
        except Exception as e:
            print(f"\n[ERROR] 체크포인트 정리 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
