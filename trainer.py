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

eps = 1e-6

class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.max_epochs = config.num_epochs

        # Mixed Precision 설정
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
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
            
            # 목표 달성률 관련 메트릭
            'train_target_dist_acc': [],  # 목표 거리(0.35m) 달성률
            'train_target_angle_acc': [], # 목표 각도(25°) 달성률
            'val_target_dist_acc': [],
            'val_target_angle_acc': [],
            
            # 논문 비교용 메트릭
            'train_paper_dist_ratio': [],  # 논문 거리 오차(0.31m) 대비 비율
            'train_paper_angle_ratio': [], # 논문 각도 오차(34.3°) 대비 비율
            'val_paper_dist_ratio': [],
            'val_paper_angle_ratio': [],  # 여기 콤마 추가

            # 환경별 메트릭
            'env_metrics': {
                'same_user_same_space': {
                    'distance_errors': [],
                    'angle_errors': []
                },
                'diff_user_same_space': {
                    'distance_errors': [],
                    'angle_errors': []
                },
                'diff_user_diff_space': {
                    'distance_errors': [],
                    'angle_errors': []
                }
            }
        }

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
        if hasattr(self.model, 'current_epoch'):
            if self.model.current_epoch < self.warmup_epochs:
                # warmup 기간에는 위치 학습에만 집중
                loss = self.position_loss_weight * pos_loss
                current_rotation_weight = 0.0
            else:
                # warmup 이후 회전 손실 점진적 도입
                progress = min(1.0, (self.model.current_epoch - self.warmup_epochs) / 
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
            self.model.set_epoch(epoch)  # 현재 epoch 정보 모델에 전달

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

                # 위치 오차 계산
                with torch.no_grad():
                    distance_errors = torch.sqrt(torch.sum((pos_pred - pos_target) ** 2, dim=1))
                    mean_distance_error = distance_errors.mean().item()
                    total_distance_error += distance_errors.sum().item()
                    total_distance_error_sq += torch.sum(distance_errors ** 2).item()

                    # 방향 오차 계산
                    rot_pred = F.normalize(rot_pred, p=2, dim=1)
                    rot_target = F.normalize(rot_target, p=2, dim=1)
                    angle_diff = self.quaternion_angle_difference(rot_pred, rot_target)
                    mean_angle_error = angle_diff.mean().item()
                    total_angle_error += angle_diff.sum().item()

                    # 임계값 내 샘플 수 계산
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

            # 매 10 에폭마다 또는 첫 에폭에서 그래프 저장
            if epoch % 10 == 0 or epoch == 1:
                self.plot_metrics(epoch)
                print(f"\n[Visualization] Metrics plot saved at epoch {epoch}")

                # 학습률 스케줄러 업데이트
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()

                # 복합 점수 계산 및 체크포인트 저장
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

                # 체크포인트 저장
                self.save_checkpoint(
                    epoch=epoch,
                    composite_score=composite_score,
                    val_metrics=val_metrics,
                    is_best_composite=is_best_composite,
                    is_best_angle=is_best_angle
                )

        self.plot_metrics()  # 최종 그래프 저장
        self.writer.close()
        print("Training completed.")

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
            
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

        val_loader_tqdm = tqdm(self.val_loader, desc='Validation', unit='batch')
            
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                for batch_idx, (inputs, (pos_target, rot_target)) in enumerate(val_loader_tqdm):
                    # 데이터를 디바이스로 이동
                    inputs = inputs.to(self.device)
                    pos_target = pos_target.to(self.device)
                    rot_target = rot_target.to(self.device)
                    
                    # 모델 추론
                    pos_pred, rot_pred = self.model(inputs)
                    
                    # 손실 계산
                    loss, pos_loss, rot_loss = self.criterion(
                        (pos_pred, rot_pred),
                        (pos_target, rot_target)
                    )
                        
                    # 배치의 세션 ID 가져오기
                    batch_session_ids = self.val_loader.dataset.get_session_ids(batch_idx)
                        
                    # 각 샘플별로 환경 분류하여 메트릭 계산
                    for i, sess_id in enumerate(batch_session_ids):
                        # 환경 확인
                        if sess_id in self.val_loader.dataset.environment_sessions['same_user_same_space']:
                            env = 'same_user_same_space'
                        elif sess_id in self.val_loader.dataset.environment_sessions['diff_user_same_space']:
                            env = 'diff_user_same_space'
                        else:
                            env = 'diff_user_diff_space'
                            
                        # 개별 샘플의 오차 계산
                        # 위치 오차
                        distance_error = torch.norm(pos_pred[i] - pos_target[i]).item()
                            
                        # 각도 오차
                        rot_pred_i = F.normalize(rot_pred[i:i+1], p=2, dim=1)
                        rot_target_i = F.normalize(rot_target[i:i+1], p=2, dim=1)
                        angle_error = self.quaternion_angle_difference(rot_pred_i, rot_target_i).item()
                            
                        # 환경별 메트릭 저장
                        env_metrics[env]['distance_errors'].append(distance_error)
                        env_metrics[env]['angle_errors'].append(angle_error)
                        env_metrics[env]['loss'] += loss.item() / inputs.size(0)  # 배치 크기로 나누기
                        env_metrics[env]['samples'] += 1
                            
                        running_loss += loss.item() / inputs.size(0)
                        total_samples += 1

                    # 진행 바 업데이트
                    val_loader_tqdm.set_postfix(
                        loss=running_loss/total_samples,
                        pos_loss=pos_loss.item(),
                        rot_loss=rot_loss.item()
                    )

                # 환경별 결과 출력
                print("\n=== 환경별 검증 결과 ===")
                for env, metrics in env_metrics.items():
                    if metrics['samples'] == 0:
                        continue
                            
                    avg_distance = np.mean(metrics['distance_errors'])
                    avg_angle = np.mean(metrics['angle_errors'])
                    avg_loss = metrics['loss'] / metrics['samples']
                        
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
                    print(f"└─ 목표 각도 달성률: {within_angle/metrics['samples']:.2%}")

                # 전체 메트릭 계산
                all_distance_errors = []
                all_angle_errors = []
                for env_metric in env_metrics.values():
                    all_distance_errors.extend(env_metric['distance_errors'])
                    all_angle_errors.extend(env_metric['angle_errors'])
                    
                # 환경별 평균 계산 후 히스토리에 저장
                for env in env_metrics:
                    if env_metrics[env]['samples'] > 0:
                        avg_distance = np.mean(env_metrics[env]['distance_errors'])
                        avg_angle = np.mean(env_metrics[env]['angle_errors'])
                            
                        self.history['env_metrics'][env]['distance_errors'].append(avg_distance)
                        self.history['env_metrics'][env]['angle_errors'].append(avg_angle)
                    
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
                                            self.paper_metrics['same_user_same_space']['angle']
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

    def plot_metrics(self, current_epoch=None):
        # 데이터가 없으면 그래프를 그리지 않음
        if not self.history['train_loss']:
            print("No metrics data to plot yet.")
            return
                
        # epochs 범위 설정
        epochs = range(1, len(self.history['train_loss']) + 1)
            
        sns.set_style("whitegrid")
        
        # 메인 메트릭 그래프 (2x3 구조로 변경)
        plt.figure(figsize=(20, 12))

        # 폰트 크기 조정 (발표용)
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.7,
            'grid.linestyle': '--'
        })
                
        # 색상 팔레트 재정의
        colors = {
            'train_primary': '#2E86C1',    # 진한 파랑
            'train_secondary': '#5DADE2',  # 연한 파랑
            'val_primary': '#E74C3C',      # 진한 빨강
            'val_secondary': '#F1948A',    # 연한 빨강
            'phase_line': '#27AE60',       # 초록색 (학습 단계 구분선)
            'weight_line': '#8E44AD'       # 보라색 (가중치 변화)
        }
            
        # 1. Loss 그래프
        plt.subplot(2, 3, 1)
        plt.plot(epochs, self.history['train_loss'], color=colors['train_primary'], 
                label='Train Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], color=colors['val_primary'], 
                label='Val Loss', linewidth=2)
        
        # Warmup 및 Ramp 구간 표시
        if self.warmup_epochs > 0:
            plt.axvline(x=self.warmup_epochs, color=colors['phase_line'], 
                    linestyle='--', alpha=0.5, label='Warmup End')
        
        plt.title('Loss over epochs', fontsize=12, pad=10)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0)
        plt.legend(fontsize=10)
            
        # 2. Distance Error 그래프
        plt.subplot(2, 3, 2)
        plt.plot(epochs, self.history['train_mae'], color=colors['train_primary'], 
                label='Train MAE', linewidth=2)
        plt.plot(epochs, self.history['train_rmse'], color=colors['train_secondary'], 
                label='Train RMSE', linewidth=2)
        plt.plot(epochs, self.history['val_mae'], color=colors['val_primary'], 
                label='Val MAE', linewidth=2)
        plt.plot(epochs, self.history['val_rmse'], color=colors['val_secondary'], 
                label='Val RMSE', linewidth=2)
        
        if self.warmup_epochs > 0:
            plt.axvline(x=self.warmup_epochs, color=colors['phase_line'], 
                    linestyle='--', alpha=0.5, label='Warmup End')
        
        plt.title('Distance Errors', fontsize=12, pad=10)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Error (meters)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(bottom=0)
        plt.legend(fontsize=10)

        # 3. Rotation Weight 그래프 (새로 추가)
        plt.subplot(2, 3, 3)
        rotation_weights = []
        for epoch in epochs:
            if epoch < self.warmup_epochs:
                rotation_weights.append(0.0)
            else:
                progress = min(1.0, (epoch - self.warmup_epochs) / self.rotation_ramp_epochs)
                weight = self.initial_rotation_weight + \
                        (self.final_rotation_weight - self.initial_rotation_weight) * progress
                rotation_weights.append(weight)
        
        plt.plot(epochs, rotation_weights, color=colors['weight_line'], 
                label='Rotation Weight', linewidth=2)
        plt.title('Rotation Weight Progression', fontsize=12, pad=10)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Weight', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 4. Distance Accuracy 그래프
        plt.subplot(2, 3, 4)
        plt.plot(epochs, self.history['train_target_dist_acc'], color=colors['train_primary'], 
                label=f'Train (<{self.target_thresholds["distance"]}m)', linewidth=2)
        plt.plot(epochs, self.history['val_target_dist_acc'], color=colors['val_primary'], 
                label=f'Val (<{self.target_thresholds["distance"]}m)', linewidth=2)
        
        if self.warmup_epochs > 0:
            plt.axvline(x=self.warmup_epochs, color=colors['phase_line'], 
                    linestyle='--', alpha=0.5, label='Warmup End')
        
        plt.title('Distance Accuracy', fontsize=12, pad=10)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy (%)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 5. Angle Accuracy 그래프
        plt.subplot(2, 3, 5)
        plt.plot(epochs, self.history['train_target_angle_acc'], color=colors['train_primary'], 
                label=f'Train (<{self.target_thresholds["angle"]}°)', linewidth=2)
        plt.plot(epochs, self.history['val_target_angle_acc'], color=colors['val_primary'], 
                label=f'Val (<{self.target_thresholds["angle"]}°)', linewidth=2)
        
        if self.warmup_epochs > 0:
            plt.axvline(x=self.warmup_epochs, color=colors['phase_line'], 
                    linestyle='--', alpha=0.5, label='Warmup End')
        
        plt.title('Angle Accuracy', fontsize=12, pad=10)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Accuracy (%)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 6. Paper Comparison 그래프
        plt.subplot(2, 3, 6)
        plt.plot(epochs, self.history['train_paper_dist_ratio'], color=colors['train_primary'], 
                label='Distance Ratio', linewidth=2)
        plt.plot(epochs, self.history['train_paper_angle_ratio'], color=colors['train_secondary'], 
                label='Angle Ratio', linewidth=2)
        plt.axhline(y=1.0, color='red', linestyle='--', label='Paper Baseline')
        
        if self.warmup_epochs > 0:
            plt.axvline(x=self.warmup_epochs, color=colors['phase_line'], 
                    linestyle='--', alpha=0.5, label='Warmup End')
        
        plt.title('Performance vs Paper', fontsize=12, pad=10)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Ratio to Paper Results', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        # 그리드 스타일 수정
        for ax in plt.gcf().get_axes():
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
        plt.tight_layout(pad=3.0)
            
        # 저장
        if current_epoch:
            save_path = os.path.join(self.timestamp_dir, 
                                f'training_metrics_epoch_{current_epoch}.png')
        else:
            save_path = os.path.join(self.timestamp_dir, 
                                'training_metrics_final.png')
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 환경별 성능 변화 그래프
        self.plot_environment_metrics(current_epoch)

    def plot_environment_metrics(self, current_epoch=None):
        """환경별 성능 변화를 보여주는 별도의 그래프"""
        plt.figure(figsize=(15, 6))
        
        # 데이터가 없으면 그래프를 그리지 않음
        if not self.history['env_metrics']['same_user_same_space']['distance_errors']:
            print("No environment metrics data to plot yet.")
            return
            
        # epochs 범위 재계산
        epochs = range(1, len(self.history['env_metrics']['same_user_same_space']['distance_errors']) + 1)
            
        # 1. 위치 오차 변화 추이
        plt.subplot(1, 2, 1)
        for env in self.paper_metrics.keys():
            if (env in self.history['env_metrics'] and 
                len(self.history['env_metrics'][env]['distance_errors']) > 0):
                
                plt.plot(epochs, self.history['env_metrics'][env]['distance_errors'], 
                        linewidth=2, marker='o', label=env)
                plt.axhline(y=self.paper_metrics[env]['distance'], 
                        linestyle='--', alpha=0.5, label=f'{env} (Paper)')
        
        if self.warmup_epochs > 0:
            plt.axvline(x=self.warmup_epochs, color='g', 
                    linestyle='--', alpha=0.5, label='Warmup End')
            
        plt.title('Position Error by Environment')
        plt.xlabel('Epochs')
        plt.ylabel('Position Error (meters)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 2. 각도 오차 변화 추이
        plt.subplot(1, 2, 2)
        for env in self.paper_metrics.keys():
            if (env in self.history['env_metrics'] and 
                len(self.history['env_metrics'][env]['angle_errors']) > 0):
                
                plt.plot(epochs, self.history['env_metrics'][env]['angle_errors'], 
                        linewidth=2, marker='o', label=env)
                plt.axhline(y=self.paper_metrics[env]['angle'], 
                        linestyle='--', alpha=0.5, label=f'{env} (Paper)')
        
        if self.warmup_epochs > 0:
            plt.axvline(x=self.warmup_epochs, color='g', 
                    linestyle='--', alpha=0.5, label='Warmup End')
            
        plt.title('Orientation Error by Environment')
        plt.xlabel('Epochs')
        plt.ylabel('Orientation Error (degrees)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout(pad=3.0)
            
        # 저장
        if current_epoch:
            save_path = os.path.join(self.timestamp_dir, 
                                f'environment_metrics_epoch_{current_epoch}.png')
        else:
            save_path = os.path.join(self.timestamp_dir, 
                                'environment_metrics_final.png')
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def close(self):
        self.writer.close()
