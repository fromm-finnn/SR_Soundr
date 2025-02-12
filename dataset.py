import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.spatial.transform import Rotation as R 


class AudioDataset(Dataset):
    def __init__(self, input_path, output_path, starts_path, config=None, mode='train', transform=None, fraction=1.0, env_type='same_user_same_space'):
        print(f"데이터셋 초기화 시작... (mode: {mode})")
        
        # 기본 속성 설정
        self.config = config
        self.mode = mode
        self.transform = transform

        # 채널 선택 추가
        self.selected_channels = [0,8]  # [0,4,8,12]
        print(f"선택된 마이크 채널: {self.selected_channels}")
            
        # 1. config 설정 (별도 메서드로 분리)
        self._setup_config(config)
        
        # 2. 데이터 분할 비율 검증
        self.validate_split_ratios()
        
        # 3. 데이터 로드
        self._load_data(input_path, output_path, starts_path)
        
        # 4. 환경별 세션 분류 및 분할
        self._classify_and_split_sessions(env_type, mode)
        
        # 5. 데이터셋 통계 출력
        self.print_dataset_statistics()

    def _setup_config(self, config):
        """config 설정"""
        if config:
            # params.py의 설정값 사용
            self.train_ratio = getattr(config, 'train_ratio', 0.8)
            self.val_ratio = getattr(config, 'val_ratio', 0.2)
            self.test_ratio = getattr(config, 'test_ratio', 0.1)
            self.random_seed = getattr(config, 'random_seed', 42)
            self.environment_type = getattr(config, 'environment_type', 'same_user_same_space')
            
            # 오디오 관련 설정
            self.audio_mean = getattr(config, 'audio_mean', 0.0)
            self.audio_std = getattr(config, 'audio_std', 0.0008)
            self.pos_min = np.array(getattr(config, 'pos_min', [-3.70873404, 0.81547654, -13.88833714]))
            self.pos_max = np.array(getattr(config, 'pos_max', [1.81399226, 2.48111653, 4.08869743]))
            self.use_augmentation = getattr(config, 'use_augmentation', False)
            

            # 점진적 학습 관련 설정 추가
            self.warmup_epochs = getattr(config, 'warmup_epochs', 20)
            self.rotation_ramp_epochs = getattr(config, 'rotation_ramp_epochs', 10)
            self.initial_rotation_weight = getattr(config, 'initial_rotation_weight', 0.0)
            self.final_rotation_weight = getattr(config, 'final_rotation_weight', 3.0)
            
            # 증강 관련 설정 추가
            self.aug_params = config.get_augmentation_params()
            self.phase_shift_range = getattr(config, 'phase_shift_range', (-np.pi/8, np.pi/8))
            self.attenuation_range = getattr(config, 'attenuation_range', (0.7, 1.0))
            self.channel_variance = getattr(config, 'channel_variance', 0.1)
            self.noise_smoothing_kernel = getattr(config, 'noise_smoothing_kernel', 5)
            self.impulse_response_length = getattr(config, 'impulse_response_length', 32)
        else:
            # 기본값 설정
            self.train_ratio = 0.7
            self.val_ratio = 0.15
            self.test_ratio = 0.15
            self.random_seed = 42
            self.environment_type = 'same_user_same_space'
            self.audio_mean = 0.0
            self.audio_std = 0.0008
            self.pos_min = np.array([-3.70873404, 0.81547654, -13.88833714])
            self.pos_max = np.array([1.81399226, 2.48111653, 4.08869743])
            self.use_augmentation = False

        self.sequence_length = getattr(config, 'sequence_length', 30)  # 논문에서 제시한 시퀀스 길이


    def get_sequence_angles(self, session_data, seq_len):
        """시퀀스 단위로 각도 통계 계산"""
        quaternions = session_data[:, 3:7]
        rot = R.from_quat(quaternions)
        euler_angles = rot.as_euler('xyz', degrees=True)
        
        angles = []
        for i in range(0, len(euler_angles) - seq_len + 1):
            seq = euler_angles[i:i + seq_len]
            # 시퀀스의 대표 각도 (중간값 사용)
            yaw = np.median(seq[:, 2])
            pitch = np.median(seq[:, 1])
            angles.append([yaw, pitch])
        return np.array(angles)

    def _load_data(self, input_path, output_path, starts_path):
        """데이터 로드"""
        print("입력 데이터 매핑 중...")
        self.inputs = np.load(input_path, mmap_mode='r')
        print("출력 데이터 로딩 중...")
        self.outputs = np.load(output_path, mmap_mode='r')
        print("세션 정보 로딩 중...")
        self.starts = np.load(starts_path, allow_pickle=True)

    def _classify_and_split_sessions(self, env_type, mode):
        """환경별 세션 분류 및 분할"""
        print("\n=== 환경별 세션 분류 ===")
        self.environment_sessions = {
            'same_user_same_space': [],
            'diff_user_same_space': [],
            'diff_user_diff_space': []
        }

        # 기준 사용자와 공간 설정
        reference_session = self.starts[0]['config']
        ref_user = reference_session['user_id']
        ref_room = reference_session['room_id']
        
        print(f"기준 사용자 ID: {ref_user}")
        print(f"기준 공간 ID: {ref_room}")
        
        # 세션 분류
        for sess_id, session in enumerate(self.starts):
            config = session['config']
            curr_user = config['user_id']
            curr_room = config['room_id']
            
            if curr_user == ref_user and curr_room == ref_room:
                self.environment_sessions['same_user_same_space'].append(sess_id)
            elif curr_user != ref_user and curr_room == ref_room:
                self.environment_sessions['diff_user_same_space'].append(sess_id)
            elif curr_room != ref_room:
                self.environment_sessions['diff_user_diff_space'].append(sess_id)

        # 선택된 환경의 세션만 사용
        selected_env_sessions = self.environment_sessions[env_type]
        print(f"\n선택된 환경: {env_type}")
        print(f"전체 세션 수: {len(selected_env_sessions)}")
        
        # 모든 시퀀스를 하나로 모으기
        all_sequences = []
        session_map = []  # 각 시퀀스가 어느 세션에서 왔는지 추적
        
        for sess_id in selected_env_sessions:
            session = self.starts[sess_id]
            start_idx = session['start']
            end_idx = session['end']
            
            # 시퀀스 길이를 고려하여 유효한 시작점들 추출
            valid_starts = list(range(start_idx, end_idx - self.sequence_length + 1))
            all_sequences.extend(valid_starts)
            session_map.extend([sess_id] * len(valid_starts))
        
        # 전체 시퀀스를 섞기
        np.random.seed(self.random_seed)  # 재현성을 위한 시드 설정
        all_sequences = np.array(all_sequences)
        session_map = np.array(session_map)
        shuffle_idx = np.random.permutation(len(all_sequences))
        all_sequences = all_sequences[shuffle_idx]
        session_map = session_map[shuffle_idx]
        
        # train/val/test 분할
        n_total = len(all_sequences)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # 모드에 따라 인덱스 할당
        if mode == 'train':
            self.indices = all_sequences[:n_train]
            self.session_map = session_map[:n_train]
        elif mode == 'val':
            self.indices = all_sequences[n_train:n_train + n_val]
            self.session_map = session_map[n_train:n_train + n_val]
        else:  # test
            self.indices = all_sequences[n_train + n_val:]
            self.session_map = session_map[n_train + n_val:]
        
        # 각도 분포 분석 및 출력
        print("\n=== 데이터 분할 요약 ===")
        print(f"전체 시퀀스 수: {n_total}")
        print(f"학습 시퀀스 수: {n_train}")
        print(f"검증 시퀀스 수: {n_val}")
        print(f"테스트 시퀀스 수: {len(all_sequences) - n_train - n_val}")
        print(f"현재 모드({mode}) 시퀀스 수: {len(self.indices)}")
        
        # 각도 분포 분석
        if mode == 'train':
            train_angles = self.get_sequence_angles(self.outputs[self.indices], self.sequence_length)
            print("\n=== 학습 데이터 각도 분포 ===")
            print(f"Yaw 범위: {np.min(train_angles[:,0]):.1f}° ~ {np.max(train_angles[:,0]):.1f}°")
            print(f"Pitch 범위: {np.min(train_angles[:,1]):.1f}° ~ {np.max(train_angles[:,1]):.1f}°")
        
        # 세션별 데이터 포인트 수 출력
        unique_sessions = np.unique(self.session_map)
        print(f"\n현재 모드의 세션별 데이터 포인트 수:")
        for sess_id in unique_sessions:
            count = np.sum(self.session_map == sess_id)
            print(f"세션 {sess_id}: {count} 포인트")

    def validate_split_ratios(self):
        """데이터 분할 비율 검증"""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"분할 비율의 합이 1이 되어야 합니다. (현재: {total_ratio})")

    def print_dataset_statistics(self):
        """데이터셋 통계 출력"""
        print("\n=== 데이터셋 통계 ===")
        print(f"모드: {self.mode}")
        
        # session_map에서 유니크한 세션 수를 계산
        unique_sessions = np.unique(self.session_map)
        print(f"세션 수: {len(unique_sessions)}")
        print(f"데이터 포인트 수: {len(self.indices)}")
        
        # 현재 모드의 환경별 통계만 출력
        if hasattr(self, 'environment_type'):
            print(f"\n현재 환경 ({self.environment_type}) 통계:")
            total_datapoints = len(self.indices)
            print(f"  - 세션 수: {len(unique_sessions)}")
            print(f"  - 데이터 포인트 수: {total_datapoints}")

    def augment_audio(self, audio, position, quaternion):
        """
        화자의 발화 각도를 증강
        audio: [n_channels, sequence_length, samples]
        position: [3] - (x, y, z)
        quaternion: [4] - (w, x, y, z)
        """
        if not self.use_augmentation or self.mode != 'train':
            return audio, position, quaternion
        
        if np.random.random() < self.aug_params['angle_aug_prob']:
            # 현재 각도 계산
            rot = R.from_quat(quaternion)
            current_euler = rot.as_euler('xyz', degrees=True)
            
            # yaw 각도에 랜덤 오프셋 추가 (-30도 ~ +30도)
            yaw_offset = np.random.uniform(-30, 30)
            new_euler = current_euler.copy()
            new_euler[2] += yaw_offset  # z축 회전 (yaw)
            
            # 새로운 쿼터니온 생성
            new_rot = R.from_euler('xyz', new_euler, degrees=True)
            new_quaternion = new_rot.as_quat()
            
            # 위치 기반으로 오디오 시간 지연 계산 및 적용
            delays = self._calculate_delays(position, new_quaternion)
            audio = self._apply_delays(audio, delays)
            
            return audio, position, new_quaternion
        
        return audio, position, quaternion

    def _calculate_delays(self, position, quaternion):
        """마이크별 시간 지연 계산"""
        # 마이크 배열의 상대 위치 (예시 값)
        mic_positions = np.array([
            [0, 0, 0],  # 첫 번째 마이크를 기준점으로
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0.1, 0.1, 0]
        ])
        
        # 음속 (m/s)
        speed_of_sound = 343.0
        
        # 화자 방향 벡터 계산
        rot = R.from_quat(quaternion)
        direction = rot.apply([1, 0, 0])  # 정면 방향
        
        # 각 마이크까지의 지연 시간 계산
        delays = []
        for mic_pos in mic_positions:
            # 화자로부터 마이크까지의 상대 벡터
            relative_pos = mic_pos - position
            # 방향에 따른 지연 계산
            delay = np.dot(relative_pos, direction) / speed_of_sound
            delays.append(delay)
        
        return np.array(delays)

    def _apply_delays(self, audio, delays):
        """계산된 지연을 오디오에 적용"""
        sample_rate = 48000  # 샘플링 레이트
        delayed_audio = torch.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            delay_samples = int(delays[ch] * sample_rate)
            if delay_samples > 0:
                delayed_audio[ch] = torch.roll(audio[ch], shifts=delay_samples, dims=1)
            else:
                delayed_audio[ch] = audio[ch]
        
        return delayed_audio

    @torch.no_grad()
    def _generate_temporal_noise(self, n_channels, sequence_length, samples, noise_range, correlation=0.7):
        """
        시간적/공간적 일관성이 있는 노이즈 생성
        Args:
            n_channels: 마이크 채널 수
            sequence_length: 시퀀스 길이
            samples: 샘플 수
            noise_range: (min_noise, max_noise) 노이즈 레벨 범위
            correlation: 채널 간 노이즈 상관관계 (0: 완전 독립, 1: 완전 동일)
        """
        noise_level = np.random.uniform(*noise_range)
        
        # 기본 노이즈 생성 (채널 공유)
        base_noise = torch.randn(1, sequence_length, samples) * noise_level
        
        # 채널별 독립 노이즈
        channel_noise = torch.randn(n_channels, sequence_length, samples) * noise_level
        
        # 상관관계를 고려한 노이즈 조합
        noise = correlation * base_noise + (1 - correlation) * channel_noise
        
        # 시간적 스무딩
        kernel_size = self.config.noise_smoothing_kernel
        padding = (kernel_size - 1) // 2
        
        smoothed_noise = torch.zeros_like(noise)
        for ch in range(n_channels):
            smoothed_noise[ch] = torch.nn.functional.avg_pool1d(
                noise[ch].unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            ).squeeze(0)
        
        return smoothed_noise

        """
        audio: shape [n_channels, sequence_length, samples]
        """
        # 홀수 길이의 임펄스 응답 생성
        ir_length = self.config.impulse_response_length
        if ir_length % 2 == 0:  # 짝수면 홀수로 만들기
            ir_length += 1
            
        impulse_response = torch.zeros(ir_length)
        impulse_response[0] = 1.0
        mid_point = ir_length // 3
        impulse_response[mid_point] = self.config.early_reflection_factor
        impulse_response[-1] = self.config.late_reflection_factor
        
        output = torch.zeros_like(audio)
        for ch in range(audio.shape[0]):
            for seq in range(audio.shape[1]):
                # [1, 1, samples]로 reshape
                audio_frame = audio[ch, seq].unsqueeze(0).unsqueeze(0)
                # [1, 1, ir_length]로 reshape
                ir = impulse_response.unsqueeze(0).unsqueeze(0)
                
                # 컨볼루션 적용 (explicit padding)
                pad_size = ir_length // 2
                padded_audio = torch.nn.functional.pad(audio_frame, (pad_size, pad_size), mode='reflect')
                output[ch, seq] = torch.nn.functional.conv1d(
                    padded_audio,
                    ir,
                    padding=0  # explicit padding을 사용했으므로 0
                ).squeeze()
        
        return output

    def normalize_audio(self, audio):
        """오디오 데이터 정규화"""
        # 전체 스케일만 조정 (너무 큰 값이나 작은 값 방지)
        audio = (audio - self.audio_mean) / self.audio_std
        return audio
    
    def normalize_position(self, position):
        """위치 데이터 정규화 (-1 ~ 1 범위로)"""
        return 2.0 * (position - self.pos_min) / (self.pos_max - self.pos_min) - 1.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sequence_length = self.sequence_length
        
        # 시퀀스 시작점
        start_idx = self.indices[idx]
        
        # 전체 시퀀스 한 번에 로드
        sequence_indices = range(start_idx, start_idx + sequence_length)
        x = self.inputs[sequence_indices]  # [sequence_length, all_channels, samples]
        
        # 채널 선택 및 다운샘플링
        x = x[:, self.selected_channels]   # [sequence_length, n_selected_channels, samples]
        x = x[:, :, ::4]                   # 4배 다운샘플링
        
        # 정규화
        x = self.normalize_audio(x)
        x = torch.from_numpy(x).float()    # [sequence_length, n_channels, samples]
        
        # 증강을 위한 shape 변경
        x = x.permute(1, 0, 2)            # [n_channels, sequence_length, samples]
        
        # 타겟 데이터 처리
        y = self.outputs[start_idx]
        position = y[:3]
        quaternion = y[3:7]
        
        # 각도 증강 적용
        if self.mode == 'train' and self.use_augmentation:
            x, position, quaternion = self.augment_audio(x, position, quaternion)
        
        x = x.permute(0, 2, 1)  # 채널은 그대로, sequence와 samples를 교환
        
        # 위치 정규화
        position_normalized = self.normalize_position(position)
        
        # 쿼터니온 정규화 (단위 벡터로)
        quaternion = quaternion / (np.linalg.norm(quaternion) + 1e-7)
        
        # 위치와 회전을 별도로 반환
        position_tensor = torch.from_numpy(position_normalized).float()
        quaternion_tensor = torch.from_numpy(quaternion).float()
        
        return x, (position_tensor, quaternion_tensor)
        
    def get_session_ids(self, batch_idx, real_batch_size):
        """실제 배치 크기를 고려하여 세션 ID 반환"""
        start_idx = batch_idx * self.config.batch_size
        end_idx = start_idx + real_batch_size
        
        # 디버깅 출력 추가
        print(f"Debug: start_idx={start_idx}, end_idx={end_idx}, len(self)={len(self)}")
        
        if start_idx >= len(self):  # 시작 인덱스가 데이터셋 크기를 넘어가는지 체크
            print(f"Warning: start_idx {start_idx} exceeds dataset size {len(self)}")
            return []
            
        if end_idx > len(self):
            end_idx = len(self)
        
        session_ids = []
        for idx in range(start_idx, end_idx):
            real_idx = self.indices[idx]
            
            # 세션 찾기
            for sess_id, session in enumerate(self.starts):
                if session['start'] <= real_idx < session['end']:
                    session_ids.append(sess_id)
                    break
        
        # 디버깅 출력 추가
        print(f"Debug: Found {len(session_ids)} session ids for batch {batch_idx}")
        
        return session_ids
        
    @staticmethod
    def collate_fn(batch):
        """배치 데이터 처리"""
        # 입력 데이터
        inputs = torch.stack([item[0] for item in batch])
        
        # 타겟 데이터 (위치, 회전 분리)
        positions = torch.stack([item[1][0] for item in batch])
        rotations = torch.stack([item[1][1] for item in batch])
        
        return inputs, (positions, rotations)