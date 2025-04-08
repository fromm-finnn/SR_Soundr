import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.spatial.transform import Rotation as R 


class AudioDataset(Dataset):
    # 클래스 변수로 초기화 상태 추적
    _initialized = False
    _env_sessions_initialized = False
    _data_loaded = False
    _env_sessions = None
    _all_sequences = None
    _session_map_full = None
    _total_sequences = 0
    _train_count = 0
    _val_count = 0
    _test_count = 0
    
    def __init__(self, input_path, output_path, starts_path, config=None, mode='train', transform=None, fraction=1.0, env_type='same_user_same_space'):
        print(f"데이터셋 초기화 시작... (mode: {mode})")
        
        # 기본 속성 설정
        self.config = config
        self.mode = mode
        self.transform = transform
        self.environment_type = env_type

        # 채널 선택 설정
        # 명령줄에서 설정된 채널이 있으면 우선 사용
        if config and hasattr(config, 'selected_channels') and config.selected_channels is not None:
            self.selected_channels = config.selected_channels
            if not AudioDataset._initialized:
                print(f"명령줄 지정 채널 사용: {self.selected_channels}")
        # 2채널 최적화가 활성화된 경우 two_channel_indices 사용
        elif config and hasattr(config, 'optimize_for_two_channel') and config.optimize_for_two_channel:
            self.selected_channels = list(config.two_channel_indices) if hasattr(config, 'two_channel_indices') else [0, 8]
            if not AudioDataset._initialized:
                print(f"2채널 최적화 모드 활성화: {self.selected_channels}")
        else:
            self.selected_channels = [0, 4, 8, 12]  # 기본 4채널 설정
            if not AudioDataset._initialized:
                print(f"4채널 모드 사용: {self.selected_channels}")
            
        # 1. config 설정 (별도 메서드로 분리)
        self._setup_config(config)
        
        # 2. 데이터 분할 비율 검증
        self.validate_split_ratios()
        
        # 3. 데이터 로드 
        if not AudioDataset._data_loaded:
            if not AudioDataset._initialized:
                print("데이터 로드 중...")
            self._load_data(input_path, output_path, starts_path)
            AudioDataset._data_loaded = True
            AudioDataset.inputs = self.inputs
            AudioDataset.outputs = self.outputs
            AudioDataset.starts = self.starts
        else:
            self.inputs = AudioDataset.inputs
            self.outputs = AudioDataset.outputs
            self.starts = AudioDataset.starts
        
        # 4. 환경별 세션 분류 및 분할 
        if not AudioDataset._env_sessions_initialized:
            self._classify_and_split_sessions(env_type)
            AudioDataset._env_sessions_initialized = True
            AudioDataset._env_sessions = self.environment_sessions
            AudioDataset._all_sequences = self.all_sequences
            AudioDataset._session_map_full = self.session_map_full
            AudioDataset._total_sequences = len(self.all_sequences)
        else:
            self.environment_sessions = AudioDataset._env_sessions
            self.all_sequences = AudioDataset._all_sequences
            self.session_map_full = AudioDataset._session_map_full
        
        # 모드에 따라 인덱스 할당
        self._assign_indices_by_mode(mode)
        
        # 5. 데이터셋 통계 출력 
        if not AudioDataset._initialized and mode == 'train':
            self.print_dataset_summary()
        
        # 현재 모드의 데이터셋 통계 출력
        self.print_dataset_statistics()
        
        # 초기화 완료 표시
        AudioDataset._initialized = True

    def _setup_config(self, config):
        """config 설정"""
        if config:
            # params.py의 설정값 사용
            self.train_ratio = getattr(config, 'train_ratio', 0.8)
            self.val_ratio = getattr(config, 'val_ratio', 0.2)
            self.test_ratio = getattr(config, 'test_ratio', 0.1)
            self.random_seed = getattr(config, 'random_seed', 42)
            self.sequence_length = getattr(config, 'sequence_length', 20)
            
            # 오디오 관련 설정
            self.audio_mean = getattr(config, 'audio_mean', 0.0)
            self.audio_std = getattr(config, 'audio_std', 0.0008)
            self.pos_min = np.array(getattr(config, 'pos_min', [-3.70873404, 0.81547654, -13.88833714]))
            self.pos_max = np.array(getattr(config, 'pos_max', [1.81399226, 2.48111653, 4.08869743]))
            
            # 데이터 증강 설정
            self.use_augmentation = getattr(config, 'use_augmentation', False)
            self.aug_params = {
                'angle_aug_prob': getattr(config, 'angle_aug_prob', 0.5),
                'noise_aug_prob': getattr(config, 'noise_aug_prob', 0.3),
                'noise_level': getattr(config, 'noise_level', 0.01)
            }
            
            # 마이크 위치 및 음속 설정
            # config에서 all_mic_positions가 있으면 선택된 채널에 맞게 마이크 위치 설정
            if hasattr(config, 'all_mic_positions') and len(config.all_mic_positions) >= 16:
                self.mic_positions = [config.all_mic_positions[ch] for ch in self.selected_channels]
            else:
                # 기존 방식 유지 (config.mic_positions 사용)
                self.mic_positions = getattr(config, 'mic_positions', [
                    [0, -0.126, 0],    # 첫 번째 마이크 - 0번 채널 (M1)
                    [0, 0.042, 0],     # 두 번째 마이크 - 4번 채널 (M5)
                    [0.126, 0.126, 0],  # 세 번째 마이크 - 8번 채널 (M9)
                    [0.126, -0.042, 0]  # 네 번째 마이크 - 12번 채널 (M13)
                ])
            
            self.speed_of_sound = getattr(config, 'speed_of_sound', 343.0)
            
            # 마이크 위치 설정 정보 출력 (한 번만)
            if not AudioDataset._initialized:
                print(f"마이크 위치 설정 완료: {len(self.mic_positions)}개 채널")
                for i, pos in enumerate(self.mic_positions):
                    print(f"  채널 {i}: 위치 {pos}")
                
                # UMA-16 레이아웃 격자 시각화
                if hasattr(config, '_visualize_mic_grid'):
                    config._visualize_mic_grid()
        else:
            # 기본값 설정
            self.train_ratio = 0.8
            self.val_ratio = 0.1
            self.test_ratio = 0.1
            self.random_seed = 42
            self.sequence_length = 20
            self.use_augmentation = False
            self.audio_mean = 0.0
            self.audio_std = 0.0008
            self.pos_min = np.array([-3.70873404, 0.81547654, -13.88833714])
            self.pos_max = np.array([1.81399226, 2.48111653, 4.08869743])
            self.aug_params = {
                'angle_aug_prob': 0.5,
                'noise_aug_prob': 0.3,
                'noise_level': 0.01
            }
            # 기본 마이크 위치 설정 (UMA-16 레이아웃의 채널 0, 4, 8, 12)
            self.mic_positions = [
                [0, -0.126, 0],    # 첫 번째 마이크 - 0번 채널 (M1)
                [0, 0.042, 0],     # 두 번째 마이크 - 4번 채널 (M5)
                [0.126, 0.126, 0],  # 세 번째 마이크 - 8번 채널 (M9)
                [0.126, -0.042, 0]  # 네 번째 마이크 - 12번 채널 (M13)
            ]
            self.speed_of_sound = 343.0

    def _load_data(self, input_path, output_path, starts_path):
        """데이터 로드"""
        print("입력 데이터 매핑 중...")
        self.inputs = np.load(input_path, mmap_mode='r')
        print("출력 데이터 로딩 중...")
        self.outputs = np.load(output_path, mmap_mode='r')
        print("세션 정보 로딩 중...")
        self.starts = np.load(starts_path, allow_pickle=True)

    def _classify_and_split_sessions(self, env_type):
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

        # 모든 세션 사용 (환경 타입에 관계없이)
        print("\n모든 세션을 사용합니다!")
        all_sessions = list(range(len(self.starts)))
        print(f"환경 타입 '{env_type}'에서 모든 세션으로 변경")
        print(f"전체 세션 수: {len(all_sessions)}")
        
        # 모든 시퀀스를 하나로 모으기
        self.all_sequences = []
        self.session_map_full = []  # 각 시퀀스가 어느 세션에서 왔는지 추적
        
        for sess_id in all_sessions:
            session = self.starts[sess_id]
            start_idx = session['start']
            end_idx = session['end']
            
            # 시퀀스 길이를 고려하여 유효한 시작점들 추출
            valid_starts = list(range(start_idx, end_idx - self.sequence_length + 1))
            self.all_sequences.extend(valid_starts)
            self.session_map_full.extend([sess_id] * len(valid_starts))
        
        # 전체 시퀀스를 섞기
        np.random.seed(self.random_seed)  # 재현성을 위한 시드 설정
        self.all_sequences = np.array(self.all_sequences)
        self.session_map_full = np.array(self.session_map_full)
        shuffle_idx = np.random.permutation(len(self.all_sequences))
        self.all_sequences = self.all_sequences[shuffle_idx]
        self.session_map_full = self.session_map_full[shuffle_idx]
        
        # 데이터 분할 계산
        n_total = len(self.all_sequences)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # 클래스 변수에 저장
        AudioDataset._train_count = n_train
        AudioDataset._val_count = n_val
        AudioDataset._test_count = n_total - n_train - n_val

    def _assign_indices_by_mode(self, mode):
        """모드에 따라 인덱스 할당"""
        n_total = len(self.all_sequences)
        n_train = AudioDataset._train_count
        n_val = AudioDataset._val_count
        
        if mode == 'train':
            self.indices = self.all_sequences[:n_train]
            self.session_map = self.session_map_full[:n_train]
        elif mode == 'val':
            self.indices = self.all_sequences[n_train:n_train + n_val]
            self.session_map = self.session_map_full[n_train:n_train + n_val]
        else:  # test
            self.indices = self.all_sequences[n_train + n_val:]
            self.session_map = self.session_map_full[n_train + n_val:]

    def validate_split_ratios(self):
        """데이터 분할 비율 검증"""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"분할 비율의 합이 1이 되어야 합니다. (현재: {total_ratio})")

    def print_dataset_summary(self):
        """데이터셋 요약 정보 출력"""
        print("\n=== 데이터셋 요약 ===")
        
        # 기본 정보
        total_samples = len(self.inputs)
        print(f"총 샘플 수: {total_samples:,}")
        print(f"입력 형태: {self.inputs.shape}")
        print(f"출력 형태: {self.outputs.shape}")
        print(f"세션 수: {len(self.starts)}")
        
        # 위치 및 쿼터니언 정보
        positions = self.outputs[:, :3]  # 위치 (x, y, z)
        quaternions = self.outputs[:, 3:7]  # 쿼터니언 (w, x, y, z)
        
        # 노름이 0인 쿼터니언 통계
        quat_norms = np.linalg.norm(quaternions, axis=1)
        zero_norm_count = np.sum(quat_norms < 1e-5)
        print(f"노름이 0인 쿼터니언 비율: {zero_norm_count / len(quaternions) * 100:.2f}%")
        
        # 위치 통계
        x_min, y_min, z_min = np.min(positions, axis=0)
        x_max, y_max, z_max = np.max(positions, axis=0)
        print(f"위치 범위:")
        print(f"  X: {x_min:.2f} ~ {x_max:.2f}")
        print(f"  Y: {y_min:.2f} ~ {y_max:.2f}")
        print(f"  Z: {z_min:.2f} ~ {z_max:.2f}")
        
        # 각도 정보 (노름이 0이 아닌 샘플만 사용)
        valid_quaternions = quaternions[quat_norms >= 1e-5]
        if len(valid_quaternions) > 0:
            # 단위 벡터로 정규화
            valid_quaternions = valid_quaternions / np.linalg.norm(valid_quaternions, axis=1, keepdims=True)
            # 처음 100개 샘플의 각도 계산
            sample_indices = np.arange(0, min(len(valid_quaternions), 100))
            sample_quats = valid_quaternions[sample_indices]
            
            rots = R.from_quat(sample_quats)
            eulers = rots.as_euler('zyx', degrees=True)
            
            yaw_angles = eulers[:, 0]  # z축 각도 (요)
            pitch_angles = eulers[:, 1]  # y축 각도 (피치)
            
            print(f"Yaw 범위 (샘플): {np.min(yaw_angles):.1f}° ~ {np.max(yaw_angles):.1f}°")
            print(f"Pitch 범위 (샘플): {np.min(pitch_angles):.1f}° ~ {np.max(pitch_angles):.1f}°")
        else:
            print("경고: 유효한 쿼터니언이 없습니다")
        
        # 세션 통계
        session_lengths = [session['end'] - session['start'] for session in self.starts]
        min_length = min(session_lengths)
        max_length = max(session_lengths)
        avg_length = sum(session_lengths) / len(session_lengths)
        print(f"세션 길이: 최소 {min_length}, 최대 {max_length}, 평균 {avg_length:.1f}")
        
        # 환경 통계
        print("\n환경별 세션 수:")
        for env, sessions in self.environment_sessions.items():
            print(f"  {env}: {len(sessions):,} 세션")
            
        print("===============")

    def print_dataset_statistics(self):
        """현재 모드의 데이터셋 통계 출력"""
        print(f"\n=== 데이터셋 통계 ({self.mode}) ===")
        unique_sessions = np.unique(self.session_map)
        print(f"세션 수: {len(unique_sessions)}")
        print(f"데이터 포인트 수: {len(self.indices):,}")

    def get_sequence_angles(self, outputs, sequence_length):
        """시퀀스의 각도 정보 계산"""
        angles = []
        for i in range(0, min(len(outputs), 100), sequence_length):
            # 쿼터니언 추출
            quat = outputs[i, 3:7]
            
            # 노름이 0인 쿼터니언 확인 및 처리
            norm = np.linalg.norm(quat)
            if norm < 1e-5:
                # 기본 단위 쿼터니언 [1, 0, 0, 0] 사용 (정면 방향)
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                # 단위 벡터로 정규화
                quat = quat / norm
                
            # 쿼터니언을 회전 객체로 변환
            rot = R.from_quat(quat)
            
            # 오일러 각(각도)으로 변환 - ZYX 회전 순서 사용 (z는 요, y는 피치, x는 롤)
            euler = rot.as_euler('zyx', degrees=True)
            
            angles.append(euler[0])  # z축 각도 (요/방위각) 사용
        
        return angles

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
        # 마이크 배열의 상대 위치 (config에서 가져옴)
        mic_positions = np.array(self.mic_positions)
        
        # 음속 (m/s)
        speed_of_sound = self.speed_of_sound
        
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
        
        # 채널 수와 지연 배열 길이 확인
        n_channels = min(audio.shape[0], len(delays))
        
        for ch in range(n_channels):
            delay_samples = int(delays[ch] * sample_rate)
            if delay_samples > 0:
                delayed_audio[ch] = torch.roll(audio[ch], shifts=delay_samples, dims=1)
            else:
                delayed_audio[ch] = audio[ch]
        
        # 나머지 채널은 원본 그대로 유지
        if audio.shape[0] > n_channels:
            delayed_audio[n_channels:] = audio[n_channels:]
        
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
        
        try:
            # 채널 선택 - 데이터에 맞게 조정
            all_channels = x.shape[1]
            if all_channels == 4:
                # 실제 데이터에는 4개 채널만 있음: 4개 모두 사용
                selected_channels = list(range(all_channels))
                if idx == 0:  # 첫 번째 호출에서만 메시지 출력
                    print(f"데이터에 맞게 채널 조정됨: {selected_channels} (원래 선택: {self.selected_channels})")
            else:
                # 실제 채널이 더 많은 경우 원래 선택된 채널 사용
                selected_channels = self.selected_channels
                
            x = x[:, selected_channels]   # [sequence_length, n_selected_channels, samples]
            x = x[:, :, ::4]              # 4배 다운샘플링
        except Exception as e:
            print(f"\n=== 오류 발생: __getitem__ 채널 선택 중 ===")
            print(f"오류 메시지: {str(e)}")
            print(f"입력 데이터 형태: {x.shape}")
            print(f"선택할 채널: {self.selected_channels}")
            print(f"실제 채널 수: {all_channels}")
            raise e
        
        # 정규화
        x = self.normalize_audio(x)
        x = torch.from_numpy(x).float()    # [sequence_length, n_channels, samples]
        
        # 증강을 위한 shape 변경
        x = x.permute(1, 0, 2)            # [n_channels, sequence_length, samples]
        
        # 타겟 데이터 처리
        y = self.outputs[start_idx]
        position = y[:3]
        quaternion = y[3:7]
        
        # 쿼터니언 유효성 검사 및 처리
        quat_norm = np.linalg.norm(quaternion)
        if quat_norm < 1e-5:
            # 노름이 0인 쿼터니언은 기본 단위 쿼터니언 사용 (정면 방향)
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # 이미 유효한 쿼터니언이면 정규화
            quaternion = quaternion / quat_norm
        
        # 각도 증강 적용
        if self.mode == 'train' and self.use_augmentation:
            x, position, quaternion = self.augment_audio(x, position, quaternion)
        
        x = x.permute(0, 2, 1)  # 채널은 그대로, sequence와 samples를 교환
        
        # 위치 정규화
        position_normalized = self.normalize_position(position)
        
        # 쿼터니언 정규화 (단위 벡터로)
        quaternion = quaternion / (np.linalg.norm(quaternion) + 1e-7)
        
        # 위치와 회전을 별도로 반환
        position_tensor = torch.from_numpy(position_normalized).float()
        quaternion_tensor = torch.from_numpy(quaternion).float()
        
        return x, (position_tensor, quaternion_tensor)
        
    def get_session_ids(self, batch_idx, real_batch_size):
        """실제 배치 크기를 고려하여 세션 ID 반환"""
        # 데이터셋 크기를 고려하여 안전하게 인덱스 계산
        # 복제 방지를 위한 실제 배치 인덱스 계산
        dataset_size = len(self)
        
        # 새로운 계산 방식: batch_idx는 실제 배치 번호로 간주
        start_idx = batch_idx * real_batch_size
        end_idx = min(start_idx + real_batch_size, dataset_size)
        
        if start_idx >= dataset_size:
            print(f"Warning: start_idx {start_idx} exceeds dataset size {dataset_size}")
            return []
        
        if end_idx > dataset_size:
            print(f"Debug: Adjusted end_idx from {start_idx + real_batch_size} to {end_idx}")
            
        session_ids = []
        for idx in range(start_idx, end_idx):
            if idx >= dataset_size:
                break
                
            real_idx = self.indices[idx]
            
            # 세션 찾기
            for sess_id, session in enumerate(self.starts):
                if session['start'] <= real_idx < session['end']:
                    session_ids.append(sess_id)
                    break
                    
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