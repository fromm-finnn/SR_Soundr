import torch
import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation as R
import librosa
from network import AudioNet
import quaternion
import time

class Config:
    """AudioNet 설정을 위한 임시 클래스"""
    def __init__(self):
        self.use_amp = True
        self.sequence_length = 20  # inference에서는 1
        self.dropout_rate = 0.5
        self.warmup_epochs = 5
        self.rotation_ramp_epochs = 10
        self.position_loss_weight = 1.0
        self.initial_rotation_weight = 0.0
        self.final_rotation_weight = 1.0

class SoundRInferencer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # 오디오 설정
        self.target_sr = 48000
        self.segment_samples = 1200  # 25ms * 48000Hz
        self.selected_channels = [0, 1]  # 오디오 채널 선택
        
        # 오디오 정규화 파라미터
        self.audio_mean = 0.0
        self.audio_std = 0.0008
        
        # 모델 로드
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        """모델 로드"""
        config = Config()
        checkpoint = torch.load(model_path, map_location=self.device)
        model = AudioNet(
            sample_num=self.segment_samples,
            microphone_num=len(self.selected_channels),
            output_num=7,  # position(3) + quaternion(4)
            config=config
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def load_audio(self, audio_path):
        """오디오 파일 로드 및 전처리"""
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) == 1:
            raise ValueError("모노 오디오는 지원되지 않습니다.")
        elif len(audio.shape) == 2:
            if audio.shape[1] <= max(self.selected_channels):
                raise ValueError(f"오디오 채널 수가 부족합니다. 필요: {max(self.selected_channels) + 1}, 현재: {audio.shape[1]}")
        
        # 리샘플링
        if sr != self.target_sr:
            print(f"리샘플링: {sr}Hz → {self.target_sr}Hz")
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.target_sr).T
        
        # 선택된 채널만 추출
        audio = audio[:, self.selected_channels]
        return audio
    
    def preprocess_segment(self, audio_segment):
        """오디오 세그먼트 전처리"""
        audio_norm = (audio_segment - self.audio_mean) / self.audio_std
        
        audio_tensor = torch.from_numpy(audio_norm).float()
        audio_tensor = audio_tensor.permute(1, 0)  # [channels, samples]
        audio_tensor = audio_tensor.unsqueeze(0)   # [1, channels, samples]
        audio_tensor = audio_tensor.unsqueeze(-1)  # [1, channels, samples, sequence_length]
        
        # sequence_length 차원을 모델이 기대하는 크기로 확장
        if audio_tensor.size(-1) < 20:  # Config의 sequence_length 값
            audio_tensor = audio_tensor.repeat(1, 1, 1, 20)
        
        return audio_tensor
    
    def denormalize_position(self, position_tensor):
        """위치 후처리"""
        return position_tensor.cpu().numpy()
    
    def quaternion_to_euler(self, quaternion_tensor):
        """쿼터니온을 오일러 각도로 변환 (degree)"""
        quat = quaternion_tensor.cpu().numpy()
        quat = quat / (np.linalg.norm(quat) + 1e-7)  # 정규화
        
        # 모델(w,x,y,z) → scipy(x,y,z,w) 순서 변환
        quat_scipy = [quat[1], quat[2], quat[3], quat[0]]
        r = R.from_quat(quat_scipy)
        
        euler_deg = r.as_euler('xyz', degrees=True)
        euler_deg[2] = 0.0  # roll 각도는 0으로 고정
        
        return euler_deg
    
    @torch.no_grad()
    def infer_file(self, audio_path):
        """오디오 파일 추론"""
        try:
            audio = self.load_audio(audio_path)
            print(f"Loaded audio shape: {audio.shape}")
            
            total_samples = audio.shape[0]
            predictions = []
            
            # 세그먼트 단위로 iterate
            for start in range(0, total_samples - self.segment_samples + 1, self.segment_samples):
                segment = audio[start:start + self.segment_samples]
                if len(segment) < self.segment_samples:
                    break
                
                # 전처리
                segment_tensor = self.preprocess_segment(segment)
                segment_tensor = segment_tensor.to(self.device)
                
                # 추론
                pos_pred, rot_pred = self.model(segment_tensor)
                
                # 후처리
                position = self.denormalize_position(pos_pred[0])
                rotation = self.quaternion_to_euler(rot_pred[0])
                
                predictions.append({
                    'timestamp': start / self.target_sr,
                    'position': position,
                    'rotation': rotation
                })
            
            if not predictions:
                return None
            
            # 모든 세그먼트 예측의 중간값으로 대표 추정
            all_positions = np.array([p['position'] for p in predictions])
            all_rotations = np.array([p['rotation'] for p in predictions])
            
            median_position = np.median(all_positions, axis=0)
            median_rotation = np.median(all_rotations, axis=0)
            
            return {
                'position': median_position,
                'rotation': median_rotation,
                'detailed_predictions': predictions
            }
            
        except Exception as e:
            print(f"추론 중 오류 발생: {str(e)}")
            return None

    @torch.no_grad()
    def measure_detailed_latency(self, audio_path, num_runs=100):
        """상세 latency 측정 (전처리, 모델 추론, 후처리 구분)"""
        print(f"\n=== 상세 Latency 측정 ({num_runs}회) ===")
        
        # 오디오 로드 (한 번만)
        audio = self.load_audio(audio_path)
        segment = audio[:self.segment_samples]  # 25ms 세그먼트
        
        # 측정할 구성요소별 시간
        preprocessing_times = []
        inference_times = []
        postprocessing_times = []
        total_times = []
        
        # 웜업
        print("웜업 실행 중...")
        segment_tensor = self.preprocess_segment(segment)
        segment_tensor = segment_tensor.to(self.device)
        pos_pred, rot_pred = self.model(segment_tensor)
        _ = self.denormalize_position(pos_pred[0])
        _ = self.quaternion_to_euler(rot_pred[0])
        
        print("측정 중...")
        for i in range(num_runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
            # 전체 시간 시작
            total_start = time.perf_counter()
            
            # 1. 전처리 시간
            preprocess_start = time.perf_counter()
            segment_tensor = self.preprocess_segment(segment)
            segment_tensor = segment_tensor.to(self.device)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            preprocess_end = time.perf_counter()
            
            # 2. 모델 추론 시간
            inference_start = time.perf_counter()
            pos_pred, rot_pred = self.model(segment_tensor)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            inference_end = time.perf_counter()
            
            # 3. 후처리 시간
            postprocess_start = time.perf_counter()
            position = self.denormalize_position(pos_pred[0])
            rotation = self.quaternion_to_euler(rot_pred[0])
            if self.device == 'cuda':
                torch.cuda.synchronize()
            postprocess_end = time.perf_counter()
            
            # 전체 시간 종료
            total_end = time.perf_counter()
            
            # 시간 기록 (ms 단위)
            preprocessing_times.append((preprocess_end - preprocess_start) * 1000)
            inference_times.append((inference_end - inference_start) * 1000)
            postprocessing_times.append((postprocess_end - postprocess_start) * 1000)
            total_times.append((total_end - total_start) * 1000)
            
            if (i + 1) % 10 == 0:
                print(f"진행률: {i+1}/{num_runs}")
        
        # 통계 계산
        def calculate_stats(times):
            return {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'median': np.median(times),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99)
            }
        
        stats = {
            'preprocessing': calculate_stats(preprocessing_times),
            'inference': calculate_stats(inference_times),
            'postprocessing': calculate_stats(postprocessing_times),
            'total': calculate_stats(total_times)
        }
        
        return stats

def main():
    model_path = "./checkpoints/model_epoch_60_best_composite.pth" 
    audio_path = "data/test_sample_2ch.wav"
    
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n=== 하드웨어 정보 ===")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
    else:
        print("GPU 사용 불가")
    
    try:
        inferencer = SoundRInferencer(model_path, device)
        
        # 상세 Latency 측정
        stats = inferencer.measure_detailed_latency(audio_path, num_runs=1000)
        
        print("\n=== 상세 Latency 통계 (단위: ms) ===")
        
        print("\n1. 전처리 시간:")
        print(f"평균: {stats['preprocessing']['mean']:.3f} ± {stats['preprocessing']['std']:.3f}")
        print(f"중간값: {stats['preprocessing']['median']:.3f}")
        print(f"95퍼센타일: {stats['preprocessing']['p95']:.3f}")
        
        print("\n2. 모델 추론 시간:")
        print(f"평균: {stats['inference']['mean']:.3f} ± {stats['inference']['std']:.3f}")
        print(f"중간값: {stats['inference']['median']:.3f}")
        print(f"95퍼센타일: {stats['inference']['p95']:.3f}")
        
        print("\n3. 후처리 시간:")
        print(f"평균: {stats['postprocessing']['mean']:.3f} ± {stats['postprocessing']['std']:.3f}")
        print(f"중간값: {stats['postprocessing']['median']:.3f}")
        print(f"95퍼센타일: {stats['postprocessing']['p95']:.3f}")
        
        print("\n4. 총 처리 시간:")
        print(f"평균: {stats['total']['mean']:.3f} ± {stats['total']['std']:.3f}")
        print(f"중간값: {stats['total']['median']:.3f}")
        print(f"95퍼센타일: {stats['total']['p95']:.3f}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()