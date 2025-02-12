import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc  # 가비지 컬렉션을 위해 추가

def analyze_dataset_efficiently(data_dir):
    try:
        data_dir = Path(data_dir)
        
        print("=== 기본 데이터 정보 ===")
        # 파일 존재 확인
        if not (data_dir / "input.npy").exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {data_dir}/input.npy")
        
        # 메모리 매핑으로 파일 로드
        input_data = np.load(data_dir / "input.npy", mmap_mode='r')
        output_data = np.load(data_dir / "output.npy", mmap_mode='r')
        starts_data = np.load(data_dir / "starts.npy", allow_pickle=True)
        
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output_data.shape}")
        print(f"Number of sessions: {len(starts_data)}")
        
        # same_user_same_space 세션 필터링
        reference_session = starts_data[0]['config']
        ref_user = reference_session['user_id']
        ref_room = reference_session['room_id']
        
        same_space_same_user_sessions = []
        for sess_id, session in enumerate(starts_data):
            config = session['config']
            if config['user_id'] == ref_user and config['room_id'] == ref_room:
                same_space_same_user_sessions.append(sess_id)
        
        print(f"\n선택된 세션 수: {len(same_space_same_user_sessions)}")
        
        # 통계 변수 초기화
        pos_min = np.array([float('inf')] * 3)
        pos_max = np.array([float('-inf')] * 3)
        audio_min = float('inf')
        audio_max = float('-inf')
        audio_sum = 0.0
        audio_sq_sum = 0.0
        total_elements = 0
        
        # 세션별로 처리
        for sess_id in tqdm(same_space_same_user_sessions, desc="세션 분석 중"):
            start_idx = starts_data[sess_id]['start']
            end_idx = starts_data[sess_id]['end']
            
            # 위치 데이터 분석
            pos_data = output_data[start_idx:end_idx, :3]
            pos_min = np.minimum(pos_min, np.min(pos_data, axis=0))
            pos_max = np.maximum(pos_max, np.max(pos_data, axis=0))
            del pos_data
            gc.collect()
            
            # 오디오 데이터 분석 (작은 청크로 나누어 처리)
            chunk_size = 1000
            for i in range(start_idx, end_idx, chunk_size):
                end = min(i + chunk_size, end_idx)
                chunk = input_data[i:end].copy()  # 명시적으로 복사
                
                audio_min = min(audio_min, np.min(chunk))
                audio_max = max(audio_max, np.max(chunk))
                audio_sum += np.sum(chunk)
                audio_sq_sum += np.sum(chunk ** 2)
                total_elements += chunk.size
                
                del chunk
                gc.collect()
        
        # 최종 통계 계산
        audio_mean = audio_sum / total_elements
        audio_std = np.sqrt((audio_sq_sum / total_elements) - (audio_mean ** 2))
        
        print("\n=== 분석 결과 ===")
        print(f"위치 (XYZ) 범위: {pos_min} ~ {pos_max}")
        print(f"\n오디오 데이터 통계:")
        print(f"최소값: {audio_min}")
        print(f"최대값: {audio_max}")
        print(f"평균값: {audio_mean:.8f}")
        print(f"표준편차: {audio_std:.8f}")
        
        return {
            'audio_mean': float(audio_mean),
            'audio_std': float(audio_std),
            'pos_min': pos_min.tolist(),
            'pos_max': pos_max.tolist()
        }
        
    finally:
        # 명시적으로 메모리 매핑된 파일들을 닫음
        if 'input_data' in locals():
            del input_data
        if 'output_data' in locals():
            del output_data
        if 'starts_data' in locals():
            del starts_data
        gc.collect()

if __name__ == "__main__":
    try:
        data_dir = "data"
        stats = analyze_dataset_efficiently(data_dir)
        print("\n=== 정규화 파라미터 (Python 형식) ===")
        print(f"audio_mean = {stats['audio_mean']}")
        print(f"audio_std = {stats['audio_std']}")
        print(f"pos_min = {stats['pos_min']}")
        print(f"pos_max = {stats['pos_max']}")
    except KeyboardInterrupt:
        print("\n프로그램 강제 종료됨")
        # 메모리 정리
        gc.collect()