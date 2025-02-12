import numpy as np
import matplotlib.pyplot as plt

def analyze_sessions(input_path, output_path, starts_path):
    # 데이터 로드
    inputs = np.load(input_path, mmap_mode='r')
    outputs = np.load(output_path, mmap_mode='r')
    starts = np.load(starts_path, allow_pickle=True)
    
    # 선택된 마이크 채널
    selected_channels = [0,4,8,12]
    
    # same_user_same_space 세션만 선택
    same_user_same_space_sessions = []
    reference_session = starts[0]['config']
    ref_user = reference_session['user_id']
    ref_room = reference_session['room_id']
    
    for sess_id, session in enumerate(starts):
        config = session['config']
        curr_user = config['user_id']
        curr_room = config['room_id']
        if curr_user == ref_user and curr_room == ref_room:
            same_user_same_space_sessions.append(sess_id)
    
    print(f"=== same_user_same_space 세션 통계 (총 {len(same_user_same_space_sessions)}개 세션) ===")
    print(f"기준 사용자 ID: {ref_user}")
    print(f"기준 공간 ID: {ref_room}\n")
    
    for sess_id in same_user_same_space_sessions:
        session = starts[sess_id]
        start_idx = session['start']
        end_idx = session['end']
        session_length = end_idx - start_idx
        
        print(f"\n세션 {sess_id}:")
        print(f"전체 데이터 수: {session_length}")
        
        # 각 부분에서 100개씩 샘플링
        sample_size = 100
        indices = []
        
        # 시작 부분
        indices.extend(range(start_idx, min(start_idx + sample_size, end_idx)))
        # 중간 부분
        mid_idx = (start_idx + end_idx) // 2
        mid_start = max(mid_idx - sample_size//2, start_idx)
        mid_end = min(mid_idx + sample_size//2, end_idx)
        indices.extend(range(mid_start, mid_end))
        # 끝 부분
        indices.extend(range(max(end_idx - sample_size, start_idx), end_idx))
        
        # 데이터 샘플링
        audio = inputs[indices][:, selected_channels]
        positions = outputs[indices, :3]
        
        print(f"샘플링 데이터 수: {len(indices)}")
        print(f"오디오 평균: {np.mean(audio):.6f}")
        print(f"오디오 표준편차: {np.std(audio):.6f}")
        print(f"위치 범위: {np.min(positions, axis=0)} ~ {np.max(positions, axis=0)}")

if __name__ == "__main__":
    input_path = "./data/input.npy"
    output_path = "./data/output.npy"
    starts_path = "./data/starts.npy"
    
    analyze_sessions(input_path, output_path, starts_path)