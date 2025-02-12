import numpy as np
from tqdm import tqdm
from pathlib import Path

def compute_and_save_session_stats(input_path, starts_path, output_path, selected_channels=[0,4,8,12]):
    print("데이터 로딩 중...")
    inputs = np.load(input_path, mmap_mode='r')
    starts = np.load(starts_path, allow_pickle=True)
    
    print("세션 통계 계산 중...")
    session_stats = {}
    
    # 전체 세션에 대해 계산
    for sess_id, session in tqdm(enumerate(starts), total=len(starts), desc="세션 처리"):
        start_idx = session['start']
        end_idx = session['end']
        
        if end_idx - start_idx == 0:  # 빈 세션 스킵
            continue
            
        # 세션의 처음/중간/끝에서 샘플링
        sample_size = 1000
        indices = []
        
        # 시작 부분
        indices.extend(range(start_idx, min(start_idx + sample_size, end_idx)))
        # 중간 부분
        mid_idx = (start_idx + end_idx) // 2
        indices.extend(range(max(mid_idx - sample_size//2, start_idx), 
                           min(mid_idx + sample_size//2, end_idx)))
        # 끝 부분
        indices.extend(range(max(end_idx - sample_size, start_idx), end_idx))
        
        # 통계 계산
        audio_samples = inputs[indices][:, selected_channels]
        config = session['config']
        session_stats[sess_id] = {
            'mean': float(np.mean(audio_samples)),
            'std': float(np.std(audio_samples)),
            'session_id': sess_id,
            'user_id': config['user_id'],
            'room_id': config['room_id'],
            'n_samples': end_idx - start_idx
        }
    
    print(f"\n총 {len(session_stats)} 개 세션 처리 완료")
    
    # 통계 출력
    print("\n=== 세션 통계 요약 ===")
    print(f"전체 세션 수: {len(session_stats)}")
    
    # 환경별 세션 수
    env_counts = {'same_user_same_space': 0, 'diff_user_same_space': 0, 'diff_user_diff_space': 0}
    ref_user = starts[0]['config']['user_id']
    ref_room = starts[0]['config']['room_id']
    
    for stats in session_stats.values():
        if stats['user_id'] == ref_user and stats['room_id'] == ref_room:
            env_counts['same_user_same_space'] += 1
        elif stats['user_id'] != ref_user and stats['room_id'] == ref_room:
            env_counts['diff_user_same_space'] += 1
        else:
            env_counts['diff_user_diff_space'] += 1
    
    print("\n환경별 세션 수:")
    for env, count in env_counts.items():
        print(f"- {env}: {count}")
    
    # 통계 저장
    stats_dir = Path(output_path).parent
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, session_stats)
    print(f"\n통계 저장 완료: {output_path}")

if __name__ == "__main__":
    input_path = "./data/input.npy"
    starts_path = "./data/starts.npy"
    output_path = "./data/session_stats.npy"
    
    compute_and_save_session_stats(input_path, starts_path, output_path)