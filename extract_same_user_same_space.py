import numpy as np
import os
from tqdm import tqdm

# 데이터 경로 설정
data_dir = "data"
output_dir = "data_suss"  # same_user_same_space
input_path = os.path.join(data_dir, "input.npy")
output_path = os.path.join(data_dir, "output.npy")
starts_path = os.path.join(data_dir, "starts.npy")

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 메모리 매핑으로 데이터 로드
print("데이터 로드 중...")
input_data = np.load(input_path, mmap_mode='r')
output_data = np.load(output_path, mmap_mode='r')
starts_data = np.load(starts_path, allow_pickle=True)

print(f"원본 입력 데이터 형태: {input_data.shape}")
print(f"원본 출력 데이터 형태: {output_data.shape}")
print(f"원본 세션 수: {len(starts_data)}")

# same_user_same_space 세션 정보 및 인덱스 범위
suss_sessions = []
suss_indices = []
suss_session_ranges = [
    (0, 7650),         # 세션 1
    (83270, 95917),    # 세션 2
    (95917, 102996),   # 세션 3
    (110127, 117085),  # 세션 4
    (241023, 254650),  # 세션 5
    (254650, 268381),  # 세션 6
    (268381, 276974),  # 세션 7
    (318543, 330022),  # 세션 8
    (330022, 336780),  # 세션 9
    (344048, 357412),  # 세션 10
    (404556, 417684)   # 세션 11
]

# 세션 정보 추출 및 인덱스 생성
for start_idx, end_idx in suss_session_ranges:
    suss_indices.extend(range(start_idx, end_idx))

# 원본 세션 정보에서 same_user_same_space 세션 찾기
for session in starts_data:
    start_idx = session['start']
    end_idx = session['end']
    
    # 세션 범위가 suss_session_ranges에 있는지 확인
    for suss_start, suss_end in suss_session_ranges:
        if start_idx == suss_start and end_idx == suss_end:
            suss_sessions.append(session)
            break

print(f"추출할 세션 수: {len(suss_sessions)}")
print(f"추출할 샘플 수: {len(suss_indices)}")

# 새로운 starts 데이터 생성 (인덱스 재조정)
new_starts = []
current_idx = 0

for session in suss_sessions:
    old_start = session['start']
    old_end = session['end']
    samples = old_end - old_start
    
    # 새로운 세션 정보 생성
    new_session = {
        'start': current_idx,
        'end': current_idx + samples,
        'config': session['config']
    }
    
    new_starts.append(new_session)
    current_idx += samples

# 메모리 효율적인 방법으로 데이터 추출
# 배치 단위로 처리하여 메모리 사용량 제한
batch_size = 1000
num_batches = (len(suss_indices) + batch_size - 1) // batch_size

# 새 데이터 배열 생성
filtered_input_shape = (len(suss_indices), input_data.shape[1], input_data.shape[2])
filtered_output_shape = (len(suss_indices), output_data.shape[1])

print(f"필터링된 입력 데이터 형태: {filtered_input_shape}")
print(f"필터링된 출력 데이터 형태: {filtered_output_shape}")
print(f"예상 입력 데이터 크기: {np.prod(filtered_input_shape) * 4 / (1024**3):.2f} GB")
print(f"예상 출력 데이터 크기: {np.prod(filtered_output_shape) * 4 / (1024**3):.2f} GB")

# 메모리 매핑 파일 생성
filtered_input = np.lib.format.open_memmap(
    os.path.join(output_dir, "input.npy"),
    dtype=np.float32,
    mode='w+',
    shape=filtered_input_shape
)

filtered_output = np.lib.format.open_memmap(
    os.path.join(output_dir, "output.npy"),
    dtype=np.float32,
    mode='w+',
    shape=filtered_output_shape
)

# 배치 단위로 데이터 복사
print("데이터 추출 중...")
for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(suss_indices))
    batch_indices = suss_indices[start_idx:end_idx]
    
    # 입력 데이터 복사
    for j, idx in enumerate(batch_indices):
        filtered_input[start_idx + j] = input_data[idx]
    
    # 출력 데이터 복사
    filtered_output[start_idx:end_idx] = output_data[batch_indices]
    
    # 메모리 정리
    if i % 10 == 0:
        filtered_input.flush()
        filtered_output.flush()

# 파일 동기화
filtered_input.flush()
filtered_output.flush()

# starts.npy 저장
np.save(os.path.join(output_dir, "starts.npy"), new_starts)

print(f"same_user_same_space 데이터가 '{output_dir}' 디렉토리에 저장되었습니다.")
print(f"입력 데이터: {os.path.getsize(os.path.join(output_dir, 'input.npy')) / (1024**3):.2f} GB")
print(f"출력 데이터: {os.path.getsize(os.path.join(output_dir, 'output.npy')) / (1024**3):.2f} GB")
print(f"세션 정보: {os.path.getsize(os.path.join(output_dir, 'starts.npy')) / (1024):.2f} KB")

# 새로운 세션 정보 출력
print("\n새로운 세션 정보:")
for i, session in enumerate(new_starts):
    config = session['config']
    start_idx = session['start']
    end_idx = session['end']
    samples = end_idx - start_idx
    
    user_id = config.get('user_id', 'unknown')
    room_id = config.get('room_id', 'unknown')
    trial_id = config.get('trial_id', 'unknown')
    
    print(f"세션 {i+1}:")
    print(f"  - 사용자 ID: {user_id}")
    print(f"  - 공간 ID: {room_id}")
    print(f"  - 시험 ID: {trial_id}")
    print(f"  - 샘플 범위: {start_idx} ~ {end_idx} (총 {samples}개)") 