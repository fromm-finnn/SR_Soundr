import numpy as np
from pathlib import Path

def check_starts_structure():
    # starts.npy 파일 로드
    starts_path = Path("data/starts.npy")
    starts = np.load(starts_path, allow_pickle=True)
    
    print(f"=== starts.npy 파일 구조 분석 ===")
    print(f"전체 세션 수: {len(starts)}")
    
    # 첫 번째 세션 상세 정보
    print("\n첫 번째 세션 데이터:")
    first_session = starts[0]
    print(f"타입: {type(first_session)}")
    
    if isinstance(first_session, dict):
        print("\n포함된 키:")
        for key in first_session.keys():
            print(f"- {key}: {type(first_session[key])}")
            
        # 값 예시 출력
        print("\n첫 번째 세션 값 예시:")
        for key, value in first_session.items():
            print(f"{key}: {value}")
    
    # 몇 개 더 샘플링해서 구조 확인
    print("\n다른 세션들 샘플 확인:")
    sample_indices = [len(starts)//4, len(starts)//2, len(starts)-1]  # 25%, 50%, 100% 지점
    for idx in sample_indices:
        print(f"\n세션 {idx}:")
        session = starts[idx]
        if isinstance(session, dict):
            for key, value in session.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    check_starts_structure()