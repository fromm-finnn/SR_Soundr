# check_labels.py

import numpy as np

def check_output_labels(output_path, num_samples=10):
    """
    output.npy 파일에 저장된 라벨을 불러와서 
    위치(3차원) + 쿼터니언(4차원)이 기대하는 순서로 되어 있는지,
    실제 쿼터니언 노름이 1에 가까운지 확인.
    """
    outputs = np.load(output_path)  # (total_samples, 8) 형태라고 가정
    
    print(f"Output 라벨 shape: {outputs.shape}")
    
    for i in range(min(num_samples, len(outputs))):
        label = outputs[i]
        # 가정: label = [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, vad_flag] (8차원)
        pos = label[:3]
        quat = label[3:7]
        vad_flag = label[7]  # 가정: 8번째 요소가 VAD 여부
        
        # 쿼터니언 노름
        quat_norm = np.linalg.norm(quat)
        
        print(f"Index {i}: pos={pos}, quat={quat}, quat_norm={quat_norm:.4f}, vad={vad_flag}")

if __name__ == "__main__":
    output_path = "data/output.npy"
    check_output_labels(output_path, num_samples=10)
