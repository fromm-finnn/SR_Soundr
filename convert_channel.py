import soundfile as sf
import numpy as np

def convert_to_4ch(input_wav, output_wav, selected_channels=[0, 8]):
    """16채널 WAV 파일을 4채널로 변환"""
    print(f"Loading {input_wav}...")
    audio_data, sr = sf.read(input_wav)
    
    print(f"원본 shape: {audio_data.shape}")
    
    # 선택된 4개 채널만 추출
    selected_data = audio_data[:, selected_channels]
    print(f"변환 후 shape: {selected_data.shape}")
    
    # 4채널 WAV로 저장
    print(f"Saving to {output_wav}...")
    sf.write(output_wav, selected_data, sr)
    
    # 저장된 파일 확인
    audio_info = sf.info(output_wav)
    print("\n=== 생성된 WAV 파일 정보 ===")
    print(f"채널 수: {audio_info.channels}")
    print(f"샘플링 레이트: {audio_info.samplerate}Hz")
    print(f"길이: {audio_info.duration:.2f}초")

# 변환 실행
convert_to_4ch('data/test_sample.wav', 'data/test_sample_2ch.wav')