# AudioNet : 발화자 방향 및 위치 추정 시스템

## 데이터 준비

### 1. Stanford Soundr 데이터셋 다운로드

먼저 Stanford Soundr 데이터셋 저장소를 클론합니다:
```bash
# Stanford Soundr 데이터셋 저장소 클론
git clone https://github.com/stanford-soundr/soundr-dataset.git
cd soundr-dataset
```

데이터 다운로드 스크립트를 실행합니다:
```bash
# 데이터 다운로드 스크립트 실행 (약 125GB 필요)
bash download.sh
```

> **참고**: 원본 데이터셋은 Stanford Library에서 호스팅되며, 약 125GB의 저장 공간이 필요합니다.

### 2. 데이터 전처리

필요한 패키지를 설치하고 전처리 스크립트를 실행합니다:
```bash
# 필요한 패키지 설치
python3 -m pip install -r requirements.txt

# 데이터 전처리 실행
python3 ./preprocess.py {다운로드된 데이터 폴더} --output {출력 폴더}
```

전처리 과정을 통해 다음 파일들이 생성됩니다:
- `input.npy`: 다중 채널 오디오 데이터 (세그먼트 수, 오디오 채널(16), 세그먼트 길이)
- `output.npy`: 헤드셋 트래킹 데이터 (세그먼트 수, 트래킹 데이터(x, y, z, q_x, q_y, q_z, q_w, vad))
- `starts.npy`: 데이터 세션 구성 정보

### 3. 전처리된 데이터 이동

전처리된 데이터 파일을 soundr-dl의 data 디렉토리로 이동합니다:
```bash
# soundr-dl의 data 디렉토리 생성
mkdir -p ../soundr-dl/data

# 전처리된 데이터 파일 이동
cp input.npy output.npy starts.npy ../soundr-dl/data/
```

## 사용 방법

### 학습

기본 4채널 모델 학습:
```bash
python main.py train --mic_num 4 --mic_array "0,4,8,12" --sequence_length 20 --gpu 0
```

2채널 최적화 모드 학습:
```bash
python main.py train --mic_num 2 --mic_array "0,8" --sequence_length 20 --gpu 0 --optimize_two_channel
```

학습 속도 향상을 위한 latency 측정 비활성화:
```bash
# 기본적으로 latency 측정은 비활성화되어 있습니다
# 필요한 경우 --measure_latency 옵션을 추가하여 활성화할 수 있습니다
python main.py train --mic_num 4 --mic_array "0,4,8,12" --sequence_length 20 --gpu 0 --measure_latency
```

### 평가

기본 모델 평가:
```bash
python main.py test --mic_num 4 --mic_array "0,4,8,12" --checkpoint "checkpoints/dov_soundr_4ch(0,4,8,12)_audionet_v3/model_epoch_100_best_composite.pth" --gpu 0
```

노이즈 강건성 테스트:
```bash
python main.py test --mic_num 4 --mic_array "0,4,8,12" --checkpoint "checkpoints/dov_soundr_4ch(0,4,8,12)_audionet_v3/model_epoch_100_best_composite.pth" --gpu 0 --run_snr_test
```

## 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--mic_num` | 사용할 마이크 채널 수 | 4 |
| `--mic_array` | 사용할 마이크 인덱스 (쉼표로 구분) | "0,4,8,12" |
| `--sequence_length` | 시퀀스 길이 | 20 |
| `--optimize_two_channel` | 2채널 최적화 활성화 | False |
| `--batch_size` | 배치 크기 | params.py에서 설정 |
| `--learning_rate` | 학습률 | params.py에서 설정 |
| `--experiment_name` | 실험 이름 (체크포인트 저장 폴더명) | 자동 생성 |
| `--measure_latency` | 검증 시 latency 측정 여부 | False |

## 마이크 배열 구성

SoundR-DL은 UMA-16 마이크 배열 레이아웃을 기반으로 합니다. 기본 구성은 다음과 같습니다:

```
┌───┬───┬───┬───┐
│ 7 │ 6 │ 9 │ 8 │
├───┼───┼───┼───┤
│ 5 │ 4 │ 11│ 10│
├───┼───┼───┼───┤
│ 3 │ 2 │ 13│ 12│
├───┼───┼───┼───┤
│ 1 │ 0 │ 15│ 14│
└───┴───┴───┴───┘
```

기본 4채널 구성은 채널 0, 4, 8, 12를 사용합니다.

## 성능 지표

모델 성능은 다음 지표를 통해 평가됩니다:

- **거리 오차 (MAE)**: 예측된 위치와 실제 위치 간의 평균 절대 오차 (미터)
- **각도 오차**: 예측된 방향과 실제 방향 간의 평균 각도 오차 (도)
- **목표 달성률**: 지정된 임계값 이내의 예측 비율 
