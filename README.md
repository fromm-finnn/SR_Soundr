# SoundR-DL: 음향 기반 방향 및 위치 추정 시스템

SoundR-DL은 마이크 배열을 통해 수집된 오디오 데이터를 분석하여 음원의 방향과 위치를 추정하는 딥러닝 기반 시스템입니다. 이 프로젝트는 UMA-16 마이크 배열을 사용하여 3D 공간에서 음원의 위치와 방향을 정확하게 예측합니다.

## 데이터 준비

1. 데이터 다운로드:
```bash
# soundr-dataset 디렉토리로 이동
cd soundr-dataset

# 데이터 다운로드 스크립트 실행
bash download.sh
```

2. 데이터 전처리:
```bash
# 전처리 스크립트 실행
python preprocess.py
```

3. 전처리된 데이터 이동:
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
- **목표 달성률**: 지정된 임계값(거리 0.35m, 각도 25°) 이내의 예측 비율 