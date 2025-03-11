import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class AudioNet(nn.Module):
    def __init__(self, sample_num=2400, microphone_num=4, output_num=7, config=None):
        super(AudioNet, self).__init__()
        print(f"AudioNet 초기화 시작: sample_num={sample_num}, microphone_num={microphone_num}, output_num={output_num}")
        
        # config 통합
        self.config = config
        self.use_amp = config.use_amp if config else True
        
        # 실제 샘플 수 계산 (4배 다운샘플링 적용)
        actual_sample_num = sample_num // 4
        print(f"실제 사용 샘플 수: {actual_sample_num} (4배 다운샘플링 적용)")
        
        # 커널 크기 설정
        self.kernel_sizes = {
            'conv1': 7,
            'conv2': 5,
            'conv3': 3
        }
        
        # CNN 특징 추출기 - 적절한 채널 수 설정
        self.features = nn.Sequential(
            # 첫 번째 블록 - stride 증가
            nn.Conv1d(microphone_num, 96, kernel_size=self.kernel_sizes['conv1'], 
                    padding=self.kernel_sizes['conv1']//2, stride=4, bias=False),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(inplace=True),
            
            # 두 번째 블록 - stride 증가
            nn.Conv1d(96, 192, kernel_size=self.kernel_sizes['conv2'], 
                    padding=self.kernel_sizes['conv2']//2, stride=4, bias=False),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(inplace=True),
            
            # 세 번째 블록 - 마지막 레이어
            nn.Conv1d(192, 704, kernel_size=self.kernel_sizes['conv3'], 
                    padding=self.kernel_sizes['conv3']//2, stride=1, bias=False),
            nn.BatchNorm1d(704),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 적응형 풀링으로 변경 - 항상 1x1 출력
        )
        
        # FC 레이어 - 적절한 크기 설정
        self.fc = nn.Sequential(
            nn.Linear(704, 1408, bias=False),  # 704는 features의 출력 채널 수
            nn.BatchNorm1d(1408),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(config.dropout_rate if config else 0.5)
        )
        
        # LSTM - 적절한 크기 설정
        self.lstm = nn.LSTM(
            input_size=1408,
            hidden_size=704,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate if config else 0.5,
            bidirectional=True
        )
        
        # 출력 레이어 분리 (위치와 회전)
        self.fc_position = nn.Linear(1408, 3)  # 양방향 LSTM이므로 704*2=1408
        self.fc_rotation = nn.Linear(1408, 4)
        
        # 가중치 초기화
        self._initialize_weights()
        print("AudioNet 초기화 완료")
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        sequence_length = x.size(3)  # 데이터 형태가 (batch, channels=4, samples=1200, seq_len=20)
        
        # Mixed Precision 사용
        with torch.amp.autocast('cuda', enabled=self.use_amp): 
            # 데이터 형태 변환 (batch, channels, samples, seq_len) -> (batch*seq_len, channels, samples)
            x = x.permute(0, 3, 1, 2).contiguous()  # (batch, seq_len, channels, samples)
            x = x.view(-1, x.size(2), x.size(3))    # (batch*seq_len, channels, samples)
            
            # 한 번에 모든 프레임 처리
            features = self.features(x)          # (batch*seq_len, 704, 1)
            features = features.squeeze(-1)      # (batch*seq_len, 704)
            features = self.fc(features)         # (batch*seq_len, 1408)
            
            # 원래 배치 및 시퀀스 형태로 복원
            features = features.view(batch_size, sequence_length, -1)  # (batch, seq_len, 1408)
            
            # LSTM 처리
            lstm_out, hidden = self.lstm(features)  # lstm_out: (batch, seq_len, hidden_size*2)
            
            # 마지막 시점의 출력 사용
            x = lstm_out[:, -1]  # (batch, hidden_size*2)
            
            # 위치와 회전 분리 예측
            position = self.fc_position(x)
            rotation = self.fc_rotation(x)
            
            # 쿼터니언 정규화
            rotation = F.normalize(rotation, p=2, dim=1)
        
        # 결과 반환
        return position, rotation
    
    def get_loss(self, pred, target):
        pos_pred, quat_pred = pred
        pos_target, quat_target = target[:, :3], target[:, 3:]
        
        # 위치 손실 (동일)
        pos_loss = F.l1_loss(pos_pred, pos_target)
        
        # 쿼터니온 정규화
        quat_pred = F.normalize(quat_pred, p=2, dim=1)
        quat_target = F.normalize(quat_target, p=2, dim=1)
        
        # 양방향 쿼터니온 손실 계산 (double cover 고려)
        dot_product_pos = torch.sum(quat_pred * quat_target, dim=1)
        dot_product_neg = torch.sum(quat_pred * (-quat_target), dim=1)
        
        # 더 작은 각도를 주는 방향 선택
        dot_product = torch.where(
            torch.abs(dot_product_pos) > torch.abs(dot_product_neg),
            dot_product_pos,
            dot_product_neg
        )
        
        # 안전한 범위로 클램핑
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 각도 기반 손실 계산
        quat_loss = 1 - torch.abs(dot_product).mean()
        
        # 전체 손실 계산
        total_loss = self.config.position_loss_weight * pos_loss + \
                    self.config.rotation_loss_weight * quat_loss
        
        return total_loss, {'pos_loss': pos_loss.item(), 'quat_loss': quat_loss.item()}