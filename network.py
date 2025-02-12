import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import math

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        return out

class AudioNet(nn.Module):
    def __init__(self, sample_num=4800, microphone_num=2, output_num=7, config=None):
        super(AudioNet, self).__init__()
        print(f"AudioNet 초기화 시작: sample_num={sample_num}, microphone_num={microphone_num}, output_num={output_num}")
        
        self.config = config
        self.use_amp = config.use_amp if config else True
        self.current_epoch = 0  # 현재 epoch 추적용
        
        # CNN 특징 추출부 (기존과 동일)
        self.features = nn.Sequential(
            nn.Conv2d(microphone_num, 64, kernel_size=(7,3), stride=(2,1), padding=(3,1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,1), stride=(4,1)),
            
            nn.Conv2d(64, 128, kernel_size=(7,3), stride=(2,1), padding=(3,1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,1), stride=(4,1)),
            
            nn.Conv2d(128, 256, kernel_size=(7,3), stride=(1,1), padding=(3,1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4,1), stride=(4,1)),
            
            nn.Conv2d(256, 512, kernel_size=(7,3), stride=(1,1), padding=(3,1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )
        
        # 특성 크기 계산 (기존과 동일)
        with torch.no_grad():
            dummy_input = torch.zeros(1, microphone_num, sample_num, config.sequence_length)
            x = self.features(dummy_input)
            self.feature_size = x.size(1) * x.size(2)
            print(f"계산된 feature size: {self.feature_size}")
        
        # LSTM 및 Attention (기존과 동일)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(config.dropout_rate if config else 0.5)
        )
        
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate if config else 0.5,
            bidirectional=True
        )
        
        self.temporal_attention = SelfAttention(1024)
        
        # 위치 예측 헤드 (기존과 동일)
        self.fc_position = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 3)
        )
        
        # 방향 예측 헤드 (위치 정보 활용을 위해 수정)
        self.fc_rotation = nn.Sequential(
            nn.Linear(2048 + 3, 1024),  # 2048(특징) + 3(위치)
            nn.LayerNorm(1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 4)
        )
        
        self._initialize_weights()
        print("AudioNet 초기화 완료")
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN 특징 추출
        features = self.features(x)
        features = features.permute(0, 3, 1, 2)
        features = features.reshape(batch_size, -1, self.feature_size)
        features = self.fc(features)
            
        # LSTM 및 Attention (기존과 동일)
        lstm_out, _ = self.lstm(features)
        attended = self.temporal_attention(lstm_out)
        
        global_context = attended.mean(dim=1)
        local_context = attended[:, -1]
        combined_context = torch.cat([global_context, local_context], dim=1)
        
        # 위치 먼저 예측
        position = self.fc_position(combined_context)
        
        # 위치 정보를 방향 예측에 활용
        rotation_features = torch.cat([combined_context, position], dim=1)
        rotation = F.normalize(self.fc_rotation(rotation_features), p=2, dim=1)
            
        return position, rotation  # 튜플로 반환

    def get_loss(self, pred, target):
        pos_pred, quat_pred = pred  # 튜플로 받음
        pos_target, quat_target = target[:, :3], target[:, 3:]
        
        # 위치 손실
        pos_loss = F.l1_loss(pos_pred, pos_target)
        
        # 쿼터니온 손실 (기존과 동일)
        quat_pred = F.normalize(quat_pred, p=2, dim=1)
        quat_target = F.normalize(quat_target, p=2, dim=1)
        
        dot_product_pos = torch.sum(quat_pred * quat_target, dim=1)
        dot_product_neg = torch.sum(quat_pred * (-quat_target), dim=1)
        
        dot_product = torch.where(
            torch.abs(dot_product_pos) > torch.abs(dot_product_neg),
            dot_product_pos,
            dot_product_neg
        )
        
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        quat_loss = 1 - torch.abs(dot_product).mean()
            
        # 점진적 학습을 위한 손실 가중치 조정
        if self.current_epoch < self.config.warmup_epochs:
            # warmup 기간에는 위치 학습에만 집중
            total_loss = self.config.position_loss_weight * pos_loss
        else:
            # warmup 이후 회전 손실 점진적 도입
            progress = min(1.0, (self.current_epoch - self.config.warmup_epochs) / 
                        self.config.rotation_ramp_epochs)
            current_rotation_weight = (self.config.final_rotation_weight - 
                                    self.config.initial_rotation_weight) * progress + \
                                    self.config.initial_rotation_weight
            
            total_loss = self.config.position_loss_weight * pos_loss + \
                        current_rotation_weight * quat_loss
        
        return total_loss, {'pos_loss': pos_loss.item(), 
                        'quat_loss': quat_loss.item(),
                        'rotation_weight': current_rotation_weight if 'current_rotation_weight' in locals() 
                                            else 0.0}

    def set_epoch(self, epoch):
        """현재 epoch 설정"""
        self.current_epoch = epoch

    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)