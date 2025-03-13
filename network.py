import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class ChannelAttention(nn.Module):
    """채널 어텐션 모듈"""
    def __init__(self, channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 공유 MLP
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 평균 풀링
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        
        # 최대 풀링
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        
        # 합치기
        out = avg_out + max_out
        
        # 시그모이드 적용
        out = self.sigmoid(out).unsqueeze(-1)
        
        # 채널별 가중치 적용
        return x * out  # 채널별 가중치 적용

class SelfAttention(nn.Module):
    """셀프 어텐션 모듈"""
    def __init__(self, hidden_size, dropout=0.1):
        super(SelfAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # 쿼리, 키, 밸류 변환
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))
        
    def forward(self, x, batch_size, sequence_length):
        # x: [batch*seq_len, hidden_size]
        
        # 원래 형태로 복원
        x_reshaped = x.view(batch_size, sequence_length, -1)  # [batch, seq_len, hidden_size]
        
        # 쿼리, 키, 밸류 변환
        Q = self.query(x_reshaped)  # [batch, seq_len, hidden_size]
        K = self.key(x_reshaped)    # [batch, seq_len, hidden_size]
        V = self.value(x_reshaped)  # [batch, seq_len, hidden_size]
        
        # 어텐션 스코어 계산
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)  # [batch, seq_len, seq_len]
        
        # 마스킹 적용 (필요한 경우)
        # 여기서는 모든 위치에 어텐션을 적용
        
        # 소프트맥스 적용
        attention = self.dropout(F.softmax(energy, dim=-1))  # [batch, seq_len, seq_len]
        
        # 가중치 적용
        x_attended = torch.matmul(attention, V)  # [batch, seq_len, hidden_size]
        
        # 다시 평탄화
        x_attended = x_attended.reshape(-1, x_attended.size(-1))  # [batch*seq_len, hidden_size]
        
        return x_attended

class ContextModule(nn.Module):
    """컨텍스트 모듈 - 시간적 문맥 정보 추출"""
    def __init__(self, in_features):
        super(ContextModule, self).__init__()
        
        # 시간적 문맥 추출을 위한 LSTM
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=in_features // 2,  # 양방향이므로 절반으로
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 출력 투영
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU()
        )
    
    def forward(self, x, batch_size, sequence_length):
        # x: [batch*seq_len, features]
        
        # 원래 형태로 복원
        x_reshaped = x.view(batch_size, sequence_length, -1)  # [batch, seq_len, features]
        
        # LSTM 적용
        x_context, _ = self.lstm(x_reshaped)  # [batch, seq_len, features]
        
        # 다시 평탄화
        x_context = x_context.reshape(-1, x_context.size(-1))  # [batch*seq_len, features]
        
        # 투영
        x_context = self.projection(x_context)
        
        # 잔차 연결
        return x + x_context

class CNNBlock(nn.Module):
    """CNN 블록 (컨볼루션 + 배치 정규화 + 활성화 함수 + 채널 어텐션)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, use_attention=True, reduction_ratio=8):
        super(CNNBlock, self).__init__()
        
        # 패딩이 지정되지 않은 경우 자동 계산
        if padding is None:
            padding = kernel_size // 2
        
        self.use_attention = use_attention
        
        # 컨볼루션 레이어
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        
        # 배치 정규화
        self.bn = nn.BatchNorm1d(out_channels)
        
        # 활성화 함수
        self.relu = nn.LeakyReLU(inplace=True)
        
        # 채널 어텐션 (선택적)
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels, reduction_ratio)
    
    def forward(self, x):
        # 컨볼루션 + 배치 정규화 + 활성화 함수
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # 채널 어텐션 적용 (선택적)
        if self.use_attention:
            x = self.channel_attention(x)
        
        return x

class AudioNetV3(nn.Module):
    def __init__(self, sample_num=2400, microphone_num=4, output_num=7, config=None):
        super(AudioNetV3, self).__init__()
        print(f"AudioNetV3 초기화 시작: sample_num={sample_num}, microphone_num={microphone_num}, output_num={output_num}")
        
        # config 통합
        self.config = config
        self.use_amp = config.use_amp if config else True
        self.dropout_rate = config.dropout_rate if config else 0.2
        
        # 실제 샘플 수 계산 (4배 다운샘플링 적용)
        actual_sample_num = sample_num // 4
        print(f"실제 사용 샘플 수: {actual_sample_num} (4배 다운샘플링 적용)")
        
        # 커널 크기 설정 - config에서 가져오기
        if config and hasattr(config, 'kernel_sizes'):
            self.kernel_sizes = {
                'conv1': config.kernel_sizes[0],
                'conv2': config.kernel_sizes[1],
                'conv3': config.kernel_sizes[2]
            }
        else:
            self.kernel_sizes = {
                'conv1': 9,  # 7에서 9로 증가
                'conv2': 7,  # 5에서 7로 증가
                'conv3': 5   # 3에서 5로 증가
            }
        
        # 채널 어텐션 reduction ratio 설정
        self.channel_attention_reduction_ratio = config.channel_attention_reduction_ratio if config and hasattr(config, 'channel_attention_reduction_ratio') else 8
        
        # 셀프 어텐션 드롭아웃 설정
        self.self_attention_dropout = config.self_attention_dropout if config and hasattr(config, 'self_attention_dropout') else 0.1
        
        # 컨텍스트 모듈 사용 여부
        self.use_context_module = config.use_context_module if config and hasattr(config, 'use_context_module') else True
        
        # 위치-회전 연결 사용 여부
        self.use_position_for_rotation = config.use_position_for_rotation if config and hasattr(config, 'use_position_for_rotation') else True
        
        # 2채널 최적화 설정
        self.optimize_for_two_channel = config.optimize_for_two_channel if config and hasattr(config, 'optimize_for_two_channel') else False
        
        # 2채널 최적화가 활성화되어 있고 microphone_num이 2인 경우
        if self.optimize_for_two_channel and microphone_num == 2:
            print("2채널 최적화 모드 활성화")
            # 2채널 최적화 로직 구현
            # 1. 커널 크기 조정 - 더 큰 커널 사용하여 공간 정보 보완
            self.kernel_sizes = {
                'conv1': 11,  # 9에서 11로 증가
                'conv2': 9,   # 7에서 9로 증가
                'conv3': 7    # 5에서 7로 증가
            }
            
            # 2. 채널 어텐션 강화 - 채널 간 정보 교환 강화
            self.channel_attention_reduction_ratio = 4  # 8에서 4로 감소
            
            # 3. 드롭아웃 감소 - 더 많은 정보 유지
            self.dropout_rate = 0.1  # 0.2에서 0.1로 감소
            
            # 4. 셀프 어텐션 강화
            self.self_attention_dropout = 0.05  # 0.1에서 0.05로 감소
            
            # 5. 특성 맵 수 조정 - 2채널에 맞게 최적화 (139MB에 맞춤)
            self.feature_maps = [112, 224, 576]  # 96, 192, 512에서 조정
            
            print("2채널 최적화 설정 완료:")
            print(f"- 커널 크기 증가: {self.kernel_sizes}")
            print(f"- 채널 어텐션 강화: reduction ratio {self.channel_attention_reduction_ratio}")
            print(f"- 드롭아웃 감소: {self.dropout_rate}")
            print(f"- 셀프 어텐션 강화: dropout {self.self_attention_dropout}")
            print(f"- 특성 맵 수 조정: {self.feature_maps}")
        else:
            # 기본 특성 맵 수 설정 (139MB에 맞춤)
            self.feature_maps = [88, 176, 608]  # 96, 192, 640에서 조정
        
        # CNN 특징 추출기 - 채널 어텐션 통합
        self.cnn_blocks = nn.ModuleList([
            # 첫 번째 블록 - stride 증가
            CNNBlock(
                microphone_num, 
                self.feature_maps[0], 
                kernel_size=self.kernel_sizes['conv1'], 
                stride=4, 
                use_attention=True,
                reduction_ratio=self.channel_attention_reduction_ratio
            ),
            
            # 두 번째 블록 - stride 증가
            CNNBlock(
                self.feature_maps[0], 
                self.feature_maps[1], 
                kernel_size=self.kernel_sizes['conv2'], 
                stride=4, 
                use_attention=True,
                reduction_ratio=self.channel_attention_reduction_ratio
            ),
            
            # 세 번째 블록 - 마지막 레이어
            CNNBlock(
                self.feature_maps[1], 
                self.feature_maps[2], 
                kernel_size=self.kernel_sizes['conv3'], 
                stride=1, 
                use_attention=True,
                reduction_ratio=self.channel_attention_reduction_ratio
            )
        ])
        
        # 적응형 풀링
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC 레이어 - 적절한 크기 설정
        self.fc = nn.Sequential(
            nn.Linear(self.feature_maps[2], self.feature_maps[2] * 2, bias=False),
            nn.BatchNorm1d(self.feature_maps[2] * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )
        
        # LSTM - 적절한 크기 설정 (크기 감소)
        lstm_hidden_size = self.feature_maps[2]
        if self.optimize_for_two_channel and microphone_num == 2:
            lstm_hidden_size = int(self.feature_maps[2] * 1.25)  
        
        self.lstm = nn.LSTM(
            input_size=self.feature_maps[2] * 2,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )
        
        # 셀프 어텐션 모듈 추가
        self.self_attention = SelfAttention(self.feature_maps[2] * 2, self.self_attention_dropout)
        
        # 컨텍스트 모듈 추가 
        if self.use_context_module:
            self.context_module = ContextModule(self.feature_maps[2] * 2)
        
        # 출력 레이어 분리 (위치와 회전)
        # 셀프 어텐션과 컨텍스트 모듈을 통과한 후의 특성 크기는 feature_maps[2] * 2
        self.fc_position = nn.Linear(self.feature_maps[2] * 2, 3)
        
        # 위치 정보를 회전 예측에 활용 
        if self.use_position_for_rotation:
            self.fc_rotation = nn.Sequential(
                nn.Linear(self.feature_maps[2] * 2 + 3, 384),  
                nn.BatchNorm1d(384),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(384, 4)
            )
        else:
            self.fc_rotation = nn.Linear(self.feature_maps[2] * 2, 4)
        
        # 가중치 초기화
        self._initialize_weights()
        print("AudioNetV3 초기화 완료")
        
        # 모델 구성 요약 출력
        print("\n=== AudioNetV3 구성 요약 ===")
        print(f"커널 크기: {self.kernel_sizes}")
        print(f"채널 어텐션 reduction ratio: {self.channel_attention_reduction_ratio}")
        print(f"셀프 어텐션 드롭아웃: {self.self_attention_dropout}")
        print(f"컨텍스트 모듈 사용: {self.use_context_module}")
        print(f"위치-회전 연결 사용: {self.use_position_for_rotation}")
        print(f"2채널 최적화: {self.optimize_for_two_channel}")
        print("===========================\n")
    
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
        sequence_length = x.size(3)  
        
        # Mixed Precision 사용
        with torch.amp.autocast('cuda', enabled=self.use_amp): 
            # 데이터 형태 변환 (batch, channels, samples, seq_len) -> (batch*seq_len, channels, samples)
            x = x.permute(0, 3, 1, 2).contiguous()  # (batch, seq_len, channels, samples)
            x = x.view(-1, x.size(2), x.size(3))    # (batch*seq_len, channels, samples)
            
            # CNN 블록 순차 처리
            for cnn_block in self.cnn_blocks:
                x = cnn_block(x)
            
            # 적응형 풀링
            x = self.adaptive_pool(x)
            x = x.squeeze(-1)  # (batch*seq_len, 704)
            
            # FC 레이어
            x = self.fc(x)  # (batch*seq_len, 1408)
            
            # 셀프 어텐션 적용
            x = self.self_attention(x, batch_size, sequence_length)
            
            # 컨텍스트 모듈 적용 (사용 여부에 따라)
            if self.use_context_module:
                x = self.context_module(x, batch_size, sequence_length)
            
            # 위치 예측
            pos_pred = self.fc_position(x)  # (batch*seq_len, 3)
            
            # 회전 예측 (위치 정보 활용 여부에 따라)
            if self.use_position_for_rotation:
                # 위치 정보와 특성 결합
                combined = torch.cat([x, pos_pred], dim=1)
                rot_pred = self.fc_rotation(combined)  # (batch*seq_len, 4)
            else:
                rot_pred = self.fc_rotation(x)  # (batch*seq_len, 4)
            
            # 쿼터니언 정규화
            rot_pred = F.normalize(rot_pred, p=2, dim=1)
            
            # 출력 형태 변환 (batch*seq_len, dim) -> (batch, dim)
            pos_pred = pos_pred.view(batch_size, sequence_length, -1)[:, -1, :]  # 마지막 시퀀스만 사용
            rot_pred = rot_pred.view(batch_size, sequence_length, -1)[:, -1, :]  # 마지막 시퀀스만 사용
        
        return pos_pred, rot_pred
    
    def get_loss(self, pred, target):
        pos_pred, quat_pred = pred
        pos_target, quat_target = target[:, :3], target[:, 3:]
        
        # 위치 손실 (동일)
        pos_loss = F.l1_loss(pos_pred, pos_target)
        
        # 쿼터니언 정규화
        quat_pred = F.normalize(quat_pred, p=2, dim=1)
        quat_target = F.normalize(quat_target, p=2, dim=1)
        
        # 양방향 쿼터니언 손실 계산 (double cover 고려)
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