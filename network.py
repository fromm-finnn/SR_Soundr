import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class ChannelAttention(nn.Module):
    """채널 어텐션 모듈"""
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 채널 수가 적을 경우를 대비해 최소 reduction 값 설정
        reduction_channels = max(in_channels // reduction_ratio, 4)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch, channels, time]
        batch_size, channels, _ = x.size()
        
        # 평균 풀링
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        # 최대 풀링
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        
        # 결합 및 시그모이드 적용
        out = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1)
        return x * out  # 채널별 가중치 적용

class SelfAttention(nn.Module):
    """셀프 어텐션 모듈"""
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = x.shape
        
        # 쿼리, 키, 밸류 변환
        Q = self.query(x)  # [batch_size, seq_len, hidden_size]
        K = self.key(x)    # [batch_size, seq_len, hidden_size]
        V = self.value(x)  # [batch_size, seq_len, hidden_size]
        
        # 어텐션 스코어 계산
        energy = torch.bmm(Q, K.permute(0, 2, 1)) / self.scale.to(x.device)  # [batch_size, seq_len, seq_len]
        
        # 마스킹 적용 (필요한 경우)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # 어텐션 가중치 계산
        attention = torch.softmax(energy, dim=2)  # [batch_size, seq_len, seq_len]
        attention = self.dropout(attention)
        
        # 컨텍스트 벡터 계산
        context = torch.bmm(attention, V)  # [batch_size, seq_len, hidden_size]
        
        return context, attention

class ContextModule(nn.Module):
    """글로벌/로컬 컨텍스트 모듈"""
    def __init__(self, hidden_size):
        super(ContextModule, self).__init__()
        self.global_context = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True)
        )
        self.local_context = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(inplace=True)
        )
        self.combined_context = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, global_x, local_x):
        # global_x: [batch_size, hidden_size] (마지막 시간 단계)
        # local_x: [batch_size, hidden_size] (어텐션 가중 평균)
        
        # 글로벌 컨텍스트 처리
        global_ctx = self.global_context(global_x)
        
        # 로컬 컨텍스트 처리
        local_ctx = self.local_context(local_x)
        
        # 컨텍스트 결합
        combined = torch.cat([global_ctx, local_ctx], dim=1)
        return self.combined_context(combined)

class CNNBlock(nn.Module):
    """확장된 시간적 수용 영역을 가진 CNN 블록"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_attention=True):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size//2, 
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.use_attention = use_attention
        
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
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
            # 여기에 2채널 최적화 로직 추가 가능
        
        # CNN 특징 추출기 - 채널 어텐션 통합
        self.cnn_blocks = nn.ModuleList([
            # 첫 번째 블록 - stride 증가
            CNNBlock(
                microphone_num, 
                96, 
                kernel_size=self.kernel_sizes['conv1'], 
                stride=4, 
                use_attention=True
            ),
            
            # 두 번째 블록 - stride 증가
            CNNBlock(
                96, 
                192, 
                kernel_size=self.kernel_sizes['conv2'], 
                stride=4, 
                use_attention=True
            ),
            
            # 세 번째 블록 - 마지막 레이어
            CNNBlock(
                192, 
                704, 
                kernel_size=self.kernel_sizes['conv3'], 
                stride=1, 
                use_attention=True
            )
        ])
        
        # 적응형 풀링
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC 레이어 - 적절한 크기 설정
        self.fc = nn.Sequential(
            nn.Linear(704, 1408, bias=False),  # 704는 features의 출력 채널 수
            nn.BatchNorm1d(1408),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )
        
        # LSTM - 적절한 크기 설정
        self.lstm = nn.LSTM(
            input_size=1408,
            hidden_size=704,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )
        
        # 셀프 어텐션 모듈 추가
        self.self_attention = SelfAttention(1408, self.self_attention_dropout)
        
        # 컨텍스트 모듈 추가 (사용 여부에 따라)
        if self.use_context_module:
            self.context_module = ContextModule(1408)
        
        # 출력 레이어 분리 (위치와 회전)
        self.fc_position = nn.Linear(1408, 3)  # 양방향 LSTM이므로 704*2=1408
        
        # 위치 정보를 회전 예측에 활용 (사용 여부에 따라)
        if self.use_position_for_rotation:
            self.fc_rotation = nn.Sequential(
                nn.Linear(1408 + 3, 512),  # 위치 정보(3) 추가
                nn.BatchNorm1d(512),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, 4)
            )
        else:
            self.fc_rotation = nn.Linear(1408, 4)
        
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
        sequence_length = x.size(3)  # 데이터 형태가 (batch, channels=4, samples=1200, seq_len=20)
        
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
            
            # 원래 배치 및 시퀀스 형태로 복원
            x = x.view(batch_size, sequence_length, -1)  # (batch, seq_len, 1408)
            
            # LSTM 처리
            lstm_out, hidden = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size*2)
            
            # 셀프 어텐션 적용
            attended_seq, attention_weights = self.self_attention(lstm_out)
            
            # 글로벌 컨텍스트 (마지막 시점)
            global_context = lstm_out[:, -1]  # (batch, hidden_size*2)
            
            # 로컬 컨텍스트 (어텐션 가중 평균)
            attention_mean = attention_weights.mean(dim=2, keepdim=True)  # [batch_size, seq_len, 1]
            local_context = torch.sum(
                attended_seq * attention_mean, 
                dim=1
            )  # (batch, hidden_size*2)
            
            # 컨텍스트 결합 (사용 여부에 따라)
            if self.use_context_module:
                combined_context = self.context_module(global_context, local_context)
            else:
                combined_context = global_context
            
            # 위치 예측
            position = self.fc_position(combined_context)
            
            # 위치 정보를 회전 예측에 활용 (사용 여부에 따라)
            if self.use_position_for_rotation:
                rotation_input = torch.cat([combined_context, position], dim=1)
                rotation = self.fc_rotation(rotation_input)
            else:
                rotation = self.fc_rotation(combined_context)
            
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

# 기존 AudioNet 클래스를 유지하고 AudioNetV3를 새로 추가
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