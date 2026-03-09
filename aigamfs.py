import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import os
import time

# ======================
# 2. 模型架构 (FIXED FOR 181x360 INPUT AND SKIP CONNECTIONS)
# ======================
class CubeEmbedding(nn.Module):
    """立方体嵌入模块"""
    def __init__(self, in_channels=60, hidden_dim=640):
        super().__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 复合卷积网络 (3层)
        self.conv_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(3):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(current_dim, current_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(current_dim, current_dim*2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(current_dim*2, current_dim*2, kernel_size=1),
                nn.ReLU(),
                nn.MaxPool2d(2)  # 空间降采样
            ))
            current_dim *= 2
        
        # 最终维度
        self.final_dim = current_dim
    
    def forward(self, x):
        # x: [batch, channels, lat, lon]
        x = self.conv1(x)
        
        # 通过3层复合卷积
        skip_connections = []
        for i, layer in enumerate(self.conv_layers):
            skip_connections.append(x)  # 保存跳跃连接
            x = layer(x)
        
        # 返回下采样后的空间尺寸
        spatial_size = (x.shape[2], x.shape[3])
        return x, skip_connections, spatial_size


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 生成Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out)


class ViTBlock(nn.Module):
    """单个ViT编码器层"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 自注意力块
        x = x + self.dropout1(self.attn(self.norm1(x)))
        # MLP块
        x = x + self.mlp(self.norm2(x))
        return x


class ViTModule(nn.Module):
    """ViT模块 - 14层堆叠"""
    def __init__(self, input_dim=5120, embed_dim=512, num_heads=16, num_layers=14, dropout=0.1):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # 位置编码 (动态初始化)
        self.embed_dim = embed_dim
        # self.pos_embed = None
        # self.register_buffer('pos_embed', None)
        
        # ViT层
        self.layers = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影 (恢复到原始维度)
        self.output_proj = nn.Linear(embed_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    # def _init_pos_embed(self, seq_len, device):
    #     """动态初始化位置编码"""
    #     if self.pos_embed is None or self.pos_embed.shape[1] != seq_len:
    #         pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.embed_dim, device=device))
    #         nn.init.normal_(pos_embed, std=0.02)
    #         self.pos_embed = pos_embed
    #         return pos_embed
    #     return self.pos_embed
    


    def forward(self, x, spatial_size):
        """
        x: [batch, channels, H, W] - 下采样后的特征图
        spatial_size: (H, W) - 下采样后的空间尺寸
        """
        batch_size = x.shape[0]
        device = x.device
        H, W = spatial_size
        seq_len = H * W
        
        # 展平空间维度: [batch, C, H, W] -> [batch, C, H*W] -> [batch, H*W, C]
        x = x.view(batch_size, x.shape[1], -1).transpose(1, 2)  # [batch, seq_len, input_dim]
        
        # 初始化或获取位置编码
        # pos_embed = self._init_pos_embed(seq_len, device)
        
        # 投影到嵌入空间
        x = self.input_proj(x)  # [batch, seq_len, embed_dim]
        
        # 添加位置编码
        # x = x + pos_embed
        
        # 通过ViT层
        for layer in self.layers:
            x = layer(x)
        
        # 投影回原始维度
        x = self.output_proj(x)
        
        # 恢复空间维度: [batch, seq_len, C] -> [batch, C, H, W]
        x = x.transpose(1, 2).view(batch_size, -1, H, W)
        
        return x
    


class CubeUnembedding(nn.Module):
    """立方体反嵌入模块 (FIXED FOR CORRECT CHANNEL MATCHING)"""
    def __init__(self, final_dim=5120, hidden_dim=640, out_channels=54):
        super().__init__()
        # 转置卷积上采样 - 修正通道数匹配
        self.up_layers = nn.ModuleList()
        
        # 第1层上采样 (5120 -> 2560)
        self.up_layers.append(nn.Sequential(
            nn.ConvTranspose2d(final_dim, final_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_dim//2, final_dim//2, kernel_size=3, padding=1),
            nn.ReLU()
        ))
        
        # 第2层上采样 (2560*2=5120 -> 1280)
        self.up_layers.append(nn.Sequential(
            nn.ConvTranspose2d(final_dim, final_dim//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_dim//4, final_dim//4, kernel_size=3, padding=1),
            nn.ReLU()
        ))
        
        # 第3层上采样 (1280*2=2560 -> 640)
        self.up_layers.append(nn.Sequential(
            nn.ConvTranspose2d(final_dim//2, final_dim//8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_dim//8, final_dim//8, kernel_size=3, padding=1),
            nn.ReLU()
        ))
        
        # 最终输出卷积 (640*2=1280 -> out_channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, skip_connections):
        # 通过3层上采样 (匹配3次下采样)
        for i, layer in enumerate(self.up_layers):
            x = layer(x)
            # 添加跳跃连接 (U-Net结构)
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                # 调整尺寸匹配
                if skip.shape[2:] != x.shape[2:]:
                    skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                # 拼接通道维度
                x = torch.cat([x, skip], dim=1)
        
        # 最终输出
        x = self.final_conv(x)
        return x


class AI_GAMFS(nn.Module):
    """AI-GAMFS完整模型 (去除时间特征)"""
    def __init__(self, in_channels=220, hidden_dim=256, out_channels=110, dropout=0.15):
        super().__init__()
        # 合并输入通道 (只有气象场)
        self.total_channels = in_channels
        
        # 1. Cube Embedding
        self.cube_embed = CubeEmbedding(self.total_channels, hidden_dim)
        
        # 2. ViT模块
        self.vit = ViTModule(
            input_dim=hidden_dim * 8,  # 8x downsample后的通道数
            embed_dim=32,
            num_heads=8,
            num_layers=4,
            dropout=dropout
        )
        
        # 3. Cube Unembedding - 传入正确的维度
        self.cube_unembed = CubeUnembedding(
            final_dim=hidden_dim * 8,  # 5120
            hidden_dim=hidden_dim,     # 640
            out_channels=out_channels
        )
        
        # 边界填充 (181x360 -> 尺寸需要是8的倍数，因为有3次下采样)
        # 181 -> 184 (181 + 3)
        # 360 已经是8的倍数 (360/8=45)
        self.pad_top = 1
        self.pad_bottom = 2
        self.pad_left = 0
        self.pad_right = 0
    
    def forward(self, x):
        # x: [batch, channels, 181, 360]  (无时间特征)
        
        # 检查输入尺寸
        batch_size = x.shape[0]
        _, channels, lat, lon = x.shape
        
        # 验证输入尺寸
        assert lat == 181 and lon == 360, f"Expected input size (181, 360), got ({lat}, {lon})"
        
        # 空间填充 (181 -> 184)
        x = nn.functional.pad(x, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), mode='constant', value=0)
        
        # 1. Cube Embedding - 现在返回空间尺寸
        x, skip_connections, spatial_size = self.cube_embed(x)  # 现在尺寸是 [batch, 5120, 23, 45]
        
        # 2. ViT处理 - 传入空间尺寸
        x = self.vit(x, spatial_size)  # 保持 [batch, 5120, 23, 45]
        
        # 3. Cube Unembedding
        x = self.cube_unembed(x, skip_connections)
        
        # 移除填充 - 确保精确匹配原始尺寸
        x = x[:, :, self.pad_top:x.shape[2]-self.pad_bottom, self.pad_left:x.shape[3]-self.pad_right]
        
        # 最终验证输出尺寸
        assert x.shape[2:] == (181, 360), f"Output size mismatch: expected (181, 360), got {x.shape[2:]}"
        
        return x


# ======================
# 3. 训练函数 (无时间特征)
# ======================
def train_ai_gamfs(model, dataloader, optimizer, scheduler, device, epochs=80):
    """训练单个基础模型(3h/6h/9h/12h)，无时间特征"""
    model.train()
    criterion = nn.L1Loss()  # MAE损失函数
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (X, y) in enumerate(dataloader):  # 假设dataloader只返回输入和目标
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(X)  # 只传入X
            
            # 计算损失
            loss = criterion(outputs, y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
        
        # 学习率调度
        scheduler.step()
        
        epoch_loss /= len(dataloader)
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} completed. Avg Loss: {epoch_loss:.6f}, Time: {elapsed:.2f}s')
        
        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'ai_gamfs_model_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, 'ai_gamfs_final_model.pth')
    
    return model


# ======================
# 4. 中继预报函数 (无时间特征)
# ======================
def relay_forecast(models, initial_input, lead_times=[3,6,9,12], max_forecast=120):
    """
    执行级联中继预报 (无时间特征)
    models: 按[3h,6h,9h,12h]顺序排列的模型列表
    lead_times: 模型对应的预报时效(小时)
    max_forecast: 最大预报时效(小时)，默认120(5天)
    """
    forecast_steps = max_forecast // 3  # 3小时间隔
    current_input = initial_input
    forecast_results = []
    
    step = 0
    while step < forecast_steps:
        remaining_steps = forecast_steps - step
        
        # 选择适当的模型
        if remaining_steps >= 4:  # 12小时模型
            model_idx = 3
        elif remaining_steps >= 3:  # 9小时模型
            model_idx = 2
        elif remaining_steps >= 2:  # 6小时模型
            model_idx = 1
        else:  # 3小时模型
            model_idx = 0
        
        # 执行预报 (只传入当前场)
        with torch.no_grad():
            forecast = models[model_idx](current_input)
        
        # 保存结果
        forecast_results.append(forecast.cpu())
        
        # 更新输入
        current_input = forecast
        step += lead_times[model_idx] // 3
    
    return torch.stack(forecast_results)


if __name__ == "__main__":
    # 1. 设置设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {DEVICE}")
    
    # 2. 创建测试用的随机数据 (181x360)
    batch_size = 8
    input_channels = 220  
    height, width  = 181, 360
    
    # 随机生成一个初始气象状态
    initial_input = torch.randn(batch_size, input_channels, height, width).to(DEVICE)
    print(f"Initial input shape: {initial_input.shape}")

    # 3. 创建模型 (无时间特征)
    model = AI_GAMFS().to(DEVICE)
    
    # 4. 执行单步前向传播测试 (不再需要时间特征)
    print("\nTesting single-step forecast with 181x360 input...")
    start_time = time.time()
    with torch.no_grad():
        output = model(initial_input)
        print(f"Output shape: {output.shape}")
    elapsed = time.time() - start_time
    print(f"Single-step forecast completed in {elapsed:.4f} seconds")
    print(f"Output statistics: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    
    # # 5. 保存测试结果 (可选)
    # torch.save(output.cpu(), "test_output_181x360.pt")
    # print("Test output saved to 'test_output_181x360.pt'")
    
    # print("\nFramework test with 181x360 input (no time features) completed successfully!")
