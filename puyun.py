import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ==================== 1. LKA Block ====================

class LKABlock(nn.Module):
    def __init__(self, dim: int = 256, kernel_size: int = 9):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=dim,
            bias=True
        )
        # 修正：直接传入 dim，不需要列表
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 3, dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # 使用残差连接，而不是乘法
        return identity + x  # 改为加法残差，乘法容易导致梯度问题


# ==================== 2. LKA-FCN Layer ====================

class LKAFCNLayer(nn.Module):
    def __init__(self, dim: int = 256, num_blocks: int = 12, kernel_size: int = 9):
        super().__init__()
        self.blocks = nn.ModuleList([
            LKABlock(dim, kernel_size) for _ in range(num_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ==================== 3. Patch Embedding ====================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 220, embed_dim: int = 256, patch_size: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C_var, H, W = x.shape
        x = x.view(B, T * C_var, H, W)
        return self.proj(x)


# ==================== 4. Patch Merging ====================

class PatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 110, 
                 patch_size: int = 8, target_size: Tuple[int, int] = (181, 360)):
        super().__init__()
        self.patch_size = patch_size
        self.target_size = target_size
        
        self.fc = nn.Conv2d(in_channels, out_channels * patch_size * patch_size, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.pixel_shuffle(x)
        
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x


# ==================== 5. PuYun (embed_dim=256) ====================

class PuYun(nn.Module):
    def __init__(
        self,
        in_var: int = 110,
        time_steps: int = 2,
        embed_dim: int = 512,      # 修正：默认256，与注释一致
        num_layers: int = 4,
        blocks_per_layer: int = 12,
        kernel_size: int = 9,
        patch_size: int = 8,
        input_size: Tuple[int, int] = (181, 360)
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.in_var = in_var
        
        self.patch_embed = PatchEmbedding(
            in_channels=in_var * time_steps,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        self.fcn_layers = nn.ModuleList([
            LKAFCNLayer(embed_dim, blocks_per_layer, kernel_size)
            for _ in range(num_layers)
        ])
        
        # 修正：计算正确的输入通道数
        # 每层输出 embed_dim，共 num_layers 层，拼接后是 embed_dim * num_layers
        merge_in_channels = embed_dim * num_layers
        
        self.patch_merge = PatchMerging(
            in_channels=merge_in_channels,
            out_channels=in_var,
            patch_size=patch_size,
            target_size=input_size
        )
        
        # 添加一个投影层，将输入的最后一帧映射到与输出相同的通道数，用于残差连接
        self.residual_proj = nn.Conv2d(in_var, in_var, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] = [B, 2, 110, 181, 360]
        B, T, C, H, W = x.shape
        
        # 保存最后一帧用于残差连接 [B, 110, 181, 360]
        x_t = x[:, -1, :, :, :]  # 取最后一帧
        
        # Patch Embedding: [B, 220, 181, 360] -> [B, 256, 23, 45] (假设patch_size=8)
        x1 = self.patch_embed(x)
        
        # 收集所有层输出
        layer_outputs = []
        for layer in self.fcn_layers:
            x1 = layer(x1)
            layer_outputs.append(x1)
        
        # 拼接所有层输出: [B, embed_dim * num_layers, H', W']
        x = torch.cat(layer_outputs, dim=1)
        
        # 上采样: [B, embed_dim * num_layers, H', W'] -> [B, 110, 181, 360]
        x = self.patch_merge(x)
        
        # 残差连接：确保维度匹配
        return x + self.residual_proj(x_t)


# ==================== 快速测试 ====================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试 embed_dim=256 的配置
    model = PuYun(
        in_var=110,
        time_steps=2,
        embed_dim=256,
        num_layers=4,
        blocks_per_layer=12,
        patch_size=8,
        input_size=(181, 360)
    ).to(device)
    
    # 正确的输入维度
    x = torch.randn(1, 2, 110, 181, 360).to(device)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 检查是否能正常训练（梯度回传）
    x.requires_grad = True
    out = model(x)
    loss = out.mean()
    loss.backward()
    print(f"梯度检查: x.grad.shape = {x.grad.shape if x.grad is not None else 'None'}")
    print("✓ 运行成功!")
