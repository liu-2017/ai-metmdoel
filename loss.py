import torch
import torch.nn as nn
import numpy as np

class LatitudeWeightedL1Loss(nn.Module):
    def __init__(self, num_latitudes=721):
        super().__init__()
        # 1. 生成纬度权重a_i：随纬度升高而减小，采用气象模型常用的余弦加权
        # 纬度范围：-90°到90°（共721个网格点，0.25°分辨率）
        latitudes = np.linspace(-np.pi/2, np.pi/2, num_latitudes)  # 弧度制转换
        self.lat_weights = torch.tensor(np.cos(latitudes), dtype=torch.float32)  # 余弦加权（赤道1，极地0）
        self.lat_weights = self.lat_weights / self.lat_weights.sum() * num_latitudes  # 归一化：权重总和=纬度数，保证整体权重均衡

    def forward(self, pred, target):
        """
        计算纬度加权L1损失
        Args:
            pred: 预测值 [batch_size, C, H, W]，C=变量数(70)，H=纬度数(721)，W=经度数(1440)
            target: 真实值 [batch_size, C, H, W]（与pred维度一致）
        Returns:
            loss: 纬度加权L1损失值（标量）
        """
        # 2. 计算每个网格点的绝对误差
        abs_error = torch.abs(pred - target)
        
        # 3. 应用纬度权重：对H维度（纬度）加权
        # 扩展权重维度以匹配输入：[H] → [1, 1, H, 1]，适配batch、C、W维度广播
        weighted_error = abs_error * self.lat_weights[None, None, :, None].to(pred.device)
        
        # 4. 对所有维度（batch、C、H、W）取平均，得到最终损失
        total_loss = weighted_error.mean()
        
        return total_loss

# 测试代码（验证逻辑正确性）
if __name__ == "__main__":

    batch_size, C, H, W = 1, 109, 181, 360
    pred = torch.randn(batch_size, C, H, W)
    target = torch.randn(batch_size, C, H, W)
    # 初始化损失函数
    loss_fn = LatitudeWeightedL1Loss(num_latitudes=181)
    # 计算损失
    loss = loss_fn(pred, target)
    print(f"纬度加权L1损失值：{loss.item():.4f}")
