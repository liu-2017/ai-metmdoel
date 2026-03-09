import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MixedLevelWeatherNormalizer:
    """
    混合层数气象数据归一化处理器
    处理变量: 
      - 21层变量: u, v, r, t, gh (各21层)
      - 1层变量: 2t, 10u, 10v, prmsl (各1层)
    总通道数: 5*21 + 5*1 = 110
    数据形状: (110, 181, 360)
    """
    
    def __init__(self):
        """初始化处理器，定义变量及其层数"""
        # 定义变量及其层数
        self.variable_specs = {
            # 21层变量
            'u': 21,    # 水平风速U分量
            'v': 21,    # 水平风速V分量  
            'r': 21,    # 相对湿度
            't': 21,    # 温度
            'gh': 21,   # 位势高度
            
            # 1层变量
            '2t': 1,    # 2米温度
            '10u': 1,   # 10米U风速
            '10v': 1,   # 10米V风速
            'prmsl': 1, # 海平面气压
            'tsk': 1, # 海平面气压
        }
        
        # 计算总通道数和各变量通道索引
        self.total_channels = 0
        self.var_channel_indices = {}
        self.var_levels = {}
        
        for var, levels in self.variable_specs.items():
            start_idx = self.total_channels
            end_idx = self.total_channels + levels
            self.var_channel_indices[var] = slice(start_idx, end_idx)
            self.var_levels[var] = levels
            self.total_channels += levels
        
        # 网格参数
        self.lats = 181
        self.lons = 360
        
        # 存储统计量
        self.means = {}
        self.stds = {}
        self.is_fitted = False
        
        # 打印配置信息
        # print("混合层数气象数据归一化处理器初始化完成:")
        # print(f"总通道数: {self.total_channels}")
        # print(f"数据形状: ({self.total_channels}, {self.lats}, {self.lons})")
        # print("\n变量配置:")
        for var in self.variable_specs:
            idx = self.var_channel_indices[var]
            # print(f"  {var:>5}: {self.variable_specs[var]:2}层, 通道索引 {idx.start:3}:{idx.stop:3}")
    
    def extract_variable(self, fused_data: np.ndarray, variable: str) -> np.ndarray:
        """
        从融合数据中提取指定变量
        
        参数:
            fused_data: 融合数据，形状 (110, 181, 360)
            variable: 变量名称
            
        返回:
            提取的数据，形状 (n_levels, 181, 360)
        """
        if fused_data.shape != (self.total_channels, self.lats, self.lons):
            raise ValueError(f"数据形状应为 ({self.total_channels}, {self.lats}, {self.lons})，但得到 {fused_data.shape}")
        
        if variable not in self.var_channel_indices:
            raise ValueError(f"未知变量: {variable}，可用变量: {list(self.variable_specs.keys())}")
        
        idx_slice = self.var_channel_indices[variable]
        return fused_data[idx_slice].copy()
    
    def compute_statistics(self, data_dir: str, pattern: str = "*.npy", 
                          sample_limit: Optional[int] = None) -> None:
        """
        从NPY文件计算统计量
        
        参数:
            data_dir: 数据目录
            pattern: 文件匹配模式
            sample_limit: 限制处理的样本数（用于测试）
        """
        print(f"\n开始计算统计量...")
        
        # 初始化累积变量
        sum_arrays = {var: np.zeros((self.var_levels[var], 1, 1)) for var in self.variable_specs}
        sum_sq_arrays = {var: np.zeros((self.var_levels[var], 1, 1)) for var in self.variable_specs}
        total_samples = 0
        
        # 获取数据文件
        data_files = list(sorted(Path(data_dir).glob(pattern)))
        
        if not data_files:
            raise FileNotFoundError(f"在 {data_dir} 中未找到匹配 {pattern} 的文件")
        
        print(f"找到 {len(data_files)} 个数据文件")
        
        if sample_limit:
            data_files = data_files[:sample_limit]
            print(f"将处理前 {sample_limit} 个文件用于测试")
        
        # 处理每个文件
        for i, file_path in enumerate(data_files):
            if (i + 1) % max(1, len(data_files)//10) == 0 or i == len(data_files) - 1:
                print(f"处理文件 {i+1}/{len(data_files)}: {file_path.name}")
            
            try:
                # 加载数据
                fused_data = np.load(file_path)
                
                # 验证形状
                if fused_data.shape != (self.total_channels, self.lats, self.lons):
                    print(f"警告: 文件 {file_path.name} 形状为 {fused_data.shape}，跳过")
                    continue
                
                # 处理每个变量
                for var in self.variable_specs:
                    var_data = self.extract_variable(fused_data, var)
                    levels = self.var_levels[var]
                    
                    # 对每层独立计算
                    for level in range(levels):
                        level_data = var_data[level] if levels > 1 else var_data
                        sum_arrays[var][level] += np.sum(level_data)
                        sum_sq_arrays[var][level] += np.sum(level_data ** 2)
                
                total_samples += self.lats * self.lons
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        if total_samples == 0:
            raise ValueError("未找到有效数据")
        
        # 计算统计量
        print(f"\n计算最终统计量，基于 {total_samples} 个空间点...")
        
        for var in self.variable_specs:
            levels = self.var_levels[var]
            
            # 计算均值
            self.means[var] = sum_arrays[var] / total_samples
            
            # 计算标准差
            variance = (sum_sq_arrays[var] / total_samples) - (self.means[var] ** 2)
            self.stds[var] = np.sqrt(np.maximum(variance, 1e-12))
            
            # 打印统计信息
            if levels == 1:
                print(f"{var:>6} (1层): 均值={self.means[var][0,0,0]:8.2f}, 标准差={self.stds[var][0,0,0]:8.2f}")
            else:
                print(f"{var:>6} ({levels}层):")
                for level in range(min(3, levels)):  # 只显示前3层
                    mean_val = self.means[var][level, 0, 0]
                    std_val = self.stds[var][level, 0, 0]
                    print(f"        层{level:2d}: 均值={mean_val:8.2f}, 标准差={std_val:8.2f}")
                if levels > 3:
                    print(f"        ... (共{levels}层)")
        
        self.is_fitted = True
        print(f"\n统计量计算完成!")
    
    def normalize_data(self, fused_data: np.ndarray) -> np.ndarray:
        """
        归一化融合数据
        
        参数:
            fused_data: 原始融合数据，形状 (110, 181, 360)
            
        返回:
            归一化后的数据
        """
        if not self.is_fitted:
            raise ValueError("请先计算统计量")
        
        if fused_data.shape != (self.total_channels, self.lats, self.lons):
            raise ValueError(f"数据形状应为 ({self.total_channels}, {self.lats}, {self.lons})，但得到 {fused_data.shape}")
        
        # 创建输出数组
        normalized_data = np.zeros_like(fused_data)
        
        # 对每个变量独立归一化
        for var in self.variable_specs:
            idx_slice = self.var_channel_indices[var]
            var_data = fused_data[idx_slice]
            
            # 应用归一化: (数据 - 均值) / 标准差
            # 注意均值和标准差的形状匹配
            normalized_var_data = (var_data - self.means[var]) / self.stds[var]
            normalized_data[idx_slice] = normalized_var_data
        
        return normalized_data
    
    def denormalize_data(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        反归一化数据
        
        参数:
            normalized_data: 归一化后的数据
            
        返回:
            反归一化到原始物理量纲的数据
        """
        if not self.is_fitted:
            raise ValueError("请先计算统计量")
        
        if normalized_data.shape != (self.total_channels, self.lats, self.lons):
            raise ValueError(f"数据形状错误")
        
        # 创建输出数组
        denormalized_data = np.zeros_like(normalized_data)
        
        # 对每个变量独立反归一化
        for var in self.variable_specs:
            idx_slice = self.var_channel_indices[var]
            var_norm_data = normalized_data[idx_slice]
            
            # 反归一化: 数据 * 标准差 + 均值
            denormalized_var_data = var_norm_data * self.stds[var] + self.means[var]
            denormalized_data[idx_slice] = denormalized_var_data
        
        return denormalized_data
    
    def apply_spherical_padding(self, fused_data: np.ndarray, pad_size: int = 1) -> np.ndarray:
        """
        应用球形边界填充
        
        参数:
            fused_data: 融合数据，形状 (110, 181, 360)
            pad_size: 填充大小
            
        返回:
            填充后的数据，形状 (110, 181+2*pad, 360+2*pad)
        """
        if fused_data.shape[0] != self.total_channels:
            raise ValueError(f"通道数应为 {self.total_channels}")
        
        # 新形状
        new_lats = self.lats + 2 * pad_size
        new_lons = self.lons + 2 * pad_size
        padded_data = np.zeros((self.total_channels, new_lats, new_lons))
        
        # 中心区域
        padded_data[:, pad_size:-pad_size, pad_size:-pad_size] = fused_data
        
        # 1. 经度方向：循环填充
        padded_data[:, pad_size:-pad_size, :pad_size] = fused_data[:, :, -pad_size:]
        padded_data[:, pad_size:-pad_size, -pad_size:] = fused_data[:, :, :pad_size]
        
        # 2. 纬度方向：极地处理
        # 北极
        for i in range(pad_size):
            # 使用最高纬度数据的镜像
            padded_data[:, i, pad_size:-pad_size] = fused_data[:, pad_size - i, :]
        
        # 南极
        for i in range(pad_size):
            south_idx = self.lats - 1 - (pad_size - i - 1)
            padded_data[:, -(i+1), pad_size:-pad_size] = fused_data[:, south_idx, :]
        
        # 3. 四个角区域（使用最近的有效值）
        # 西北角
        padded_data[:, :pad_size, :pad_size] = fused_data[:, pad_size:pad_size+1, -pad_size:]
        # 东北角
        padded_data[:, :pad_size, -pad_size:] = fused_data[:, pad_size:pad_size+1, :pad_size]
        # 西南角
        padded_data[:, -pad_size:, :pad_size] = fused_data[:, -pad_size-1:-pad_size, -pad_size:]
        # 东南角
        padded_data[:, -pad_size:, -pad_size:] = fused_data[:, -pad_size-1:-pad_size, :pad_size]
        
        return padded_data
    
    def save_statistics(self, filepath: str) -> None:
        """保存统计量"""
        stats_dict = {
            'variable_specs': self.variable_specs,
            'total_channels': self.total_channels,
            'lats': self.lats,
            'lons': self.lons,
            'means': {var: self.means[var].tolist() for var in self.variable_specs},
            'stds': {var: self.stds[var].tolist() for var in self.variable_specs},
            'is_fitted': self.is_fitted,
            'var_channel_indices': {
                var: [self.var_channel_indices[var].start, self.var_channel_indices[var].stop]
                for var in self.variable_specs
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"统计量已保存到: {filepath}")
    
    def load_statistics(self, filepath: str) -> None:
        """加载统计量"""
        with open(filepath, 'r') as f:
            stats_dict = json.load(f)
        
        # 恢复配置
        self.variable_specs = stats_dict['variable_specs']
        self.total_channels = stats_dict['total_channels']
        self.lats = stats_dict['lats']
        self.lons = stats_dict['lons']
        
        # 恢复通道索引
        indices_dict = stats_dict['var_channel_indices']
        self.var_channel_indices = {}
        self.var_levels = {}
        
        for var in self.variable_specs:
            start, stop = indices_dict[var]
            self.var_channel_indices[var] = slice(start, stop)
            self.var_levels[var] = self.variable_specs[var]
        
        # 恢复统计量
        self.means = {var: np.array(stats_dict['means'][var]) for var in self.variable_specs}
        self.stds = {var: np.array(stats_dict['stds'][var]) for var in self.variable_specs}
        self.is_fitted = stats_dict['is_fitted']
        
        # print(f"统计量已从 {filepath} 加载")
        # print(f"变量: {list(self.variable_specs.keys())}")
        # print(f"总通道数: {self.total_channels}")


# ==================== 数据加载器 ====================

class MixedLevelDataLoader:
    """混合层数数据加载器"""
    
    def __init__(self, normalizer: MixedLevelWeatherNormalizer, data_dir: str):
        self.normalizer = normalizer
        self.data_dir = Path(data_dir)
        self.file_list = list(sorted(self.data_dir.glob("*.npy")))
        
        if not self.file_list:
            raise FileNotFoundError(f"在 {data_dir} 中未找到NPY文件")
        
        print(f"找到 {len(self.file_list)} 个数据文件")
    
    def create_generator(self, batch_size: int = 4, shuffle: bool = True,
                        normalize: bool = True, apply_padding: bool = False,
                        pad_size: int = 1, infinite: bool = True):
        """
        创建数据生成器
        
        参数:
            batch_size: 批次大小
            shuffle: 是否打乱
            normalize: 是否归一化
            apply_padding: 是否应用边界填充
            pad_size: 填充大小
            infinite: 是否无限循环（用于训练）
        """
        file_indices = list(range(len(self.file_list)))
        
        while True:  # 无限循环
            if shuffle:
                np.random.shuffle(file_indices)
            
            for start_idx in range(0, len(file_indices), batch_size):
                batch_indices = file_indices[start_idx:start_idx + batch_size]
                batch_data = []
                
                for idx in batch_indices:
                    try:
                        # 加载数据
                        data = np.load(self.file_list[idx])
                        
                        # 验证形状
                        if data.shape != (self.normalizer.total_channels, 
                                        self.normalizer.lats, 
                                        self.normalizer.lons):
                            continue
                        
                        # 归一化
                        if normalize:
                            data = self.normalizer.normalize_data(data)
                        
                        # 边界填充
                        if apply_padding:
                            data = self.normalizer.apply_spherical_padding(data, pad_size)
                        
                        batch_data.append(data)
                        
                    except Exception as e:
                        continue
                
                if batch_data:
                    yield np.stack(batch_data, axis=0)
            
            if not infinite:
                break


# ==================== 使用演示 ====================

def main_demo():
    """主演示函数"""
    print("=" * 70)
    print("混合层数气象数据归一化处理器演示")
    print("=" * 70)
    
    # 1. 初始化处理器
    normalizer = MixedLevelWeatherNormalizer()
    
    # 2. 创建演示数据
    # demo_dir = create_demo_data()
    demo_dir = "/home/ubuntu01/AI-MET/train_data/inn"
    
    # 3. 计算统计量
    print("\n" + "-" * 70)
    print("步骤1: 计算统计量")
    print("-" * 70)
    
    normalizer.compute_statistics(demo_dir, pattern="*.npy", sample_limit=None)
    
    # 4. 测试归一化
    print("\n" + "-" * 70)
    print("步骤2: 测试归一化")
    print("-" * 70)
    
    # 加载一个样本
    sample_file = os.path.join(demo_dir, "fnl_20180101_00_00.npy")
    original_data = np.load(sample_file)
    
    # 归一化
    normalized_data = normalizer.normalize_data(original_data)
    
    # 检查归一化效果
    print("\n归一化后各变量统计:")
    for var in ['u', '2t', 'prmsl']:  # 检查几个关键变量
        if var in normalizer.variable_specs:
            var_data = normalizer.extract_variable(normalized_data, var)
            mean_val = np.mean(var_data)
            std_val = np.std(var_data)
            print(f"  {var:>6}: 均值={mean_val:7.4f}, 标准差={std_val:7.4f}")
    
    # 5. 测试反归一化
    print("\n" + "-" * 70)
    print("步骤3: 测试反归一化")
    print("-" * 70)
    
    denormalized_data = normalizer.denormalize_data(normalized_data)
    
    # 检查精度
    mse = np.mean((original_data - denormalized_data) ** 2)
    max_diff = np.max(np.abs(original_data - denormalized_data))
    print(f"\n反归一化精度检查:")
    print(f"  均方误差 (MSE): {mse:.8f}")
    print(f"  最大绝对误差: {max_diff:.8f}")
    
    # 6. 测试边界填充
    print("\n" + "-" * 70)
    print("步骤4: 测试边界填充")
    print("-" * 70)
    
    padded_data = normalizer.apply_spherical_padding(normalized_data, pad_size=2)
    print(f"\n填充前后形状对比:")
    print(f"  原始形状: {normalized_data.shape}")
    print(f"  填充后形状: {padded_data.shape}")
    
    # 7. 测试数据加载器
    print("\n" + "-" * 70)
    print("步骤5: 测试数据加载器")
    print("-" * 70)
    
    data_loader = MixedLevelDataLoader(normalizer, demo_dir)
    generator = data_loader.create_generator(
        batch_size=2,
        shuffle=True,
        normalize=True,
        apply_padding=True,
        pad_size=1,
        infinite=False
    )
    
    try:
        batch = next(generator)
        print(f"\n批次数据信息:")
        print(f"  批次形状: {batch.shape}")
        print(f"  批次大小: {batch.shape[0]}")
        print(f"  通道数: {batch.shape[1]}")
        print(f"  纬度: {batch.shape[2]}")
        print(f"  经度: {batch.shape[3]}")
    except StopIteration:
        print("无数据可用")
    
    # 8. 保存和加载统计量
    print("\n" + "-" * 70)
    print("步骤6: 保存和加载统计量")
    print("-" * 70)
    
    stats_file = "./mixed_level_statistics.json"
    normalizer.save_statistics(stats_file)
    
    # 创建新的处理器并加载统计量
    new_normalizer = MixedLevelWeatherNormalizer()
    new_normalizer.load_statistics(stats_file)
    
    # 验证一致性
    test_data = np.load(sample_file)
    norm1 = normalizer.normalize_data(test_data)
    norm2 = new_normalizer.normalize_data(test_data)
    
    diff = np.max(np.abs(norm1 - norm2))
    print(f"\n统计量加载验证:")
    print(f"  新旧归一化最大差异: {diff:.10f}")
    print(f"  验证{'通过' if diff < 1e-10 else '失败'}")
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


# ==================== 实际使用模板 ====================

def actual_usage_example():
    """
    实际使用示例
    将以下代码集成到您的训练流程中
    """
    usage_code = '''
# ============ 实际使用代码模板 ============

# 1. 初始化处理器
normalizer = MixedLevelWeatherNormalizer()

# 2. 计算或加载统计量
import os
if os.path.exists("./weather_statistics.json"):
    print("加载已有统计量...")
    normalizer.load_statistics("./weather_statistics.json")
else:
    print("计算新统计量...")
    normalizer.compute_statistics(
        data_dir="./your_weather_data",  # 替换为您的数据目录
        pattern="*.npy"
    )
    normalizer.save_statistics("./weather_statistics.json")

# 3. 单文件处理示例
sample_data = np.load("./your_weather_data/sample.npy")

# 归一化
normalized_data = normalizer.normalize_data(sample_data)

# 应用边界填充（如需）
padded_data = normalizer.apply_spherical_padding(normalized_data, pad_size=1)

print(f"原始形状: {sample_data.shape}")
print(f"归一化形状: {normalized_data.shape}")
print(f"填充后形状: {padded_data.shape}")

# 4. 创建训练数据流
data_loader = MixedLevelDataLoader(normalizer, "./your_weather_data")

train_generator = data_loader.create_generator(
    batch_size=32,           # 根据GPU内存调整
    shuffle=True,            # 训练时打乱
    normalize=True,          # 自动归一化
    apply_padding=True,      # 如需边界处理
    pad_size=1,              # 填充大小
    infinite=True            # 无限循环用于训练
)

# 5. 在模型训练中使用
for batch_idx, batch in enumerate(train_generator):
    # batch形状: (32, 110, 183, 362) [如果pad_size=1]
    
    # 将数据送入模型
    # predictions = model(batch)
    
    if batch_idx >= 10:  # 演示只处理10个批次
        break
    
    print(f"批次 {batch_idx}: 形状={batch.shape}")
    
# 6. 推理时使用
def predict_single_file(model, file_path, normalizer):
    """处理单个文件进行预测"""
    # 加载数据
    data = np.load(file_path)
    
    # 归一化（使用训练时的统计量）
    normalized = normalizer.normalize_data(data)
    
    # 边界填充
    padded = normalizer.apply_spherical_padding(normalized, pad_size=1)
    
    # 添加批次维度
    batch_data = padded[np.newaxis, ...]  # (1, 110, 183, 362)
    
    # 预测
    prediction = model.predict(batch_data)
    
    # 移除填充并反归一化
    if prediction.shape[2:] != data.shape[1:]:
        # 移除填充
        pad_size = 1
        prediction = prediction[:, :, pad_size:-pad_size, pad_size:-pad_size]
    
    # 反归一化到物理量纲
    prediction_physical = normalizer.denormalize_data(prediction[0])
    
    return prediction_physical

# ============ 结束 ============
'''
    print(usage_code)


if __name__ == "__main__":
    import os
    # 运行完整演示
    main_demo()
    
    # print("\n\n" + "=" * 70)
    # print("实际使用代码模板:")
    # print("=" * 70)
    
    # # 显示使用模板
    # actual_usage_example()


    # # 1. 初始化处理器
    # normalizer = MixedLevelWeatherNormalizer()

    # # 2. 计算或加载统计量

    # normalizer.load_statistics("./mixed_level_statistics.json")
 
    # # 3. 单文件处理示例
    # sample_data = np.load("/home/ubuntu01/AI-MET/train_data/inn/fnl_20180101_00_00.npy")

    # # 归一化
    # normalized_data = normalizer.normalize_data(sample_data)

    #     # 6. 推理时使用
    # def predict_single_file(model, file_path):
    #     """处理单个文件进行预测"""
    #     normalizer = MixedLevelWeatherNormalizer()
    #     # 计算或加载统计量
    #     normalizer.load_statistics("./mixed_level_statistics.json")

    #     # 加载数据
    #     data = np.load(file_path)
        
    #     # 归一化（使用训练时的统计量）
    #     normalized = normalizer.normalize_data(data)
        
    #     # 边界填充
    #     padded = normalizer.apply_spherical_padding(normalized, pad_size=1)
        
    #     # 添加批次维度
    #     batch_data = padded[np.newaxis, ...]  # (1, 109, 183, 362)
        
    #     # 预测
    #     prediction = model.predict(batch_data)
        
    #     # 移除填充并反归一化
    #     if prediction.shape[2:] != data.shape[1:]:
    #         # 移除填充
    #         pad_size = 1
    #         prediction = prediction[:, :, pad_size:-pad_size, pad_size:-pad_size]
        
    #     # 反归一化到物理量纲
    #     prediction_physical = normalizer.denormalize_data(prediction[0])
        
    #     return prediction_physical
