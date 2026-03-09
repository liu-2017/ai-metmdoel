#!/usr/bin/env python
# coding: utf-8

import faulthandler
import traceback
import sys
import gc
import signal
faulthandler.enable()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from datetime import datetime

from loss import LatitudeWeightedL1Loss
criterion = LatitudeWeightedL1Loss(num_latitudes=181)

# 自定义数据集和模型导入
from RSMdataset_3d_6h import AS_Data_2
from aigamfs import *

# 设置异常处理函数
def handle_exception(exc_type, exc_value, exc_traceback):
    """处理未捕获的异常"""
    print("\n" + "="*50)
    print("UNHANDLED EXCEPTION OCCURRED!")
    print("="*50)
    print(f"Exception type: {exc_type}")
    print(f"Exception value: {exc_value}")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    
    # 尝试保存当前状态
    try:
        print("\nAttempting to save emergency checkpoint...")
        emergency_checkpoint = {
            'emergency_save': True,
            'exception_type': str(exc_type),
            'exception_value': str(exc_value)
        }
        torch.save(emergency_checkpoint, 'models/emergency_checkpoint.pth')
        print("Emergency checkpoint saved.")
    except:
        print("Failed to save emergency checkpoint.")
    
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# 设置全局异常处理
sys.excepthook = handle_exception

# 信号处理器（处理Ctrl+C等）
def signal_handler(sig, frame):
    print("\n" + "="*50)
    print(f"Signal {sig} received. Saving checkpoint before exit...")
    save_checkpoint()
    print("Checkpoint saved. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 创建保存目录
os.makedirs('samples', exist_ok=True)
os.makedirs('models',  exist_ok=True)
os.makedirs('logs', exist_ok=True)  # 用于保存训练日志

# 配置参数
base_path = '/home/ubuntu01/AI-MET/train_data'
batch_size = 8
epochs = 251
checkpoint_path = 'models/checkpoint_latest.pth'  # 自动保存/加载的检查点

# 设备
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==================== 日志函数 ====================
def log_message(message, level="INFO"):
    """记录日志信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    
    # 写入日志文件
    with open('logs/training.log', 'a') as f:
        f.write(log_entry + '\n')

# ==================== 保存检查点函数 ====================
def save_checkpoint(epoch=None, model=None, optimizer=None, scheduler=None, 
                   train_losses=None, val_losses=None, learning_rates=None,
                   in_shape=None, force_save=False):
    """保存检查点"""
    try:
        if epoch is None:
            epoch = 0
            
        checkpoint = {
            'epoch': epoch,
            'train_losses': train_losses if train_losses is not None else [],
            'val_losses': val_losses if val_losses is not None else [],
            'learning_rates': learning_rates if learning_rates is not None else [],
            'in_shape': in_shape,
        }
        
        if model is not None:
            checkpoint['model_state_dict'] = model.state_dict()
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, checkpoint_path)
        
        # 定期保存额外检查点
        if force_save or (epoch % 10 == 0 and epoch > 0):
            backup_path = f'models/checkpoint_epoch_{epoch:04d}.pth'
            torch.save(checkpoint, backup_path)
            log_message(f"Saved backup checkpoint to {backup_path}")
            
        log_message(f"Checkpoint saved at epoch {epoch}")
        return True
    except Exception as e:
        log_message(f"Error saving checkpoint: {e}", "ERROR")
        return False

# ==================== 安全数据加载 ====================
def safe_data_loading():
    """安全地加载数据，避免崩溃"""
    try:
        log_message("Loading dataset...")
        dataset = AS_Data_2(base_path=base_path)
        
        # 检查数据集大小
        dataset_len = len(dataset)
        log_message(f"Dataset size: {dataset_len}")
        
        if dataset_len == 0:
            raise ValueError("Dataset is empty!")
        
        # 计算分割大小
        train_length = int(0.80 * dataset_len)
        val_length   = int(0.19 * dataset_len)
        test_length  = dataset_len - train_length - val_length
        
        log_message(f"Train/Val/Test split: {train_length}/{val_length}/{test_length}")
        
        # 分割数据集
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_length, val_length, test_length]
        )
        
        # 创建数据加载器（减少workers以避免内存问题）
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0, pin_memory=True)
        
        # 检查数据形状
        for input_data, conc_data in train_loader:
            if input_data.dim() != 4 or conc_data.dim() != 4:
                raise ValueError(f"Invalid data dimensions: input={input_data.shape}, label={conc_data.shape}")
            break
            
        return train_loader, val_loader, test_loader, dataset
        
    except Exception as e:
        log_message(f"Error in data loading: {e}", "ERROR")
        traceback.print_exc()
        raise

# ==================== 训练函数 ====================
def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """执行一个训练epoch"""
    model.train()
    train_loss = 0.0
    n_batches = 0
    
    try:
        with tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Train]', 
                 leave=False, dynamic_ncols=True) as pbar:
            for batch_idx, (input, label) in enumerate(pbar):
                try:
                    # 清理GPU缓存
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # 移动到设备
                    input = input.to(device, non_blocking=True).float()
                    label = label.to(device, non_blocking=True).float()
                    
                    # 前向传播
                    optimizer.zero_grad()
                    outputs = model(input)
                    
                    # 计算损失
                    loss = criterion(outputs, label)

                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # 记录损失
                    train_loss += loss.item()
                    n_batches += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{train_loss/n_batches:.4f}'
                    })
                    
                    # 定期清理
                    if batch_idx % 50 == 0:
                        del input, label, outputs, loss
                        gc.collect()
                        
                except torch.cuda.OutOfMemoryError as e:
                    log_message(f"CUDA OOM at batch {batch_idx}: {e}", "ERROR")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    log_message(f"Error in batch {batch_idx}: {e}", "ERROR")
                    continue
                    
    except Exception as e:
        log_message(f"Error in train_epoch: {e}", "ERROR")
        traceback.print_exc()
    
    avg_train_loss = train_loss / max(n_batches, 1)
    return avg_train_loss

# ==================== 验证函数 ====================
def validate_epoch(model, loader, criterion, device, epoch, total_epochs):
    """执行验证"""
    model.eval()
    val_loss = 0.0
    n_batches = 0
    
    try:
        with torch.no_grad():
            with tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Val]', 
                     leave=False, dynamic_ncols=True) as pbar:
                for batch_idx, (input, label) in enumerate(pbar):
                    try:
                        # 移动到设备
                        input = input.to(device, non_blocking=True).float()
                        label = label.to(device, non_blocking=True).float()
                        
                        # 前向传播
                        outputs = model(input)
                        
                        # 计算损失
                        loss = criterion(outputs, label)
                        
                        # 记录损失
                        val_loss += loss.item()
                        n_batches += 1
                        
                        # 更新进度条
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{val_loss/n_batches:.4f}'
                        })
                        
                    except Exception as e:
                        log_message(f"Error in validation batch {batch_idx}: {e}", "ERROR")
                        continue
                        
    except Exception as e:
        log_message(f"Error in validate_epoch: {e}", "ERROR")
        traceback.print_exc()
    
    avg_val_loss = val_loss / max(n_batches, 1)
    return avg_val_loss

# ==================== 可视化函数 ====================
def visualize_results(model, test_cond, test_data, device, epoch):
    num = 4*13+6
    """生成可视化结果"""
    try:
        model.eval()
        with torch.no_grad():
            # 只使用前4个样本
            n_samples = min(4, len(test_cond))
            test_cond_vis = test_cond[:n_samples].to(device, non_blocking=True).float()
            test_data_vis = test_data[:n_samples].to(device, non_blocking=True).float()
            
            generated_images = model(test_cond_vis).cpu()
            
            test_cond_np = test_cond_vis.cpu().numpy()
            test_data_np = test_data_vis.cpu().numpy()
            generated_np = generated_images.numpy()
            
            fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 6))
            
            for i in range(n_samples):
                # 原始图像
                im = axes[0, i].imshow(test_data_np[i, num, :, :], cmap='gray')
                axes[0, i].axis('off')
                axes[0, i].set_title(f"Original {i}")
                fig.colorbar(im, ax=axes[0, i])
                
                # 生成图像
                im = axes[1, i].imshow(generated_np[i, num, :, :], cmap='gray')
                axes[1, i].axis('off')
                axes[1, i].set_title(f"Generated {i}")
                fig.colorbar(im, ax=axes[1, i])
                
                # 差异图
                diff = generated_np[i, num ,:, :] - test_data_np[i, num, :, :]
                im = axes[2, i].imshow(diff, cmap='coolwarm')
                axes[2, i].axis('off')
                axes[2, i].set_title(f"Difference {i}")
                fig.colorbar(im, ax=axes[2, i])
            
            plt.suptitle(f'Epoch {epoch}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'samples/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
    except Exception as e:
        log_message(f"Error in visualization: {e}", "ERROR")

# ==================== 绘制损失曲线 ====================
def plot_losses(train_losses, val_losses, learning_rates, epoch):
    """绘制损失和 learning rate 曲线"""
    try:
        plt.figure(figsize=(10, 8))
        
        # 损失曲线
        plt.subplot(3, 1, 1)
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Training & Validation Loss (Epoch {epoch})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Learning rate 曲线
        plt.subplot(3, 1, 2)
        plt.plot(learning_rates, label='Learning Rate', color='green', alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate (log scale)')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 损失差值
        plt.subplot(3, 1, 3)
        if len(train_losses) > 1:
            train_diff = np.diff(train_losses)
            val_diff = np.diff(val_losses)
            x = range(1, len(train_losses))
            plt.plot(x, train_diff, label='Train Loss Diff', color='blue', alpha=0.5)
            plt.plot(x, val_diff, label='Val Loss Diff', color='red', alpha=0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss Difference')
            plt.title('Loss Changes Between Epochs')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'logs/loss_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        log_message(f"Error plotting losses: {e}", "ERROR")

# ==================== 主训练函数 ====================
def main():
    """主训练函数"""
    log_message("Starting training process...")
    
    # 初始化变量
    start_epoch = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    
    model = None
    optimizer = None
    scheduler = None
    in_shape = None
    
    # 尝试加载检查点
    try:
        if os.path.exists(checkpoint_path):
            log_message(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 重建模型
            if 'in_shape' in checkpoint:
                in_shape = checkpoint['in_shape']
                H = in_shape[1]
                model = AI_GAMFS().to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                log_message(f"Model loaded with input shape: {in_shape}")
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
            
            if 'val_losses' in checkpoint:
                val_losses = checkpoint['val_losses']
            
            if 'learning_rates' in checkpoint:
                learning_rates = checkpoint['learning_rates']
            
            log_message(f"Resumed from epoch {start_epoch}")
            
    except Exception as e:
        log_message(f"Error loading checkpoint: {e}", "ERROR")
        log_message("Starting from scratch...")
    
    # 加载数据
    try:
        train_loader, val_loader, test_loader, dataset = safe_data_loading()
        
        # 获取测试样本
        test_cond, test_data = next(iter(test_loader))
        
    except Exception as e:
        log_message(f"Failed to load data: {e}", "CRITICAL")
        return
    
    # 如果模型未加载，创建新模型
    if model is None:
        try:
            # 获取输入形状
            for input_data, conc_data in train_loader:
                _, H1, row, col = input_data.shape
                _, H2, row, col = conc_data.shape
                in_shape = (1, H1, row, col)
                break
            
            log_message(f"Creating new model with input shape: {in_shape}")
            model = AI_GAMFS().to(device)
            
        except Exception as e:
            log_message(f"Error creating model: {e}", "CRITICAL")
            return
    
    # 初始化优化器和损失函数
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    if scheduler is None:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    
    log_message(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log_message(f"Training on {device}")
    log_message(f"Input shape: {in_shape}, Batch size: {batch_size}")
    
    # 训练循环
    for epoch in range(start_epoch, epochs):
        log_message(f"\n{'='*50}")
        log_message(f"Starting Epoch {epoch}/{epochs-1}")
        log_message(f"{'='*50}")
        
        try:
            # 训练
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
            train_losses.append(train_loss)
            
            # 验证
            val_loss = validate_epoch(model, val_loader, criterion, device, epoch, epochs)
            val_losses.append(val_loss)
            
            # 更新学习率
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 可视化
            visualize_results(model, test_cond, test_data, device, epoch)
            
            # 绘制损失曲线
            plot_losses(train_losses, val_losses, learning_rates, epoch)
            
            # 打印摘要
            log_message(f"Epoch {epoch} Summary:")
            log_message(f"  Training Loss: {train_loss:.6f}")
            log_message(f"  Validation Loss: {val_loss:.6f}")
            log_message(f"  Learning Rate: {current_lr:.2e}")
            
            # 保存检查点
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_losses=train_losses,
                val_losses=val_losses,
                learning_rates=learning_rates,
                in_shape=in_shape
            )
            
            # 定期保存模型
            if epoch % 5 == 0 or epoch == epochs - 1:
                model_path = f'models/fuxi_epoch{epoch}.pth'
                torch.save(model.state_dict(), model_path)
                log_message(f"Model saved to {model_path}")
            
            # 清理GPU缓存
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except KeyboardInterrupt:
            log_message("Training interrupted by user")
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_losses=train_losses,
                val_losses=val_losses,
                learning_rates=learning_rates,
                in_shape=in_shape,
                force_save=True
            )
            break
        except Exception as e:
            log_message(f"Error in epoch {epoch}: {e}", "ERROR")
            traceback.print_exc()
            
            # 尝试保存紧急检查点
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_losses=train_losses,
                val_losses=val_losses,
                learning_rates=learning_rates,
                in_shape=in_shape,
                force_save=True
            )
            
            # 等待一段时间后继续
            import time
            time.sleep(2)
            continue
    
    log_message("\n" + "="*50)
    log_message("Training completed!")
    log_message(f"Final training loss: {train_losses[-1]:.6f}")
    log_message(f"Final validation loss: {val_losses[-1]:.6f}")
    log_message("="*50)

# ==================== 运行主函数 ====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_message(f"Fatal error in main: {e}", "CRITICAL")
        traceback.print_exc()
        sys.exit(1)