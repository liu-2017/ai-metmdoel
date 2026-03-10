 #!/usr/bin/env python3

import torch
import numpy as np
import os
import glob
import warnings
from datetime import datetime, timedelta
import requests
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs
import xarray as xr

# 自定义模块
from normalization import MixedLevelWeatherNormalizer
from puyun import PuYun
from plt_img import *

warnings.filterwarnings("ignore")


class GFSDownloader:
    """GFS数据下载器"""
    
    @staticmethod
    def download_gfs(date_str='20251230', hour='006'):
        """
        下载GFS数据
        
        Parameters:
        -----------
        date_str : str
            日期字符串，格式：YYYYMMDD
        hour : str
            预报时效，格式：HHH，如'000','006'
        """
        url = (f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?'
               f'dir=%2Fgfs.{date_str}%2F00%2Fatmos&'
               f'file=gfs.t00z.pgrb2.1p00.f{hour}&'
               f'all_var=on&all_lev=on')
        
        # 提取文件名
        query = parse_qs(urlparse(url).query)
        filename = query.get('file', ['gfs_data.grib2'])[0]
        save_path = os.path.join(os.getcwd(), filename)
        
        print(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            
            with open(save_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=filename,
                disable=total_size == 0
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"Download completed: {save_path}")
            return save_path
            
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return None


class GRIBProcessor:
    """GRIB数据处理类"""
    
    def __init__(self, pressure_levels=None):
        """
        初始化处理器
        
        Parameters:
        -----------
        pressure_levels : list
            气压层列表，默认包含常用层次
        """
        if pressure_levels is None:
            self.pressure_levels = [
                100, 150, 200, 250, 300, 350, 400, 450, 500, 
                550, 600, 650, 700, 750, 800, 850, 900, 
                925, 950, 975, 1000
            ]
            # self.pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50][::-1]
        else:
            self.pressure_levels = pressure_levels
    
    def clear_idx_files(self, directory='.'):
        """清理目录中的.idx文件"""
        for idx_file in glob.glob(f"{directory}/*.idx"):
            os.unlink(idx_file)
            print(f"Deleted: {idx_file}")
    
    def process_grib_to_npy(self, grib_file, output_file):
        """
        将GRIB文件转换为numpy格式
        
        Parameters:
        -----------
        grib_file : str
            GRIB文件路径
        output_file : str
            输出npy文件路径
        """
        print(f"Processing {grib_file} to {output_file}")
        
        # 清理索引文件
        self.clear_idx_files(os.path.dirname(grib_file))
        
        try:
            # 读取三维变量（气压层数据）
            data_u = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'isobaricInhPa',
                    'shortName': 'u',
                    'level': self.pressure_levels
                }
            )['u'].values
            
            data_v = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'isobaricInhPa',
                    'shortName': 'v',
                    'level': self.pressure_levels
                }
            )['v'].values
            
            data_r = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'isobaricInhPa',
                    'shortName': 'r',
                    'level': self.pressure_levels
                }
            )['r'].values
            
            data_t = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'isobaricInhPa',
                    'shortName': 't',
                    'level': self.pressure_levels
                }
            )['t'].values
            
            data_z = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'isobaricInhPa',
                    'shortName': 'gh',
                    'level': self.pressure_levels
                }
            )['gh'].values
            
            # 读取二维变量（地表数据）
            data_t2m = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'heightAboveGround',
                    'shortName': '2t'
                }
            )['t2m'].values[np.newaxis, :, :]
            
            data_u10 = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'heightAboveGround',
                    'shortName': '10u'
                }
            )['u10'].values[np.newaxis, :, :]
            
            data_v10 = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={
                    'typeOfLevel': 'heightAboveGround',
                    'shortName': '10v'
                }
            )['v10'].values[np.newaxis, :, :]
            
            data_msl = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={'typeOfLevel': 'meanSea'}
            )['prmsl'].values[np.newaxis, :, :]

            data_tsk = xr.open_dataset(
                grib_file, engine='cfgrib',
                filter_by_keys={'typeOfLevel': 'surface','shortName':"t"}
            )['t'].values[np.newaxis, :, :]
            
            # 注意：GFS中的MSL气压单位是Pa，我们转换为hPa以匹配常用标准
            data_msl = data_msl 
            
            # 合并所有变量
            combined = np.concatenate([
                data_u, data_v, data_r, data_t, data_z,
                data_t2m, data_u10, data_v10, data_msl,data_tsk
            ], axis=0)[np.newaxis, :]  # 添加batch维度
            
            # 保存为numpy文件
            np.save(output_file, combined)
            print(f"Saved to {output_file}")
            
            return combined
            
        except Exception as e:
            print(f"Error processing {grib_file}: {e}")
            return None


class DataEncoder:
    """数据编码器，负责归一化处理"""
    
    def __init__(self, stats_path="./mixed_level_statistics.json"):
        """
        初始化编码器
        
        Parameters:
        -----------
        stats_path : str
            归一化统计文件路径
        """
        self.normalizer = MixedLevelWeatherNormalizer()
        self.normalizer.load_statistics(stats_path)
    
    def encode(self, npy_file_path):
        """
        对数据进行归一化编码
        
        Parameters:
        -----------
        npy_file_path : str
            numpy文件路径
            
        Returns:
        --------
        torch.Tensor
            归一化后的张量，形状为(1, 110, 181, 360)
        """
        try:
            data = np.load(npy_file_path)
            if len(data.shape) == 4:
                data = data[0]  # 去除batch维度
            normalized = self.normalizer.normalize_data(data)
            return normalized[np.newaxis, :, :, :]
        except Exception as e:
            print(f"Error encoding {npy_file_path}: {e}")
            return None
    
    def decode(self, normalized_data):
        """
        对归一化数据进行反归一化
        
        Parameters:
        -----------
        normalized_data : numpy.ndarray or torch.Tensor
            归一化数据
            
        Returns:
        --------
        numpy.ndarray
            反归一化后的数据
        """
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        if len(normalized_data.shape) == 4:
            normalized_data = normalized_data.squeeze(0)
        
        return self.normalizer.denormalize_data(normalized_data)[np.newaxis, :, :, :]


class ForecastPipeline:
    """预测管道"""
    
    def __init__(self, model_path='models/fuxi_epoch35.pth', device='cpu'):
        """
        初始化预测管道
        
        Parameters:
        -----------
        model_path : str
            模型权重文件路径
        device : str
            计算设备，'cpu'或'cuda'
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.normalizer = None
        
    def load_model(self, in_shape=(1,2, 110, 181, 360)):
        """加载预测模型"""
        print("Loading model...")
        self.model = PuYun().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully")
    
    def load_normalizer(self, stats_path="./mixed_level_statistics.json"):
        """加载归一化器"""
        self.normalizer = MixedLevelWeatherNormalizer()
        self.normalizer.load_statistics(stats_path)
        print("Normalizer loaded successfully")
    
    def create_filenames(self, start_time, forecast_hour):
        """
        生成文件名
        
        Parameters:
        -----------
        start_time : str
            起始时间，格式：YYYYMMDDHH
        forecast_hour : int
            预报时效（小时）
            
        Returns:
        --------
        tuple
            (npy文件名, nc文件名)
        """
        start_dt = datetime.strptime(start_time, "%Y%m%d%H")
        forecast_dt = start_dt + timedelta(hours=forecast_hour)
        time_str = forecast_dt.strftime("%Y%m%d_%H")
        
        npy_file = f"fnl_{time_str}_00.npy"
        nc_file = f"fnl_{time_str}_00.nc"
        
        return npy_file, nc_file
    
    def autoregressive_forecast(self, initial_input, forecast_steps=10):
        """
        自回归预测
        
        Parameters:
        -----------
        initial_input : torch.Tensor
            初始输入数据，形状为(1, 220, 181, 360)
        forecast_steps : int
            预测步数
            
        Returns:
        --------
        list
            预测结果列表，每个元素为反归一化的数据
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        predictions = []
        current_input = initial_input.to(self.device)
        
        with torch.no_grad():
            for step in range(forecast_steps):
                print(f"Forecasting step {step + 1}/{forecast_steps}...")
                
                # 预测下一个时间步
                normalized_pred = self.model(current_input)  
                
                # 反归一化
                if self.normalizer:
                    denormalized_pred = self.normalizer.denormalize_data(
                        normalized_pred.squeeze(0).cpu().numpy()
                    )[np.newaxis, :, :, :]
                else:
                    denormalized_pred = normalized_pred.cpu().numpy()
                
                predictions.append(denormalized_pred)
                
                # 准备下一个输入：滚动窗口
                # 当前输入形状为(1, 220, 181, 360)，前110通道是t-1，后110通道是t
                # 我们使用t和预测的t+1作为下一个输入

                current_t = current_input[:,1,:, :, :][:,np.newaxis,:,:]  # 形状: (1, 110, 181, 360)
                normalized_pred = normalized_pred[:,np.newaxis,:,:]

                # 将预测结果（归一化的）作为t+1
                next_input = torch.cat([current_t, normalized_pred], dim=1)
                
                # 更新当前输入
                current_input = next_input
        
        return predictions
    
    def save_netcdf(self, data, save_path):
        """
        保存为NetCDF格式
        
        Parameters:
        -----------
        data : numpy.ndarray
            数据，形状为(1, 110, 181, 360)
        save_path : str
            保存路径
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # 去除batch维度
        if len(data.shape) == 4:
            data = data.squeeze(0)  # (110, 181, 360)
        
        # 定义维度
        pressure_levels = [100, 150, 200, 250, 300, 350, 400, 450, 500, 
                          550, 600, 650, 700, 750, 800, 850, 900, 
                          925, 950, 975, 1000]
        # pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50][::-1]
        lat = np.linspace(90, -90, 181)
        lon = np.linspace(0, 359, 360)
        
        # 创建数据变量
        data_vars = {}
        idx = 0
        
        # 三维变量（气压层）
        var_names = ['u', 'v', 'r', 't', 'gh']
        for var_name in var_names:
            var_data = data[idx:idx + 21, :, :]  # (13, 181, 360)
            units = self.get_units(var_name)
            
            data_vars[var_name] = xr.DataArray(
                var_data,
                dims=['level', 'lat', 'lon'],
                coords={'level': pressure_levels, 'lat': lat, 'lon': lon},
                attrs={'units': units}
            )
            idx += 21
        
        # 二维变量（地表）
        surface_vars = ['t2m', 'u10', 'v10', 'msl','tsk']
        for var_name in surface_vars:
            var_data = data[idx:idx + 1, :, :].squeeze(0)  # (181, 360)
            units = self.get_units(var_name)
            
            data_vars[var_name] = xr.DataArray(
                var_data,
                dims=['lat', 'lon'],
                coords={'lat': lat, 'lon': lon},
                attrs={'units': units}
            )
            idx += 1
        
        # 创建数据集
        ds = xr.Dataset(
            data_vars,
            attrs={
                'title': 'Fengwu Weather Prediction',
                'source': 'Fengwu Deep Learning Model',
                'history': f'Created on {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC',
                'description': 'Weather forecast generated by Fengwu model'
            }
        )
        
        # 保存
        ds.to_netcdf(save_path)
        print(f"Saved NetCDF: {save_path}")
    
    @staticmethod
    def get_units(var_name):
        """获取变量单位"""
        units_map = {
            'u': 'm s-1',
            'v': 'm s-1',
            'r': '%',
            't': 'K',
            'gh': 'dam',
            't2m': 'K',
            'u10': 'm s-1',
            'v10': 'm s-1',
            'msl': 'Pa',
            'tsk': 'k',
        }
        return units_map.get(var_name, 'unknown')





if __name__ == '__main__':
    # 配置参数
    start_date = '20260309'
    start_time = f'{start_date}00'
    device = 'cpu'
    
    # 初始化组件
    downloader = GFSDownloader()
    processor = GRIBProcessor()
    encoder = DataEncoder()
    pipeline = ForecastPipeline(model_path='models/model.pth',device=device)
    
    # # # # # # # # 步骤1: 下载初始数据2
    # # # # # # # # # print("Step 1: Downloading initial GFS data...")
    downloader.download_gfs(date_str=start_date, hour='000')
    downloader.download_gfs(date_str=start_date, hour='006')

    # 步骤2: 处理GRIB数据为numpy格式
    ###########print("\nStep 2: Processing GRIB files...")
    processor.process_grib_to_npy('gfs.t00z.pgrb2.1p00.f000', f'fnl_{start_date}_00_00.npy')
    processor.process_grib_to_npy('gfs.t00z.pgrb2.1p00.f006', f'fnl_{start_date}_06_00.npy')
    
    # 步骤3: 归一化初始数据
    print("\nStep 3: Normalizing initial data...")
    data_00 = encoder.encode(f'fnl_{start_date}_00_00.npy')
    data_06 = encoder.encode(f'fnl_{start_date}_06_00.npy')
    
    # 创建模型输入（两个时间步拼接）
    # initial_input = np.concatenate([data_00, data_06], axis=1)  # (1, 218, 181, 360)
    initial_input = np.stack([data_00, data_06],axis=1)

    print(initial_input.shape)
    initial_tensor = torch.from_numpy(initial_input).float()
    
    # 步骤4: 加载模型和归一化器
    print("\nStep 4: Loading model and normalizer...")
    pipeline.load_model()
    pipeline.load_normalizer()
    
    # 步骤5: 执行自回归预测
    print("\nStep 5: Starting autoregressive forecast...")
    predictions = pipeline.autoregressive_forecast(initial_tensor, forecast_steps=60)
    
    # 步骤6: 保存结果
    print("\nStep 6: Saving results...")
    for i, pred in enumerate(predictions):
        forecast_hour = 6 * (i + 2)  # i=0对应12:00, i=1对应18:00, 依此类推
        
        # 生成文件名
        npy_file, nc_file = pipeline.create_filenames(start_time, forecast_hour)
        
        # 保存为numpy文件
        np.save(npy_file, pred)
        print(f"Saved numpy file: {npy_file}")
        
        # 保存为NetCDF文件
        # pipeline.save_netcdf(pred, nc_file)
        
        # 生成可视化
        # plot_500GHT(file_path=npy_file,levs=500)
        
    
    print("\nForecast completed successfully!")