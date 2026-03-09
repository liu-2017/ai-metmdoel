import xarray as xr
import numpy as np
import os
import glob
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore")

os.makedirs('/home/ubuntu01/AI-MET/train_data/inn',exist_ok=True)
os.makedirs('/home/ubuntu01/AI-MET/train_data/out',exist_ok=True)
def process_single_file(file_info):
    """处理单个文件的函数"""
    file, file_path, year, inn_levels = file_info
    
    file_1 = os.path.join(file_path, file)
    out_file = file.split('.')[0]
    
    try:
        # 读取数据
        data_u = xr.open_dataset(file_1, engine='cfgrib', 
                                filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'u', 'level': inn_levels})['u'].values
        data_v = xr.open_dataset(file_1, engine='cfgrib', 
                                filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'v', 'level': inn_levels})['v'].values
        data_r = xr.open_dataset(file_1, engine='cfgrib', 
                                filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'r', 'level': inn_levels})['r'].values
        data_t = xr.open_dataset(file_1, engine='cfgrib', 
                                filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 't', 'level': inn_levels})['t'].values
        data_z = xr.open_dataset(file_1, engine='cfgrib', 
                                filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh', 'level': inn_levels})['gh'].values
        
        data_t2m = xr.open_dataset(file_1, engine='cfgrib', 
                                  filter_by_keys={'typeOfLevel': 'heightAboveGround', 'shortName': '2t'})['t2m'].values[np.newaxis, :, :]
        data_u10 = xr.open_dataset(file_1, engine='cfgrib', 
                                  filter_by_keys={'typeOfLevel': 'heightAboveGround', 'shortName': '10u'})['u10'].values[np.newaxis, :, :]
        data_v10 = xr.open_dataset(file_1, engine='cfgrib', 
                                  filter_by_keys={'typeOfLevel': 'heightAboveGround', 'shortName': '10v'})['v10'].values[np.newaxis, :, :]
        data_msl = xr.open_dataset(file_1, engine='cfgrib', 
                                  filter_by_keys={'typeOfLevel': 'meanSea'})['prmsl'].values[np.newaxis, :, :]
        
        # 合并数据
        combined = np.concatenate([data_u, data_v, data_r, data_t, data_z, 
                                   data_t2m, data_u10, data_v10, data_msl], axis=0)
        

        
        # 保存npy文件
        output_path = f'/home/ubuntu01/AI-MET/train_data/inn/{out_file}.npy'
        np.save(output_path, combined)
        
        # 创建符号链接
        idd = file.split('/')[-1].split('.')[0]
        ss = idd[4:-3].replace('_', '')
        t1 = datetime.strptime(ss, '%Y%m%d%H')
        t2 = t1 + timedelta(hours=-6)
        t3 = datetime.strftime(t2, '%Y%m%d%H')
        tt = f'fnl_{t3[:-2]}_{t3[-2:]}_00'
        
        dst = f'/home/ubuntu01/AI-MET/train_data/out/{tt}.npy'
        try:
            os.remove(dst)
        except:
            pass
        os.symlink(output_path, dst)
        
        print(f"Processed: {file}, Shape: {combined.shape}, Size: {combined.nbytes/1024/1024:.2f} MB")
        return True, file, combined.shape
        
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return False, file, str(e)

def creat_npy_parallel(year='2019', max_workers=None,file_path = f'/media/ubuntu01/Elements/FNL/fnl'):
    """并行处理某个年份的所有文件"""
    
    # 创建输出目录
    os.makedirs('/home/ubuntu01/AI-MET/train_data/inn', exist_ok=True)
    os.makedirs('/home/ubuntu01/AI-MET/train_data/out', exist_ok=True)
    
    # file_path = f'/home/ubuntu01/sdb/天气分型减排/FNL/fnl{year}'
    
    
    # 清理.idx文件
    for ff in glob.glob(f"{file_path}/*.idx"):
        os.unlink(ff)
        print(f"Deleted: {ff}")
    
    # 获取所有grib2文件
    files = sorted([i for i in os.listdir(file_path) if i.endswith('.grib2')])
    
    # # 定义气压层
    inn_levels = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                  600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
    # inn_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    
    # 准备文件信息
    file_infos = [(file, file_path, year, inn_levels) for file in files]
    
    # 设置进程数（默认为CPU核心数）
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    print(f"Processing {len(files)} files with {max_workers} workers...")
    
    # 使用进程池并行处理
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_single_file, file_info): file_info 
                          for file_info in file_infos}
        
        # 处理完成的任务
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            try:
                success, filename, result = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"Failed to process {filename}: {result}")
            except Exception as e:
                failed += 1
                print(f"Exception for {file_info[0]}: {e}")
    
    print(f"\nProcessing completed!")
    print(f"Successful: {successful}, Failed: {failed}, Total: {len(files)}")


if __name__ == '__main__':
    # 方法1：处理单个年份
    # creat_npy_parallel(year='2019')
    creat_npy_parallel(year='2024',file_path = f'/home/ubuntu01/sdb/天气分型减排/FNL/fnl2024')
    creat_npy_parallel(year='2018',file_path = f'/media/ubuntu01/Elements/FNL/fnl2018')
    creat_npy_parallel(year='2017',file_path = f'/media/ubuntu01/Elements/FNL/fnl2017')
    creat_npy_parallel(year='2016',file_path = f'/media/ubuntu01/Elements/FNL/fnl2016')
    creat_npy_parallel(year='2019',file_path = f'/home/ubuntu01/sdb/天气分型减排/FNL/fnl2019')
    creat_npy_parallel(year='2020',file_path = f'/home/ubuntu01/sdb/天气分型减排/FNL/fnl2020')
    creat_npy_parallel(year='2021',file_path = f'/home/ubuntu01/sdb/天气分型减排/FNL/fnl2021')
    creat_npy_parallel(year='2022',file_path = f'/home/ubuntu01/sdb/天气分型减排/FNL/fnl2022')
    creat_npy_parallel(year='2023',file_path = f'/home/ubuntu01/sdb/天气分型减排/FNL/fnl2023')

 
    
