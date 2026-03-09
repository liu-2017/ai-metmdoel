import numpy as np
from torch.utils.data import DataLoader,Dataset
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime, timedelta
from  normalization import *

class AS_Data_2(Dataset):
    def __init__(self, base_path):
        """
        Args:
            root_dir (string): 目录路径，其中包含子目录，每个子目录代表一个类别。
            transform (callable, optional): 一个可选的转换函数，用于预处理图像。

        """
        self.base_path  = base_path
        self.input_dir  = f'{self.base_path}/inn'
        self.target_dir = f'{self.base_path}/out'
        self.conc_files = []

        for conc_file in sorted(os.listdir(self.target_dir))[10:-10]:  #向提一天
            fff = os.path.join(self.base_path, conc_file)
            self.conc_files.append(fff)

    def __len__(self):
        return len(self.conc_files)
    

    def __getitem__(self, idx):
        met_file = self.conc_files[idx]
        idd = met_file.split('/')[-1].split('.')[0]


        ss = idd[4:-3].replace('_','')
        t1 = datetime.strptime(ss,'%Y%m%d%H')
        t2 = t1+timedelta(hours=-6)
        t3 = datetime.strftime(t2,'%Y%m%d%H')
        tt = f'fnl_{t3[:-2]}_{t3[-2:]}_00'

        # print(tt)




        # 1. 初始化处理器
        normalizer = MixedLevelWeatherNormalizer()
        normalizer.load_statistics("./mixed_level_statistics.json")
    
        met2d_1 = np.load(f'{self.base_path}/inn/{tt}.npy')
        met2d_2 = np.load(f'{self.base_path}/inn/{idd}.npy')


        met2d_11 = normalizer.normalize_data(met2d_1)
        met2d_22 = normalizer.normalize_data(met2d_2)
        met2d_data1 = np.concatenate([met2d_11,met2d_22],axis=0)




        met2d_data2 = np.load(f'{self.base_path}/out/{idd}.npy')
        met2d_data2 = normalizer.normalize_data(met2d_data2)


        return met2d_data1,met2d_data2



if __name__ == '__main__':
    base_path   = '/home/ubuntu01/AI-MET/train_data'
    dataset = AS_Data_2(base_path = base_path)

    from torch.utils.data import DataLoader

    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # # 遍历数据集中的所有项
    for out_data, con_data in data_loader:
        # 这里可以放置处理每个批次数据的代码
        B,H,row,col = out_data.shape
        print(out_data.shape)
        print(con_data.shape)
        print(B,H,row,col)
        B,H,row,col = con_data.shape
        print(B,H,row,col)
        break


