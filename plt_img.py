#!/usr/bin/env python3

import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from cartopy.io.shapereader import Reader
from matplotlib.gridspec import GridSpec

from datetime import timedelta,datetime
shape_path = ('/home/ubuntu01/AI-MET/china_shp/china.shp')
# shape_path = (r'D:\map-shp\china_shp\china.shp')
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


levels = [100, 150, 200, 250, 300, 350, 400, 450, 500, 
          550, 600, 650, 700, 750, 800, 850, 900, 
          925, 950, 975, 1000
          ]
levels = levels[::-1]
def plot_500GHT(file_path,levs=500,dpi=200,hgt_lev=None,labels=None,plt_path=None):
    '''注：
    file1: xarray数据集,使用get_xrdata(npy_path)函数获取;;
    levs: 指定高度层,默认500hPa;;
    hgt_lev: 位势高比例尺,默认None(建议数值为10**x),可根据打印的hgt-lev{levs}:mean,min,max调整,期望[100,1100];;
    dpi: 图片分辨率,默认200;;
    labels: 图片标题,默认为None,显示为{levs}hPa环流图,若有则仅显示labels;;
    plt_path: 图片保存路径,默认为None,保存至代码当前目录;;
    '''
    def get_xrdata(file_path=file_path):
        file1 = np.load(file_path)
        lats = np.linspace(90, -90, 181)
        lons = np.linspace(0, 359, 360)
        xr_data = xr.Dataset(coords={'lev': levels,'lat': lats,'lon': lons})
        for i,var in enumerate(['u','v','r','t','z']):
            xr_data[var] = xr.DataArray(
                file1[0,i*len(levels):(i+1)*len(levels), :, :],
                dims=['lev','lat', 'lon'],
                coords={'lev': levels,'lat': lats, 'lon': lons},)
        for i,var in enumerate(['t2','u10','v10','msl'],start=len(levels)*5):
            xr_data[var] = xr.DataArray(
                file1[0,i, :, :],
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons},)
        return xr_data

    lons_min, lons_max, lats_min, lats_max = 70, 140, 0, 60
    lon,lat = np.linspace(lons_min, lons_max, (lons_max-lons_min+1)), np.linspace(lats_max, lats_min, (lats_max-lats_min+1))
    file1 = get_xrdata(file_path=file_path)
    # 主程序
    file1_lev = file1.sel(lev=levs, method='nearest')
    u = file1_lev['u'].sel(lon=slice(lons_min, lons_max), lat=slice(lats_max, lats_min)).values
    v = file1_lev['v'].sel(lon=slice(lons_min, lons_max), lat=slice(lats_max, lats_min)).values
    hgt = file1_lev['z'].sel(lon=slice(lons_min, lons_max), lat=slice(lats_max, lats_min)).values

    print('#'*5,'plot_500GHT 高空环流图 绘制提示','#'*5)
    print(f'hgt-lev{levs}:mean,min,max:',hgt.mean(),hgt.min(),hgt.max())
    if hgt_lev is None:
        for i in range(-2, 5):
            hgt_lev = 10**i
            if hgt.mean()*hgt_lev >= 100 and hgt.mean()*hgt_lev <= 1100:
                break
    print(f'当前位势高比例尺:{hgt_lev},期望[100,1100],当前hgt_mean*{hgt_lev}={hgt.mean()*hgt_lev}')
    print('#'*25)

    wind_speed = np.sqrt(u**2 + v**2)
    # 设置颜色映射和色条
    cmap = ListedColormap(['#ffffff', '#E3F4F7', '#ABDEE7', '#74C8D7', '#67C9B2', '#63CF89', '#6AD66C', '#A3E684', '#DBF69B','#F6FDA6'])
    levels_ws = np.arange(16, 34, 2)  # 16,18,...,32
    levels_ws0 = np.arange(14, 36, 2)
    norm_ws = BoundaryNorm(levels_ws0, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(10, 12))
    proj = ccrs.PlateCarree()
    ax = plt.axes([0.06, 0.05, 0.8, 0.8], projection=proj)
    ax.set_extent([71, 136.5, 14.5, 59.5], crs=proj)
    china = cfeature.ShapelyFeature(Reader(shape_path).geometries(), proj, edgecolor='k', facecolor='none')
    ax.add_feature(china, lw=1, alpha=0.6, zorder=2)
    grid_lines = ax.gridlines(crs=proj, color='k', linestyle='--', alpha=0.3, 
							xlocs=np.arange(0, 180 + 1, 10), ylocs=np.arange(0, 80 + 1, 10),
							draw_labels=True, x_inline=False, y_inline=False)  #
    grid_lines.rotate_labels =0
    grid_lines.top_labels = False
    grid_lines.right_labels = False

    # 绘制风速填充图
    cf = ax.contourf(lon, lat, wind_speed, levels=levels_ws, cmap=cmap, norm=norm_ws, extend='both', transform=proj)
    cbar_ax = fig.add_axes([0.88, 0.21, 0.025, 0.48])
    cb = fig.colorbar(cf, cax = cbar_ax,orientation='vertical',label='Wind Speed (m/s)')

    # 风矢标（barbs）
    # 控制密度（避免太密），可下采样
    skip = 2 # 每2个格点取一个
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    ws_sub = np.sqrt(u_sub**2 + v_sub**2)
    lat_sub = lat[::skip]
    lon_sub = lon[::skip]
    lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)
    mask_black = ws_sub < 19
    # 展平所有数组以保持一致的维度
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    u_flat = u_sub.flatten()
    v_flat = v_sub.flatten()
    mask_flat = mask_black.flatten()
    # 分别绘制 <20 和 >=20 的风矢
    barb_kwargs = dict(
        transform=proj,
        length=5,
        linewidth=0.7,
        barb_increments=dict(half=2, full=4, flag=20))
    ax.barbs(
        lon_flat[mask_flat], lat_flat[mask_flat],
        u_flat[mask_flat], v_flat[mask_flat],
        color='black', **barb_kwargs)
    # 红色风矢（>=20 m/s）
    mask_red = ~mask_flat
    ax.barbs(
        lon_flat[mask_red], lat_flat[mask_red],
        u_flat[mask_red], v_flat[mask_red],
        color='red', **barb_kwargs)
    
    # 绘制500hPa位势高度等值线  使用 xarray 插值
    da_hgt = xr.DataArray(hgt, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})
    da_hgt_fine = da_hgt.interp(
        lat=np.arange(lat.min(), lat.max() + 0.04,  0.04),
        lon=np.arange(lon.min(), lon.max() +  0.04,  0.04),
        method='cubic')
    hgt_fine = da_hgt_fine.values
    lat_fine = da_hgt_fine.lat.values
    lon_fine = da_hgt_fine.lon.values

    levels_hgt = np.arange(0, 1200, 4)  # 400,404,...,700
    cs = ax.contour(lon_fine, lat_fine, hgt_fine*hgt_lev, levels=levels_hgt, colors='black', linewidths=1, transform=proj)
    clabels = ax.clabel(cs,inline=True,fontsize=8,fmt='%4.0f',
                        colors='black',use_clabeltext=True)  # 启用文本对象，便于后续样式设置
    for txt in clabels:   # 为每个标签添加白色背景
        txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.8, alpha=0.8))

    # 主标题
    if labels is None:
        titiles=f'{levs}hPa u-v-gh'
    else:
        titiles=labels
    ax.set_title(titiles, fontsize=18, pad=5)
    ax.set_title(file_path, loc='right', fontsize=12, pad=5)

    # plt.show()
    # base_name = os.path.splitext(os.path.basename(file1))[0]
    time_name = datetime.now().strftime('%Y%j_%H_%M_%S')
    img_filename = f"{file_path}_{levs}hPa_uvz_t{time_name}.png"
    if plt_path is None:
        save_path = os.path.join(os.path.dirname(__file__), img_filename)
    else:
        save_path = os.path.join(plt_path, img_filename)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return

def plot_SKew_T(file_list, lonlat=[114,38], dpi=200, labels=None, plt_path=None):
    '''
    :param file_list: 文件列表,如['fnl_20181001_00_00.npy','fnl_20181001_06_00.npy'];;
    :param lonlat: 经纬度,默认[114,38]石家庄;;
    :param dpi: 分辨率,默认200;;
    :param labels: 图片标题,默认为None,显示为shijiazhuang:SKew-T,若有则显示{labels}:SKew-T;;
    :param plt_path: 图片保存路径,默认为None,保存至代码当前目录;;
    '''
    lats = np.linspace(90, -90, 181)
    lons = np.linspace(0, 359, 360)
    lon_id = np.argmin(np.abs(lons - lonlat[0]))
    lat_id = np.argmin(np.abs(lats - lonlat[1]))
    time_list=[]
    t,r,u,v=np.zeros((len(levels),len(file_list))),np.zeros((len(levels),len(file_list))),np.zeros((len(levels),len(file_list))),np.zeros((len(levels),len(file_list)))
    for i,file1 in enumerate(file_list):
        time_list.append(file1.split('_')[1:3])
        data = np.load(file1)
        u[:,i]=data[0,len(levels)*0:len(levels)*1,lat_id,lon_id]
        v[:,i]=data[0,len(levels)*1:len(levels)*2,lat_id,lon_id]
        r[:,i]=data[0,len(levels)*2:len(levels)*3,lat_id,lon_id]
        t[:,i]=data[0,len(levels)*3:len(levels)*4,lat_id,lon_id]-273.15

    # 设置绘图参数
    fig = plt.figure(figsize=(15, 4))
    ax = plt.subplot(1, 1, 1)
    #########  设置x的坐标刻度 #########
    ###设置x坐标轴的主次刻度线
    x_MultipleLocator=4
    y_MultipleLocator=1
    ## 设置主刻度标签及此刻度标签的位置,
    ax.xaxis.set_major_locator(MultipleLocator(x_MultipleLocator))
    ax.xaxis.set_minor_locator(MultipleLocator(y_MultipleLocator))
    ## 设置刻度标签等信息
    
    # date_time = [i[0][4:6]+'-'+i[0][6:8]+'\n'+i[1]+':00' for i in time_list]
    
    date_time = []
    for i in file_list:
        ddd = i.split('/')[-1][4:-7]
        sss = (datetime.strptime(ddd,'%Y%m%d_%H')+timedelta(hours=8)).strftime("%Y%m%d%H")
        date_time.append(sss)

    date_time = [i[4:6]+'-'+i[6:8]+'\n'+i[8:10]+':00' for i in date_time] #2026022612
    # print(date_time)

    xticks = list(range(0, len(date_time), x_MultipleLocator))
    xlabels = [str(date_time[i]) for i in range(0, len(date_time), x_MultipleLocator)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=0, fontsize=12)
    ax.set_xlim(-0.8, len(date_time) - 0.2)
    ##########  设置y的坐标刻度 #########
    yticks = list(range(len(levels)))
    ylabels = [str(i) + 'hPa' for i in levels]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.set_ylim(-0.5, len(levels) - 0.5)
    # ax.invert_yaxis()  # 翻转纵坐标
    ########## 湿度等高线填充  #########
    r[r>100]=100


    original = plt.cm.Greens
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from matplotlib.colors import LinearSegmentedColormap

    # def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #     """
    #     截取 colormap 的一段区间，生成新的 colormap。
        
    #     参数：
    #         cmap : 原始 colormap 对象或名称
    #         minval : 起始位置 (0~1)
    #         maxval : 结束位置 (0~1)
    #         n : 采样点数，决定新 colormap 的平滑度
    #     返回：
    #         新的 LinearSegmentedColormap 对象
    #     """
    #     if isinstance(cmap, str):
    #         cmap = plt.get_cmap(cmap)
    #     # 在 [minval, maxval] 之间均匀采样 n 个点
    #     colors = cmap(np.linspace(minval, maxval, n))
    #     new_cmap = LinearSegmentedColormap.from_list(
    #         f'trunc({cmap.name},{minval:.2f},{maxval:.2f})', colors)
    #     return new_cmap
    # trunc_greens = truncate_colormap(original, 0.0, 0.5)

    ac = ax.contourf(date_time, yticks, r, cmap="Greens", levels=np.arange(50, 105, 5), extend='neither', alpha=0.75)
    cb = fig.colorbar(ac, shrink=1, pad=0.01)
    cb.ax.tick_params(labelsize=10)
    font = {'color': 'black',
            'size': 10
            }
    cb.set_label('Humidity (%)', fontdict=font)
    ########## 温度等温线  #########
    tt1 = [4, 8, 12, 16, 20, 24, 28, 32, 40, 50]
    ac1 = ax.contour(date_time, yticks, t, colors=['r'], levels=tt1, linewidths=0.5, linestyles='--')
    plt.clabel(ac1, inline=True, fontsize=10, inline_spacing=15, colors='r', fmt='%3.0f' + '℃')

    tt2 = [0]
    ac2 = ax.contour(date_time, yticks, t, colors=['r'], levels=tt2, linewidths=1.0, linestyles='-')
    plt.clabel(ac2, inline=True, fontsize=10, inline_spacing=15, colors='r', fmt='%3.0f' + '℃')

    tt3 = [-50, -40, -30, -20, -10, -6, -2]
    ac3 = ax.contour(date_time, yticks, t, colors=['b'], levels=tt3, linewidths=0.5, linestyles='--')
    plt.clabel(ac3, inline=True, fontsize=10, inline_spacing=15, colors='b', fmt='%3.0f' + '℃')
    ########## 风羽图  #########
    ax.barbs(date_time, yticks, u, v, barb_increments={'half': 2, 'full': 4, 'flag': 30}, length=4, pivot='middle')

    # 主标题
    if labels is None:
        titiles='ShiJiaZhuang:SKew-T'
    else:
        titiles=f'{labels}:SKew-T'
    ax.set_title(titiles, fontsize=10, pad=5)
    ax.set_title(f'begin:{time_list[0][0]} {time_list[0][1]}:00', loc='right', fontsize=10, pad=5)

    time_name = datetime.now().strftime('%Y%j_%H_%M_%S')
    img_filename = f"{time_list[0][0]}_SKew-T_t{time_name}.png"
    if plt_path is None:
        save_path = os.path.join(os.path.dirname(__file__), img_filename)
    else:
        save_path = os.path.join(plt_path, img_filename)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return



def plot_T2_timeseries(file_list, lonlat=[114, 38], dpi=200, labels=None, plt_path=None):
    '''
    绘制指定站点地面2m温度时间序列折线图
    
    :param file_list: 文件列表,如['fnl_20181001_00_00.npy','fnl_20181001_06_00.npy']
    :param lonlat: 经纬度,默认[114,38]石家庄
    :param dpi: 分辨率,默认200
    :param labels: 图片标题,默认为None,显示为站点温度时间序列
    :param plt_path: 图片保存路径,默认为None,保存至代码当前目录
    '''
    def get_xrdata(file_path):
        file1 = np.load(file_path)
        lats = np.linspace(90, -90, 181)
        lons = np.linspace(0, 359, 360)
        xr_data = xr.Dataset(coords={'lev': levels,'lat': lats,'lon': lons})
        for i,var in enumerate(['u','v','r','t','z']):
            xr_data[var] = xr.DataArray(
                file1[0,i*len(levels):(i+1)*len(levels), :, :],
                dims=['lev','lat', 'lon'],
                coords={'lev': levels,'lat': lats, 'lon': lons},)
        for i,var in enumerate(['t2','u10','v10','msl'],start=len(levels)*5):
            xr_data[var] = xr.DataArray(
                file1[0,i, :, :],
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons},)
        return xr_data
    
    # 提取站点温度数据
    lats = np.linspace(90, -90, 181)
    lons = np.linspace(0, 359, 360)
    lon_id = np.argmin(np.abs(lons - lonlat[0]))
    lat_id = np.argmin(np.abs(lats - lonlat[1]))
    
    times = []
    t2_values = []

    for file1 in file_list:
        # 解析文件名中的时间信息
        parts = file1.split('_')
        if len(parts) >= 3:
            date_str = parts[1]  # YYYYMMDD
            hour_str = parts[2]  # HH
            # time_str = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}:00"
            # sss = (datetime.strptime(time_str,'%Y-%m-%d_%H')+timedelta(hours=8)).strftime("%Y%m%d%H")
            # time_str = f"{date_str[4:6]}-{date_str[6:8]} {hour_str}"

            time_str = (datetime.strptime(date_str+hour_str,'%Y%m%d%H')+timedelta(hours=8)).strftime("%m-%d %H")




            times.append(time_str)
        else:
            times.append(f"File_{len(times)+1}")
        
        # 读取2m温度数据并转换为摄氏度
        data = get_xrdata(file1)
        t2_kelvin = data['t2'].isel(lat=lat_id, lon=lon_id).values
        t2_celsius = t2_kelvin - 273.15  # K -> ℃
        t2_values.append(t2_celsius)
    
    # 绘制折线图
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # 绘制温度折线
    ax.plot(times, t2_values, marker='o', linestyle='-', linewidth=2, markersize=8, color='#FF6B6B', label='2m Temperature')
    
    # 设置图形样式
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Temperature (℃)', fontsize=14)
    
    # 旋转x轴标签避免重叠
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 设置标题
    if labels is None:
        title = f'2m Temperature Time Series at Lon:{lonlat[0]}°, Lat:{lonlat[1]}°'
    else:
        title = f'{labels} - 2m Temperature Time Series'
    
    ax.set_title(title, fontsize=16, pad=20)
    
    # 添加最大值最小值标记
    max_idx = np.argmax(t2_values)
    min_idx = np.argmin(t2_values)
    
    ax.annotate(f'Max: {t2_values[max_idx]:.1f}℃', 
                xy=(max_idx, t2_values[max_idx]), 
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax.annotate(f'Min: {t2_values[min_idx]:.1f}℃', 
                xy=(min_idx, t2_values[min_idx]), 
                xytext=(10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    # 在右侧添加温度统计信息
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {np.mean(t2_values):.1f}℃\n'
    stats_text += f'Max: {np.max(t2_values):.1f}℃\n'
    stats_text += f'Min: {np.min(t2_values):.1f}℃\n'
    stats_text += f'Range: {np.max(t2_values)-np.min(t2_values):.1f}℃'
    
    ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, 
            verticalalignment='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    time_name = datetime.now().strftime('%Y%j_%H_%M_%S')
    img_filename = f"{time_name}_T2_timeseries_{lonlat[0]}_{lonlat[1]}_.png"
    if plt_path is None:
        save_path = os.path.join(os.path.dirname(__file__), img_filename)
    else:
        save_path = os.path.join(plt_path, img_filename)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Temperature time series plot saved to: {save_path}")
    return save_path




def plot_10m_wind_timeseries(file_list, lonlat=[114, 38], dpi=200, labels=None, plt_path=None):
    '''
    绘制指定站点近地面10米风速时间序列图
    
    :param file_list: 文件列表,如['fnl_20181001_00_00.npy','fnl_20181001_06_00.npy']
    :param lonlat: 经纬度,默认[114,38]石家庄
    :param dpi: 分辨率,默认200
    :param labels: 图片标题,默认为None,显示为站点风速时间序列
    :param plt_path: 图片保存路径,默认为None,保存至代码当前目录
    '''
    def get_xrdata(file_path):
        file1 = np.load(file_path)
        lats = np.linspace(90, -90, 181)
        lons = np.linspace(0, 359, 360)
        xr_data = xr.Dataset(coords={'lev': levels,'lat': lats,'lon': lons})
        for i,var in enumerate(['u','v','r','t','z']):
            xr_data[var] = xr.DataArray(
                file1[0,i*len(levels):(i+1)*len(levels), :, :],
                dims=['lev','lat', 'lon'],
                coords={'lev': levels,'lat': lats, 'lon': lons},)
        for i,var in enumerate(['t2','u10','v10','msl'],start=len(levels)*5):
            xr_data[var] = xr.DataArray(
                file1[0,i, :, :],
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons},)
        return xr_data
    
    def wind_direction(u, v):
        '''计算风向（0-360°，北风为0°，东风为90°）'''
        wd = np.degrees(np.arctan2(u, v))
        wd = (wd + 360) % 360
        return wd
    
    def wind_speed(u, v):
        '''计算风速'''
        return np.sqrt(u**2 + v**2)
    
    def beaufort_scale(ws):
        '''根据风速计算蒲福风级'''
        beaufort_thresholds = [0.3, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
        for i, threshold in enumerate(beaufort_thresholds):
            if ws < threshold:
                return i
        return 12
    
    def beaufort_description(level):
        '''蒲福风级描述'''
        descriptions = [
            '无风 (0)', '软风 (1)', '轻风 (2)', '微风 (3)', '和风 (4)', 
            '清风 (5)', '强风 (6)', '疾风 (7)', '大风 (8)', '烈风 (9)', 
            '狂风 (10)', '暴风 (11)', '飓风 (12+)'
        ]
        return descriptions[level]
    
    # 提取站点风场数据
    lats = np.linspace(90, -90, 181)
    lons = np.linspace(0, 359, 360)
    lon_id = np.argmin(np.abs(lons - lonlat[0]))
    lat_id = np.argmin(np.abs(lats - lonlat[1]))
    
    times = []
    u10_values = []
    v10_values = []
    ws_values = []
    wd_values = []
    beaufort_values = []
    
    for file1 in file_list:
        # 解析文件名中的时间信息
        parts = file1.split('_')
        if len(parts) >= 3:
            date_str = parts[1]
            hour_str = parts[2]
            time_str = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}:00"
            times.append(time_str)
        else:
            times.append(f"File_{len(times)+1}")
        
        # 读取10米风场数据
        data = get_xrdata(file1)
        u10 = data['u10'].isel(lat=lat_id, lon=lon_id).values
        v10 = data['v10'].isel(lat=lat_id, lon=lon_id).values
        
        u10_values.append(u10)
        v10_values.append(v10)
        
        # 计算风速和风向
        ws = wind_speed(u10, v10)
        wd = wind_direction(u10, v10)
        bf = beaufort_scale(ws)
        
        ws_values.append(ws)
        wd_values.append(wd)
        beaufort_values.append(bf)
    
    # ========== 关键修复：使用 GridSpec 创建合理的布局 ==========
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # 第一个子图：风速时间序列（占据上方整行）
    ax1 = fig.add_subplot(gs[0, :])
    
    # 第二个子图：风向玫瑰图（左下角）
    ax2_theta = fig.add_subplot(gs[1, :], projection='polar')
    
    # 第三个子图：风向时间序列（右下角）
    # ax2_line = fig.add_subplot(gs[1, 1])
    
    # ========== 绘制风速时间序列 ==========
    ax1.plot(times, ws_values, marker='o', linestyle='-', linewidth=2, markersize=6, color='#3498db', label='10m Wind Speed')
    
    # 填充不同风速等级区域
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1, 5))
    wind_levels = [0, 3, 6, 9, 12, 20]
    wind_labels = ['<3 m/s 轻风', '3-6 m/s 微风', '6-9 m/s 和风', '9-12 m/s 强风', '>12 m/s 大风']
    
    for i in range(len(wind_levels)-1):
        ax1.fill_between(times, wind_levels[i], wind_levels[i+1], 
                        where=[wind_levels[i] <= ws <= wind_levels[i+1] for ws in ws_values],
                        alpha=0.2, color=colors[i], label=wind_labels[i])
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=12, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 添加风速统计信息
    stats_text = f'Max: {np.max(ws_values):.1f} m/s\n'
    stats_text += f'Mean: {np.mean(ws_values):.1f} m/s\n'
    stats_text += f'Min: {np.min(ws_values):.1f} m/s'
    
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加蒲福风级参考线
    beaufort_ref = [0.3, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
    beaufort_labels = ['1级', '2级', '3级', '4级', '5级', '6级', '7级', '8级', '9级', '10级', '11级', '12级']
    
    for i, (ref, label) in enumerate(zip(beaufort_ref, beaufort_labels)):
        if ref < ax1.get_ylim()[1]:
            ax1.axhline(y=ref, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
            ax1.text(len(times)-0.5, ref+0.2, label, fontsize=8, color='gray', alpha=0.7)
    
    # ========== 绘制风向玫瑰图 ==========
    wd_radians = np.radians(wd_values)
    n_bins = 16
    theta_bins = np.linspace(0, 2*np.pi, n_bins+1)
    counts, _ = np.histogram(wd_radians, bins=theta_bins)
    
    sector_ws = []
    for i in range(n_bins):
        mask = (wd_radians >= theta_bins[i]) & (wd_radians < theta_bins[i+1])
        if np.any(mask):
            sector_ws.append(np.mean(np.array(ws_values)[mask]))
        else:
            sector_ws.append(0)
    
    max_ws = max(sector_ws) if max(sector_ws) > 0 else 1
    bars = ax2_theta.bar(theta_bins[:-1], counts, width=2*np.pi/n_bins, alpha=0.7, color=plt.cm.plasma(np.array(sector_ws)/max_ws))
    
    ax2_theta.set_theta_zero_location('N')
    ax2_theta.set_theta_direction(-1)
    ax2_theta.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax2_theta.set_title('Wind Direction Rose', fontsize=12, pad=20)
    
    # # ========== 绘制风向时间序列 ==========
    # norm_ws = np.array(ws_values) / max(ws_values) * 5 if max(ws_values) > 0 else np.zeros_like(ws_values)
    # u_norm = norm_ws * np.sin(np.radians(wd_values))
    # v_norm = norm_ws * np.cos(np.radians(wd_values))
    
    # x_positions = np.arange(len(times))
    # ax2_line.barbs(x_positions, np.zeros(len(times)), u_norm, v_norm, 
    #               barb_increments={'half': 1, 'full': 2, 'flag': 5},
    #               length=6, pivot='middle')
    
    # ax2_line.set_xlim(-0.5, len(times)-0.5)
    # ax2_line.set_ylim(-1, 1)
    # ax2_line.set_xticks(x_positions)
    # ax2_line.set_xticklabels([t.split()[-1] for t in times], rotation=45)
    # ax2_line.set_yticks([])
    # ax2_line.set_xlabel('Time (Hour)', fontsize=10)
    # ax2_line.set_title('Wind Direction (Barbs)', fontsize=12)
    # ax2_line.grid(True, linestyle='--', alpha=0.3)
    
    # 添加风向统计
    # wd_categories = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    # wd_bins = [(i*45-22.5, i*45+22.5) for i in range(8)]
    # wd_counts = []
    
    # for wd_min, wd_max in wd_bins:
    #     count = sum([1 for wd in wd_values if (wd_min % 360) <= (wd % 360) < (wd_max % 360)])
    #     wd_counts.append(count)
    
    # main_dir_idx = np.argmax(wd_counts)
    # main_dir = wd_categories[main_dir_idx]
    # main_dir_percent = wd_counts[main_dir_idx] / len(wd_values) * 100
    
    # ax2_line.text(0.02, 0.98, f'Main Wind Direction: {main_dir}\nFrequency: {main_dir_percent:.1f}%', 
    #              transform=ax2_line.transAxes, verticalalignment='top',
    #              fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置总标题
    if labels is None:
        title = f'10m Wind Time Series at Lon:{lonlat[0]}°, Lat:{lonlat[1]}°'
    else:
        title = f'{labels} - 10m Wind Time Series'
    
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # 保存图片
    time_name = datetime.now().strftime('%Y%j_%H_%M_%S')
    img_filename = f"{time_name}_10m_wind_timeseries_{lonlat[0]}_{lonlat[1]}.png"
    if plt_path is None:
        save_path = os.path.join(os.path.dirname(__file__), img_filename)
    else:
        save_path = os.path.join(plt_path, img_filename)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Wind time series plot saved to: {save_path}")
    return save_path


city_list = {'承德市': [ 40.96084, 117.92276], '张家口市': [ 40.83861667, 114.95091667], '秦皇岛市': [ 39.914525, 119.562275], '北京市': [ 40.0738125 , 116.40125417], '天津市': [ 39.18427619, 117.32580476], '唐山市': [ 39.666269, 118.19783 ], '廊坊市': [ 39.543025, 116.726325], '保定市': [ 38.88788556, 115.48081556], '沧州市': [ 38.311925, 116.856775], '石家庄市': [ 38.02426428, 114.48197254], '衡水市': [ 37.69155 , 115.651225], '邢台市': [ 37.07675, 114.5011 ], '邯郸市': [ 36.595835, 114.5024  ]}
def plot_twh_heatmap(file_list, city_list=city_list, levs=1000, dpi=200, labels=None, plt_path=None):
    '''
    :param file_list: 文件列表,如['fnl_20181001_00_00.npy','fnl_20181001_06_00.npy'];;
    :param city_list: 字典型城市列表经纬度,默认京津冀13城,格式如{'承德市': [ 40.96084, 117.92276], ...};;
    :param levs: 等压面高度,默认1000hPa-地面;;
    :param dpi: 分辨率,默认200;;
    :param labels: 图片标题,默认为None,显示为shijiazhuang:SKew-T,若有则显示{labels}:SKew-T;;
    :param plt_path: 图片保存路径,默认为None,保存至代码当前目录;;
    '''
    lats = np.linspace(90, -90, 181)
    lons = np.linspace(0, 359, 360)
    levels_id = levels.index(levs)
    city_names = list(city_list.keys())
    n_cities = len(city_names)
    city_indices = []  # 存储每个城市对应的网格索引
    for name, coord in city_list.items():
        lat0, lon0 = coord
        lon_id = np.argmin(np.abs(lons - lon0))
        lat_id = np.argmin(np.abs(lats - lat0))
        city_indices.append((lat_id, lon_id))
    n_times = len(file_list)
    if n_times == 0:
        print("Error: No file_list.")
        return
    # 初始化数据数组 (城市，时间)
    citys_t = np.zeros((n_cities, n_times))
    citys_rh = np.zeros((n_cities, n_times))
    citys_u = np.zeros((n_cities, n_times))
    citys_v = np.zeros((n_cities, n_times))
    time_list = []
    for i, file1 in enumerate(file_list):
        time_list.append(file1.split('_')[1:3])
        data = np.load(file1)
        for j, (lat_id, lon_id) in enumerate(city_indices):
            citys_u[j, i] = data[0, len(levels)*0+levels_id, lat_id, lon_id]
            citys_v[j, i] = data[0, len(levels)*1+levels_id, lat_id, lon_id]
            citys_rh[j, i] = data[0, len(levels)*2+levels_id, lat_id, lon_id]
            citys_t[j, i] = data[0, len(levels)*3+levels_id, lat_id, lon_id] - 273.15 # 转摄氏度
    citys_w = np.sqrt(citys_u**2 + citys_v**2)
    citys_rh[citys_rh > 100] = 100
    citys_rh[citys_rh < 0] = 0

    # 绘图
    ###设置x坐标轴的主次刻度线
    x_MultipleLocator=4
    y_MultipleLocator=1

    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax1 = plt.axes([0.06, 0.65, 0.8, 0.26])
    ax2 = plt.axes([0.06, 0.35, 0.8, 0.26])
    ax3 = plt.axes([0.06, 0.05, 0.8, 0.26])


    
    # date_time = [i[0][4:6]+'-'+i[0][6:8]+'\n'+i[1]+':00' for i in time_list]

    date_time = []
    for i in file_list:
        ddd = i.split('/')[-1][4:-7]
        sss = (datetime.strptime(ddd,'%Y%m%d_%H')+timedelta(hours=8)).strftime("%Y%m%d%H")
        date_time.append(sss)

    date_time = [i[4:6]+'-'+i[6:8]+'\n'+i[8:10]+':00' for i in date_time] #2026022612


    
    
    
    xticks = list(range(0, len(date_time), x_MultipleLocator))
    xlabels = [str(date_time[i]) for i in range(0, len(date_time), x_MultipleLocator)]
    im1 = ax1.imshow(citys_t, aspect='auto', cmap='RdYlBu_r', origin='upper', interpolation='none')
    im2 = ax2.imshow(citys_w, aspect='auto', cmap='YlOrRd', origin='upper', interpolation='none', vmin=0, vmax=8)
    im3 = ax3.imshow(citys_rh, aspect='auto', cmap='Blues', origin='upper', interpolation='none', vmin=30, vmax=100)
    im_labels = ['Temperature (°C)', 'Wind Speed (m/s)', 'Relative Humidity (%)']
    cb_settings = [
        (None, None, None),                     # 温度：默认
        (np.arange(0, 9, 2), 'max', None),     # 风速
        (np.arange(40, 101, 10), None, [f'{x}%' for x in np.arange(40, 101, 10)])]  # 湿度
    for ax,imx,label,(ticks, extend, tick_labels) in zip([ax1,ax2,ax3],[im1,im2,im3],im_labels,cb_settings):
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, ha='center', fontsize=8)
        ax.set_yticks(np.arange(n_cities))
        ax.set_yticklabels(city_names)
        cbar = fig.colorbar(imx, ax=ax, shrink=0.85, pad=0.015,extend=extend if extend else 'neither')
        cbar.set_label(label, fontsize=12)
        if ticks is not None:  # 自定义 colorbar 刻度
            cbar.set_ticks(ticks)
            if tick_labels is not None:
                cbar.set_ticklabels(tick_labels)
    
    # 主标题
    if labels is None:
        titiles='HeBei:HeatMap'
    else:
        titiles=f'{labels}:HeatMap'
    ax1.set_title(titiles, fontsize=18, pad=5)
    ax1.set_title(f'begin:{time_list[0][0]} {time_list[0][1]}:00', loc='right', fontsize=12, pad=5)

    time_name = datetime.now().strftime('%Y%j_%H_%M_%S')
    img_filename = f"{time_list[0][0]}_heatmap_t{time_name}.png"
    if plt_path is None:
        save_path = os.path.join(os.path.dirname(__file__), img_filename)
    else:
        save_path = os.path.join(plt_path, img_filename)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return


def plot_T2_RH2_U10V10(file_path, dpi=200, labels=None, plt_path=None,forecast_time=None, valid_time=None,out_name=''):
    '''
    绘制2米温度、相对湿度（1000hPa近似）和10米风场的组合图
    温度等值线：负值蓝色，正值红色，0值黑色；湿度用绿色填色

    Parameters
    ----------
    file_path : str
        输入数据文件路径（.npy格式）
    dpi : int, optional
        图片分辨率，默认200
    labels : str, optional
        自定义主标题，若为None则使用默认'T2, RH2 and U&V10'
    plt_path : str, optional
        图片保存路径，默认当前目录
    forecast_time : str, optional
        起报时间，例如 "2026022608"（北京时间）
    valid_time : str, optional
        预报时间，例如 "2026022808"（北京时间）
    '''
    # 读取数据（复用原逻辑）
    def get_xrdata(file_path):
        file1 = np.load(file_path)
        lats = np.linspace(90, -90, 181)
        lons = np.linspace(0, 359, 360)
        xr_data = xr.Dataset(coords={'lev': levels, 'lat': lats, 'lon': lons})
        for i, var in enumerate(['u', 'v', 'r', 't', 'z']):
            xr_data[var] = xr.DataArray(
                file1[0, i*len(levels):(i+1)*len(levels), :, :],
                dims=['lev', 'lat', 'lon'],
                coords={'lev': levels, 'lat': lats, 'lon': lons})
        for i, var in enumerate(['t2', 'u10', 'v10', 'msl'], start=len(levels)*5):
            xr_data[var] = xr.DataArray(
                file1[0, i, :, :],
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons})
        return xr_data

    # 加载数据
    ds = get_xrdata(file_path)

    # 定义目标区域（与图1一致）
    lon_min, lon_max = 110, 125
    lat_min, lat_max = 30, 45

    # 切片（注意纬度从大到小以保证绘图方向正确）
    ds_region = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

    # 提取变量
    t2 = ds_region['t2'].values - 273.15                     # 2米温度（℃）
    u10 = ds_region['u10'].values
    v10 = ds_region['v10'].values

    # 相对湿度：取1000hPa，若不存在则取最低层（如925hPa）
    if 1000 in ds_region.lev.values:
        rh = ds_region['r'].sel(lev=1000, method='nearest').values
    else:
        rh = ds_region['r'].isel(lev=0).values
        print("警告：1000hPa不存在，使用最低层（{} hPa）作为相对湿度近似".format(ds_region.lev.values[0]))

    rh = np.clip(rh, 0, 100)                                 # 限制在0-100

    # 经纬度网格
    lon = ds_region.lon.values
    lat = ds_region.lat.values

    # ========== 绘图 ==========
    fig = plt.figure(figsize=(8, 7))                          # 适当大小，铺满画布
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # 添加中国边界
    china = cfeature.ShapelyFeature(Reader(shape_path).geometries(), proj,
                                    edgecolor='k', facecolor='none')
    ax.add_feature(china, lw=1, alpha=0.8, zorder=2)

    # 网格线
    grid_lines = ax.gridlines(crs=proj, color='k', linestyle='--', alpha=0.3,
                              xlocs=np.arange(110, 126, 2), ylocs=np.arange(30, 46, 2),
                              draw_labels=True, x_inline=False, y_inline=False)
    grid_lines.top_labels = False
    grid_lines.right_labels = False

    # 1. 相对湿度填色（绿色系）
    # levels_rh = np.arange(0, 101, 10)
    levels_rh = [0,50,60,70,80,90,100]
    cf = ax.contourf(lon, lat, rh, levels=levels_rh, cmap='Greens', extend='both', transform=proj)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label('Relative Humidity (%)', fontsize=10)

    # 2. 2米温度等值线（分色）
    # 自动确定等值线间隔（可手动调整）
    t_min, t_max = np.floor(t2.min()/3)*3, np.ceil(t2.max()/3)*3
    levels_t = np.arange(t_min, t_max+0.1, 3)
    # 分别绘制负值、零、正值
    neg_levels = levels_t[levels_t < 0]
    zero_levels = levels_t[levels_t == 0]
    pos_levels = levels_t[levels_t > 0]

    if len(neg_levels) > 0:
        cs_neg = ax.contour(lon, lat, t2, levels=neg_levels, colors='b', linewidths=1.0, transform=proj,linestyles='solid',)
        clabels_neg = ax.clabel(cs_neg, inline=True, fontsize=8, fmt='%d°C', colors='b')
        for txt in clabels_neg:
            txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))

    # if len(zero_levels) > 0:
    #     cs_zero = ax.contour(lon, lat, t2, levels=zero_levels, colors='k', linewidths=1.5, transform=proj)
    #     clabels_zero = ax.clabel(cs_zero, inline=True, fontsize=8, fmt='%d°C', colors='k')
        # for txt in clabels_zero:
        #     txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0))

    if len(pos_levels) > 0:
        cs_pos = ax.contour(lon, lat, t2, levels=pos_levels, colors='r', linewidths=1.0, transform=proj)
        clabels_pos = ax.clabel(cs_pos, inline=True, fontsize=8, fmt='%d°C', colors='r')
        for txt in clabels_pos:
            txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))

    # 3. 10米风矢（下采样）
    skip = 1
    u10_sub = u10[::skip, ::skip]
    v10_sub = v10[::skip, ::skip]
    lon_sub = lon[::skip]
    lat_sub = lat[::skip]
    lon_sub_grid, lat_sub_grid = np.meshgrid(lon_sub, lat_sub)
    ax.barbs(lon_sub_grid, lat_sub_grid, u10_sub, v10_sub,
             transform=proj, length=5, linewidth=0.7,
             barb_increments=dict(half=2, full=4, flag=20),
             color='black')

    # 4. 标题
    if labels is None:
        title_main = 'T2, RH2 and U&V10 \n'
    else:
        title_main = labels

    if forecast_time and valid_time:
        title_time = f'Fengwu, {forecast_time}(BJT), Forecast:{valid_time}'
    else:
        title_time = os.path.basename(file_path)
    title_main = title_main+title_time
    ax.set_title(title_main, fontsize=12, pad=8)

    # 保存图片
    if plt_path is None:
        plt_path = os.path.dirname(__file__) or '.'
    os.makedirs(plt_path, exist_ok=True)
    # time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_name = f"T2_RH2_U10V10_{out_name}.png"
    save_path = os.path.join(plt_path, img_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至：{save_path}")
    return save_path

def plot_Wind850(file_path, dpi=200, labels=None, plt_path=None,forecast_time=None, valid_time=None,out_name = ''):
    '''
    绘制850hPa温度、风速和风场的组合图
    温度等值线：负值蓝色，正值红色，0值黑色；背景填色为850hPa风速（WS850）

    Parameters
    ----------
    file_path : str
        输入数据文件路径（.npy格式）
    dpi : int, optional
        图片分辨率，默认200
    labels : str, optional
        自定义主标题，若为None则使用默认'T850, WS850 and Wind850'
    plt_path : str, optional
        图片保存路径，默认为当前工作目录
    forecast_time : str, optional
        起报时间，例如 "2026022608"（北京时间）
    valid_time : str, optional
        预报时间，例如 "2026022808"（北京时间）
    '''
    # 读取数据（复用原逻辑）
    def get_xrdata(file_path):
        file1 = np.load(file_path)
        lats = np.linspace(90, -90, 181)
        lons = np.linspace(0, 359, 360)
        xr_data = xr.Dataset(coords={'lev': levels, 'lat': lats, 'lon': lons})
        for i, var in enumerate(['u', 'v', 'r', 't', 'z']):
            xr_data[var] = xr.DataArray(
                file1[0, i*len(levels):(i+1)*len(levels), :, :],
                dims=['lev', 'lat', 'lon'],
                coords={'lev': levels, 'lat': lats, 'lon': lons})
        for i, var in enumerate(['t2', 'u10', 'v10', 'msl'], start=len(levels)*5):
            xr_data[var] = xr.DataArray(
                file1[0, i, :, :],
                dims=['lat', 'lon'],
                coords={'lat': lats, 'lon': lons})
        return xr_data

    # 设置保存路径（修复 __file__ 问题）
    if plt_path is None:
        plt_path = os.getcwd()          # 改为当前工作目录
    os.makedirs(plt_path, exist_ok=True)

    
    # 加载数据
    ds = get_xrdata(file_path)

    # 定义目标区域
    lon_min, lon_max = 109, 126
    lat_min, lat_max = 29, 46

    # 选择850hPa层（若数据中没有850，则自动选最接近的层并警告）
    if 850 in ds.lev.values:
        lev_sel = 850
    else:
        lev_sel = ds.lev.values[np.argmin(np.abs(ds.lev.values - 850))]
        print(f"警告：850hPa不存在，使用 {lev_sel} hPa 作为近似")

    # ========== 修正数据选择 ==========
    # 1. 先通过 slice 选取经纬度范围（不指定 method）
    ds_region = ds.sel(lon=slice(lon_min, lon_max),
                       lat=slice(lat_max, lat_min))
    # 2. 再单独选取层次（使用 method='nearest'）
    ds_region = ds_region.sel(lev=lev_sel, method='nearest')

    # 提取变量
    t850 = ds_region['t'].values - 273.15                     # 温度（℃）
    u850 = ds_region['u'].values
    v850 = ds_region['v'].values
    ws850 = np.sqrt(u850**2 + v850**2)                         # 风速（m/s）

    # 经纬度网格
    lon = ds_region.lon.values
    lat = ds_region.lat.values

    # ========== 绘图 ==========
    fig = plt.figure(figsize=(8, 7))
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # 添加中国边界
    china = cfeature.ShapelyFeature(Reader(shape_path).geometries(), proj,edgecolor='k', facecolor='none')
    ax.add_feature(china, lw=1, alpha=0.8, zorder=2)

    # 网格线
    grid_lines = ax.gridlines(crs=proj, color='k', linestyle='--', alpha=0,
                              xlocs=np.arange(110, 126, 5), ylocs=np.arange(30, 46, 5),
                              draw_labels=True, x_inline=False, y_inline=False)
    grid_lines.top_labels = False
    grid_lines.right_labels = False

    # 1. 850hPa风速填色
    ws_max = np.ceil(ws850.max() / 2) * 2
    levels_ws = np.arange(0, ws_max + 0.1, 1)
    levels_ws = [8,9,10,11,12,13,14,15,16]
    cmap = ListedColormap(['#ffffff', '#E3F4F7', '#ABDEE7', '#74C8D7', '#67C9B2', '#63CF89', '#6AD66C', '#A3E684', '#DBF69B','#F6FDA6'])
    cf = ax.contourf(lon, lat, ws850, levels=levels_ws, cmap=cmap, extend='both', transform=proj)
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label('Wind Speed (m/s)', fontsize=10)

    
    # 3. 850hPa风矢（下采样）
    skip = 1
    u_sub = u850[::skip, ::skip]
    v_sub = v850[::skip, ::skip]
    lon_sub = lon[::skip]
    lat_sub = lat[::skip]
    lon_sub_grid, lat_sub_grid = np.meshgrid(lon_sub, lat_sub)
    ax.barbs(lon_sub_grid, lat_sub_grid, u_sub, v_sub,
             transform=proj, length=5, linewidth=0.7,
             barb_increments=dict(half=2, full=4, flag=20),
             color='black')

    # 4. 标题
    if labels is None:
        title_main = 'Wind Field at 850hPa\n'
    else:
        title_main = labels
    if forecast_time and valid_time:
        title_time = f'Fengwu, {forecast_time}(BJT), Forecast:{valid_time}'
    else:
        title_time = os.path.basename(file_path)
    title_main = title_main + title_time
    ax.set_title(title_main, fontsize=12, pad=8)


    # 保存图片

    img_name = f"Wind850_{out_name}.png"
    save_path = os.path.join(plt_path, img_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至：{save_path}")
    return save_path


def plot_t_uv_r(file_list,lonlat = [114, 38],dpi = 200,labels = None,plt_path = None ):
    
    # ==================== 参数设置 ====================
    levels = [100, 150, 200, 250, 300, 350, 400, 450, 500,
              550, 600, 650, 700, 750, 800, 850, 900,
              925, 950, 975, 1000]
    levels = levels[::-1]          # 反转，使纵坐标从上到下气压递减（低气压在上）
    
                # 图片保存路径，None 表示当前目录
    
    # ==================== 数据提取 ====================
    lats = np.linspace(90, -90, 181)
    lons = np.linspace(0, 359, 360)
    lon_id = np.argmin(np.abs(lons - lonlat[0]))
    lat_id = np.argmin(np.abs(lats - lonlat[1]))
    
    time_list = []
    t = np.zeros((len(levels), len(file_list)))
    r = np.zeros((len(levels), len(file_list)))
    u = np.zeros((len(levels), len(file_list)))
    v = np.zeros((len(levels), len(file_list)))
    
    for i, file1 in enumerate(file_list):
        time_list.append(file1.split('_')[1:3])          # 用于标题等
        data = np.load(file1)
        u[:, i] = data[0, len(levels)*0:len(levels)*1, lat_id, lon_id]
        v[:, i] = data[0, len(levels)*1:len(levels)*2, lat_id, lon_id]
        r[:, i] = data[0, len(levels)*2:len(levels)*3, lat_id, lon_id]
        t[:, i] = data[0, len(levels)*3:len(levels)*4, lat_id, lon_id] - 273.15   # 转换为摄氏度
    
    # 湿度最大不超过 100%
    r[r > 100] = 100
    
    # 计算风速
    wind = np.sqrt(u**2 + v**2)
    
    # ==================== 时间轴处理 ====================
    date_time = []
    for i in file_list:
        ddd = i.split('/')[-1][4:-7]                     # 提取文件名中的日期时间部分
        sss = (datetime.strptime(ddd, '%Y%m%d_%H') + timedelta(hours=8)).strftime("%Y%m%d%H")
        date_time.append(sss)
    
    # 生成 x 轴刻度标签：月-日\n时:00
    date_time = [i[4:6] + '-' + i[6:8] + '\n' + i[8:10] + ':00' for i in date_time]
    
    # 设置主次刻度间隔（单位：时次）
    x_MultipleLocator = 4
    y_MultipleLocator = 1
    
    xticks = list(range(0, len(date_time), x_MultipleLocator))
    xlabels = [str(date_time[i]) for i in range(0, len(date_time), x_MultipleLocator)]
    yticks = list(range(len(levels)))
    
    # ==================== 绘图 ====================
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True)
    
    # 定义三个子图的配置：变量、颜色映射、colorbar标签、子图标题
    plot_configs = [
        {'var': t,    'cmap': 'RdBu_r', 'label': 'Temperature (°C)', 'title': 'Temperature'},
        {'var': wind, 'cmap': 'YlOrRd', 'label': 'Wind Speed (m/s)', 'title': 'Wind Speed'},
        {'var': r,    'cmap': 'YlGnBu',  'label': 'Relative Humidity (%)', 'title': 'Humidity'},
    ]
    
    for ax, config in zip(axes, plot_configs):
        var = config['var']
        cmap = config['cmap']
        label = config['label']
        title = config['title']
    
        # 使用数值坐标（0,1,2,...）绘制填充等值线，保证坐标轴控制更清晰
        x = range(len(date_time))
        cf = ax.contourf(x, yticks, var, cmap=cmap, extend='both', alpha=1,levels=20)
        # 灰色细线，半透明
        cs = ax.contour(x, yticks, var, colors='black', linewidths=0.5, alpha=1,levels=10)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%2.0f')           # 添加标注
        
    
        # 添加 colorbar
        cb = fig.colorbar(cf, ax=ax, shrink=1, pad=0.01)
        cb.ax.tick_params(labelsize=10)
        cb.set_label(label, fontsize=10)
    
        # 设置 x 轴刻度及标签
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=0, fontsize=12)
        ax.set_xlim(-0.8, len(date_time) - 0.2)
    
        # 设置 y 轴刻度及标签（气压层）
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(lvl) + ' hPa' for lvl in levels], fontsize=12)
        ax.set_ylim(-0.5, len(levels) - 0.5)
    
        ax.set_title(title, fontsize=12)
    
    # 总标题
    if labels is None:
        suptitle = 'ShiJiaZhuang: Skew-T Profiles'
    else:
        suptitle = f'{labels}: Skew-T Profiles'
    fig.suptitle(suptitle, fontsize=14, y=1.02)   # y 略大于1，避免与子图标题重叠
    
    # ==================== 保存图片 ====================
    time_name = datetime.now().strftime('%Y%j_%H_%M_%S')
    img_filename = f"{time_list[0][0]}_Skew-T_3panels_{time_name}.png"
    if plt_path is None:
        save_path = os.path.join(os.path.dirname(__file__), img_filename)
    else:
        save_path = os.path.join(plt_path, img_filename)
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    file_list = []
    file_list = sorted([ i for i in os.listdir('.') if i.endswith('.npy')])
    
    plot_SKew_T(file_list, lonlat=[114,38], dpi=200, labels=None, plt_path=None)
    plot_T2_timeseries(file_list, lonlat=[114, 38], dpi=200, labels='ShiJiaZhuang', plt_path=None)
    plot_10m_wind_timeseries(file_list, lonlat=[114, 38], dpi=200, labels='ShiJiaZhuang', plt_path=None)
    plot_twh_heatmap(file_list)
    plot_t_uv_r(file_list,lonlat = [114, 38],dpi = 200,labels = None,plt_path = None )

    # for i in range(len(file_list)):
        
    #     v_time = (datetime.strptime(file_list[i][4:-7],'%Y%m%d_%H')+timedelta(hours=8)).strftime("%Y%m%d%H")
    #     f_time = file_list[0][4:-10].replace('_','')+'08'

    #     plot_T2_RH2_U10V10(file_list[i],forecast_time=f_time,valid_time=v_time,plt_path=None,out_name=file_list[i][:-4])
    #     plot_Wind850(file_list[i],forecast_time=f_time,valid_time=v_time,plt_path=None,out_name=file_list[i][:-4]) 
 
    

    


