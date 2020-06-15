
import pandas as pd
import datetime
from sklearn.neighbors import BallTree
#from read_sql import Mysql
from collections import deque
import os
import pickle
import numpy as np
import datetime as dt
import glovar
from Dataprocess import interp, ecmwf, Writefile, Znwg
from DataInterface import Datainterface


def mirrorGrib(path):
    # 6.14.16测试
    # 用来同步8-20/20-8的实况格点场数据气温(开式)、降水（累计）、湿度， 20时同步今天8时数据， 8时同步昨日20时数据
    grid = Datainterface.GribData()
    now = datetime.datetime.now()
    elements, subdirs, localdirs, _, freq, *ftp = Writefile.readxml(path, 0)    # freq应为None
    elements = elements.split(',')
    subdirs = subdirs.split(',')
    localdirs = localdirs.split(',')        # 为三个文件夹目录
    remote_urls = [os.path.join(subdir, now.strftime('%Y'), now.strftime('%Y%m%d')) for subdir in subdirs] # 构造三个路径
    for localdir, element, remote_url in zip(localdirs, elements, remote_urls):
        grid.mirror(element, remote_url, localdir, ftp)  # 同步至每个文件夹，此处待测试，需保证范围在08-20或次日20-当日08时
    # 查看各文件夹里数据信息,此处默认TEM, RAIN, RH 为08-20时的文件名列表
    RAINs, RHs, TEMs = [sorted(os.listdir(localdir)) for localdir in localdirs]      # 零时开始至今
    e2tTems = [tem for tem in TEMs if int(tem[-7:-5]) in range(8,21)]
    e2tRains = [rain for rain in RAINs if int(rain[-7:-5]) in range(8,21)]
    e2tRhs = [rh for rh in RHs if int(rh[-7:-5]) in range(8, 21)]
    # 认为形状为同一分辨率下的[12, lat * lon]
    tem = [Znwg.arrange(grid.readGrib(os.path.join(localdirs[2], TEM))) for TEM in e2tTems]                # temdata 包含四个要素（data, lat, lon, size）, 全国范围，需插值到青海
    lat, lon = tem[0][1], tem[0][2]
    temdata = np.array([np.nan_to_num(interp.interpolateGridData(t[0]-273.15, lat, lon, glovar.lat, glovar.lon)) for t in tem])
    raindata = np.array([np.nan_to_num(interp.interpolateGridData(Znwg.arrange(grid.readGrib(os.path.join(localdirs[0], RAIN)))[0], lat, lon, glovar.lat, glovar.lon)) for RAIN in e2tRains])
    rhdata = np.array([np.nan_to_num(interp.interpolateGridData(Znwg.arrange(grid.readGrib(os.path.join(localdirs[1], RH)))[0], lat, lon, glovar.lat, glovar.lon)) for RH in e2tRhs])
    return temdata, raindata, rhdata


def mirrorSkint(hours=12):
    # 6.14.16测试
    # 从cimiss中获取所有站点地温插值,hours控制选取的时间间隔
    interfaceId = 'getSurfEleInRegionByTimeRange'
    elements = "Station_Id_C,Lat,Lon,GST,Year,Mon,Day,Hour"
    # 设定每天8：00、 20:00运行
    lastime = datetime.datetime.now().replace(minute=0, second=0) - datetime.timedelta(hours=8)
    firstime = lastime - datetime.timedelta(hours=hours)
    temp = ('[', firstime.strftime('%Y%m%d%H%M%S'), ',', lastime.strftime('%Y%m%d%H%M%S'), ')')
    Time = ''.join(temp)
    params = {'dataCode':"SURF_CHN_MUL_HOR",        # 需更改为地面逐小时资料
      'elements':elements,
      'timeRange':Time,
      'adminCodes':"630000"
      }
    initData = Datainterface.cimissdata(interfaceId, elements, **params)  # 获取出cimiss初始数据
    initData.Time = pd.to_datetime(initData.Time)
    # 需按时间进行划分插值，得到 地温网格数据
    timeList = pd.to_datetime(initData.Time.unique()).strftime('%Y%m%d%H%M%S')    # 此处可能存在问题，目的仅按顺序取出来时间
    initData.set_index(initData.Time, inplace=True)
    oneHourgrid = [np.nan_to_num(interp.interpolateGridData(initData.loc[tm].GST.values.astype('float32'),
                                              initData.loc[tm].Lat.values.astype('float32'),
                                              initData.loc[tm].Lon.values.astype('float32'), glovar.lat, glovar.lon))
                   for tm in timeList]
    return oneHourgrid                         # oneHourgrid 格点数据形状应为[12, newlat, newlon]


def preSnow():
    # 得到前一时刻积雪深度, 返回插值后数据
    ectime = ecmwf.ecreptime()
    fh = [0]
    dic = 'ECMWF_HR/SNOD'
    lon, lat, data = Datainterface.micapsdata(ectime, dic, fh)
    return interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon)


def upSnowele(gbpath, sktpath):
    # 处理积雪历史要素文件, gbpath为格点要素pkl存放路径, sktpath为cimiss插值pkl路径
    grid = Datainterface.GribData()
    gbNlist, sktList = sorted(os.listdir(gbpath))[-5:], sorted(os.listdir(sktpath))[-5:]   #使用正则优化
    # 获取每天11时到17时， 23时到05时的数据，考虑取最后
    eleData = [Znwg.arrange(grid.readGrib(os.path.join(gbpath, gblist))) for gblist in gbNlist]
    sktData = [Znwg.arrange(grid.readGrib(os.path.join(gbpath, sktlist))) for sktlist in sktList]
    return np.array(eleData), np.array(sktData)


def updateSnow(eleGrib, sktGrib, fsnow, modelpath, savepath):
    # 6月15待测试
    # 更新积雪深度初始状态, fsnow为08点积雪
    temData, rainData, rhData = eleGrib[:, 0, ...], eleGrib[:, 1, ...], eleGrib[:, 2, ...]      # 均为[12, 901, 1401]shape
    # 根据初始积雪计算后三小时积雪，线性插值
    with open(modelpath, 'rb')as f:
        model = pickle.load(f)
    snowgrid = deque([fsnow], maxlen=12)
    t0, r0, rh0, sk0 = temData[0], rainData[0], rhData[0], sktGrib[0]
    ele0 = np.concatenate([t0.reshape(-1, 1), r0.reshape(-1, 1), rh0.reshape(-1, 1), sk0.reshape(-1, 1), fsnow.reshape(-1, 1)], axis=1)
    snowEleven = np.array(model.predict(np.nan_to_num(ele0))).reshape(901, 1401)
    snowNine, snowTen = fsnow + (1/3)*(snowEleven - fsnow), fsnow + (2/3)*(snowEleven - fsnow)
    # 添加初始时刻
    for snow in [snowNine, snowTen, snowEleven]:snowgrid.append(snow)
    # 开始计算

    for tem, rain, rh, skt in zip(temData[1:], rainData[1:], rhData[1:], sktGrib[1:]):
        fsnow = model.predict(np.concatenate([tem.reshape(-1, 1), rain.reshape(-1, 1), rh.reshape(-1, 1), skt.reshape(-1, 1), fsnow.reshape(-1, 1)], axis=1)).reshape(901, 1401)
        snowgrid.append(fsnow)

    with open(savepath, 'wb') as f:
        pickle.dump(snowgrid, f)

    return None


def mirrorskgrib(path):
    # 6月15待测试
    # 还需写一个同步实况小时所需数据的代码（包括三网格), 滞后15分钟，可使用同一个config文件
    grid = Datainterface.GribData()
    now = datetime.datetime.now()
    elements, subdirs, localdirs, _, freq, *ftp = Writefile.readxml(path, 0)    # freq应为None
    elements = elements.split(',')
    subdirs = subdirs.split(',')
    localdirs = localdirs.split(',')        # 为三个文件夹目录
    remote_urls = [os.path.join(subdir, now.strftime('%Y'), now.strftime('%Y%m%d')) for subdir in subdirs] # 构造三个路径
    for localdir, element, remote_url in zip(localdirs, elements, remote_urls):
        grid.mirror(element, remote_url, localdir, ftp)  # 同步至每个文件夹，此处待测试，需保证范围在08-20或次日20-当日08时
    # 查看各文件夹里数据信息,此处默认TEM, RAIN, RH 为08-20时的文件名列表
    RAIN, RH, TEM = [sorted(os.listdir(localdir))[-1] for localdir in localdirs]      # 零时开始至今
    tem = Znwg.arrange(grid.readGrib(os.path.join(localdirs[2], TEM)))
    lat, lon = tem[1], tem[2]
    temdata = np.array(np.nan_to_num(interp.interpolateGridData(tem[0] - 273.15, lat, lon, glovar.lat, glovar.lon)))
    raindata = np.array(np.nan_to_num(
        interp.interpolateGridData(Znwg.arrange(grid.readGrib(os.path.join(localdirs[0], RAIN)))[0], lat, lon,
                                   glovar.lat, glovar.lon)))
    rhdata = np.array(np.nan_to_num(
        interp.interpolateGridData(Znwg.arrange(grid.readGrib(os.path.join(localdirs[1], RH)))[0], lat, lon, glovar.lat,
                                   glovar.lon)))
    Time = datetime.datetime.now().strftime('%Y%m%d%H')
    savepath = ''.join(r'/home/cqkj/QHTraffic/tmp/ele', Time, r'.pkl')
    # 存储每个时刻的降水、湿度、温度
    with open(savepath, 'wb')as f:  # 文件名称用时间区分，精确到小时
        pickle.dump([temdata, raindata, rhdata], f)
    return temdata, raindata, rhdata

    '''
    e2tTems = [tem for tem in TEMs if int(tem[-7:-5]) in range(8,21)]
    e2tRains = [rain for rain in RAINs if int(rain[-7:-5]) in range(8,21)]
    e2tRhs = [rh for rh in RHs if int(rh[-7:-5]) in range(8, 21)]
    # 认为形状为同一分辨率下的[12, lat * lon]
    tem = [Znwg.arrange(grid.readGrib(os.path.join(localdirs[2], TEM))) for TEM in e2tTems]                # temdata 包含四个要素（data, lat, lon, size）, 全国范围，需插值到青海
    lat, lon = tem[0][1], tem[0][2]
    temdata = np.array([np.nan_to_num(interp.interpolateGridData(t[0]-273.15, lat, lon, glovar.lat, glovar.lon)) for t in tem])
    raindata = np.array([np.nan_to_num(interp.interpolateGridData(Znwg.arrange(grid.readGrib(os.path.join(localdirs[0], RAIN)))[0], lat, lon, glovar.lat, glovar.lon)) for RAIN in e2tRains])
    rhdata = np.array([np.nan_to_num(interp.interpolateGridData(Znwg.arrange(grid.readGrib(os.path.join(localdirs[1], RH)))[0], lat, lon, glovar.lat, glovar.lon)) for RH in e2tRhs])
    return temdata, raindata, rhdata
    return mirrorGrib(path)
    '''




def mirrorskskt(newlat, newlon, hours=1):
    # 6.14.16测试
    # 从cimiss中获取所有站点地温插值,hours控制选取的时间间隔
    interfaceId = 'getSurfEleInRegionByTimeRange'
    elements = "Station_Id_C,Lat,Lon,GST,Year,Mon,Day,Hour"
    # 设定每天8：00、 20:00运行
    lastime = datetime.datetime.now().replace(minute=0, second=0) - datetime.timedelta(hours=8)
    firstime = lastime - datetime.timedelta(hours=hours)
    temp = ('[', firstime.strftime('%Y%m%d%H%M%S'), ',', lastime.strftime('%Y%m%d%H%M%S'), ']')
    Time = ''.join(temp)
    params = {'dataCode':"SURF_CHN_MUL_HOR",        # 需更改为地面逐小时资料
      'elements':elements,
      'timeRange':Time,
      'adminCodes':"630000"
      }
    initData = Datainterface.cimissdata(interfaceId, elements, **params)  # 获取出cimiss初始数据
    initData.Time = pd.to_datetime(initData.Time)
    # 需按时间进行划分插值，得到 地温网格数据
    timeList = pd.to_datetime(initData.Time.unique()).strftime('%Y%m%d%H%M%S')    # 此处可能存在问题，目的仅按顺序取出来时间
    initData.set_index(initData.Time, inplace=True)
    oneHourgrid = [np.nan_to_num(interp.interpolateGridData(initData.loc[tm].GST.values.astype('float32'),
                                              initData.loc[tm].Lat.values.astype('float32'),
                                              initData.loc[tm].Lon.values.astype('float32'), glovar.lat, glovar.lon))
                   for tm in timeList]
    Time = datetime.datetime.now().strftime('%Y%m%d%H')
    savepath = ''.join(r'/home/cqkj/QHTraffic/tmp/skt', Time, r'.pkl')
    with open(savepath, 'rb') as f:    # 文件名用时间区分精确到小时，供updateSnow使用
        pickle.dump(oneHourgrid, f)

    return oneHourgrid


def clcsnow(ele, snowdeque, modelpath, skt, station):
    # script 2， 6月15待测试, snowdeque为积雪初始时刻文件， moodelpath为积雪模型路径，skt为积雪实况, station为站点df
    # ele 为计算积雪的实况要素
    # 计算出逐一小时积雪，假设初始时刻积雪数据已存在, snowdeque为12个容量的双端队列路径
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    with open(snowdeque, 'rb')as f:
        dq = pickle.load(f)                          # dq为包含12个元素的双端队列

    t, r, rh = ele
    ele = np.concatenate([t.reshape(-1, 1), r.reshape(-1, 1), rh.reshape(-1, 1), skt.reshape(-1, 1), dq[-2].reshape(-1, 1)], axis=1)
    newsnow = model.predict(np.nan_to_num(ele)).reshape(901, 1401)           # 假定[ele, data1]符合模型输入方式
    # 添加积雪实况入库
    stationSnow = np.nan_to_num(interp.interpolateGridData(newsnow, glovar.lat, glovar.lon, station.lat.values, station.lon.values))   # 插值到站点
    # 完成插入并更新操作
    time = datetime.datetime.now().strftime('%Y%m%d%H')
    alltime = np.repeat(time, len(stationSnow))
    df = pd.DataFrame(stationSnow[:, np.newaxis], columns=['snow'])
    Time = pd.DataFrame(alltime[:, np.newaxis], columns=['Time'])
    # dataframe = pd.concat([dt, df, Time], axis=1)


    dq.append(newsnow)                               # 新增积雪网格
    with open(snowdeque, 'wb')as f:
        pickle.dump(dq, f)
    return newsnow, r


def clcRoadic(snowdepth, skint, rain, station, modelpath, savepath):
    # 计算道路结冰厚度数据, rain从积雪要素同步里拿, 待测试
    now = datetime.datetime.now().strftime('%Y%m%d%H')
    # #########
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    ele = np.concatenate([snowdepth.reshape(-1, 1), skint.reshape(-1, 1), rain.reshape(-1, 1)], axis=1)
    Roadic = model.predict(ele)
    Roadic.resize(1, 901, 1401)
    # 保存一小时结冰网格
    #path = savepath
    #Writefile.write_to_nc(savepath, Roadic, glovar.lat, glovar.lon, 'iceDepth', '', now)

    # 结冰厚度插值到站点
    stationSnow = np.nan_to_num(interp.interpolateGridData(Roadic, glovar.lat, glovar.lon, station.lat.values, station.lon.values, isGrid=False))   # 插值到站点
    # 完成插入并更新操作
    time = datetime.datetime.now().strftime('%Y%m%d%H')
    alltime = np.repeat(time, len(stationSnow))
    df = pd.DataFrame(stationSnow[:, np.newaxis], columns=['snow'])
    Time = pd.DataFrame(alltime[:, np.newaxis], columns=['Time'])
    dataframe = pd.concat([station, df, Time], axis=1)

    return dataframe


def clcIcingindex(dbmessage):
    # 计算道路结冰指数， dbmessage为数据库
    ice_depth = dbmessage[['id', 'lat', 'lon', 'Time', 'ice']]
    # 根据时间聚合id, ice
    dz = ice_depth.groupby(['id', pd.Grouper(key='Time', freq='D')]).mean()
    dz.reset_index()
    timeList = pd.to_datetime(dz.Time.unique()).strftime('%Y%m%d%H%M%S')
    print(timeList)
    allDaydata = [dz.loc[index].ice.values for index in timeList]
    # 转化为网格数据并转换位置
    daynd = np.concatenate(allDaydata, axis=1)[::, ::-1]
    index_array = np.argmin(daynd, axis=1)    # 得到道路结冰天数
    res = np.piecewise(index_array, [index_array == 0, (index_array > 0) & (index_array <= 2), (index_array > 2) & (index_array <= 5),
                                     (index_array > 5) & (index_array <= 10), index_array > 10], [1, 2, 3, 4, 5])
    return index

















def main():
    path = r'此处提供config.xml文件路径'
    eleGrib = mirrorGrib(path)
    sktGrib = mirrorGrib(glovar.lat, glovar.lon, 12)     # glovar.lat， glovar.lon为网格经纬度信息
    fsnow = preSnow()
    updateSnow(eleGrib, sktGrib, fsnow)
    clcsnow(eleGrib, sktGrib, fsnow)



