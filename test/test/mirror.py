#  此程序用来完成要素获取
import GribData
import Writefile
import pickle
import datetime
import os
import pandas as pd
import numpy as np
import cimissdata, cleandata, interp, glovar, micapsdata, ecmwf, Writefile
from collections import deque


def mirrorGrib(path):
    # 用来同步8-20/20-8的实况格点场数据气温、降水、湿度， 20时同步今天8时数据， 8时同步昨日20时数据
    grid = GribData()
    now = datetime.datetime.now()
    elements, subdirs, localdirs, _, freq, ftp = Writefile.readxml(path, 0)    # freq应为None
    elements = elements.split(',')
    subdirs = subdirs.split(',')
    localdirs = localdirs.split(',')        # 为三个文件夹目录

    remote_urls = [os.path.join(subdir, now.strftime('%Y'), now.strftime('%Y%m%d')) for subdir in subdirs] # 构造三个路径

    for localdir, element, remote_url in zip(localdirs, elements, remote_urls):
        grid.mirror(element, remote_url, localdir, ftp)  # 同步至每个文件夹，此处待测试，需保证范围在08-20或次日20-当日08时
    # 查看各文件夹里数据信息,此处默认TEM, RAIN, RH 为08-20时的文件名列表
    TEMs, RAINs, RHs = [sorted(os.listdir(localdir)) for localdir in localdirs]      # 零时开始至今
    e2tTems = [tem for tem in TEMs if int(tem[-7:-5]) in range(8,21)]
    e2tRains = [rain for rain in RAINs if int(rain[-7:-5]) in range(8,21)]
    e2tRhs = [rh for rh in RHs if int(rh[-7:-5]) in range(8, 21)]
    # 认为形状为同一分辨率下的[12, lat * lon]
    temdata = [grid.readGrib(os.path.join(localdirs[0], TEM)) for TEM in e2tTems]
    raindata = [grid.readGrib(os.path.join(localdirs[0], RAIN)) for RAIN in e2tRains]
    rhdata = [grid.readGrib(os.path.join(localdirs[0], RH)) for RH in e2tRhs]

    return temdata, raindata, rhdata


def mirrorSkint(newlat, newlon, hours):
    # 从cimiss中获取所有站点地温插值,hours控制选取的时间间隔
    interfaceId = 'getSurfEleInRegionByTimeRange'
    elements = "Station_Id_C,Lat,Lon,GST,Year,Mon,Day,Hour"
    # 设定每天8：00、 20:00运行
    lastime = datetime.datetime.now().replace(minute=0, second=0) - datetime.timedelta(hours=8)
    firstime = lastime - datetime.timedelta(hours=hours)
    temp = ('[', firstime.strftime('%Y%m%d%H%M%S'), ',', lastime.strftime('%Y%m%d%H%M%S'), ')')
    Time = ''.join(temp)
    params = {'dataCode':"SURF_CHN_TRAFW_MUL",        # 需更改为地面逐小时资料
      'elements':elements,
      'timeRange':Time,
      'adminCodes':"630000"
      }
    initData = cimissdata(interfaceId, elements, **params)  # 获取出cimiss初始数据
    initData.Time = pd.to_datetime(initData.Time)
    # 需按时间进行划分插值，得到 地温网格数据
    timeList = sorted(list(initData.Time))      # 此处可能存在问题，目的仅按顺序取出来时间

    oneHourgrid = [interp.interpolateGridData(initData, initData.Lat, initData.Lon, newlat, newlon) for tm in timeList if initData[initData.Time == tm]]
    return oneHourgrid                         # oneHourgrid 格点数据形状应为[12, newlat, newlon]


def preSnow():
    # 得到前一时刻积雪深度, 返回插值后数据
    ectime = ecmwf.ecreptime()
    fh = [0]
    dic = 'ECMWF_HR/SNOD'
    lon, lat, data = micapsdata(ectime, dic, fh)
    return interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon)


def updateSnow(eleGrib, sktGrib, fsnow):
    # 更新积雪深度初始状态, fsnow为08\20点积雪
    temData, rainData, rhData = eleGrib

    with open(r'模型路径', 'rb')as f:
        model = pickle.load(f)
    snowgrid = deque([], maxlen=12)

    for tem, rain, rh, skt in zip(temData, rainData, rhData, sktGrib):
        fsnow = model.predict(np.concatenate([tem.reshape(-1, 1), rain.reshape(-1, 1), rh.reshape(-1, 1), skt.reshape(-1, 1), fsnow.reshape(-1, 1)], axis=1))

        snowgrid.append(fsnow)

    with open(r'积雪初始状态路径', 'wb') as f:
        pickle.dump(snowgrid, f)

    return None


def mirrorskgrib(path):
    # 还需写一个同步实况小时所需数据的代码（包括三网格), 滞后15分钟，可使用同一个config文件
    return mirrorGrib(path)




def mirrorskskt(newlat, newlon, hours):
    return mirrorSkint(newlat, newlon, hours)



def clcsnow(ele, snowdeque, model):
    # script 2
    # ele 为计算积雪的实况要素
    # 计算出逐一小时积雪，假设初始时刻积雪数据已存在, snowdeque为五个容量的双端队列路径
    with open(snowdeque, 'rb')as f:
        dq = pickle.load(f)                          # dq为包含五个元素的双端队列
    newsnow = model.predict(np.concatenate([ele, dq[-3]], axis=1))           # 假定[ele, data1]符合模型输入方式
    # 添加积雪实况入库
    # stationSnow = interp.interpolateGridData(newsnow, glovar.lat, glovar.lon, station.lat, station.lon)   # 插值到站点
    dq.append(newsnow)                               # 新增积雪网格
    with open(snowdeque, 'wb')as f:
        pickle.dump(dq)
    return newsnow


def clcRoadic(snowdepth, skint, rain):
    # 计算道路结冰厚度数据
    now = datetime.datetime.now().strftime('%Y%m%d%H')
    # #########
    with open(r'道路结冰模型', 'rb') as f:
        model = pickle.load(f)
    Roadic = model.predict([snowdepth.reshape[-1, 1], skint.reshape[-1, 1], rain.reshape[-1, 1]])
    Roadic.resize(1, glovar.lat, glovar.lon)
    path = r'填入网格文件的保存路径'
    Writefile.write_to_nc(path, Roadic, glovar.lat, glovar.lon, 'iceDepth', '', now)
    return Roadic


def main():
    path = r'此处提供config.xml文件路径'
    eleGrib = mirrorGrib(path)
    sktGrib = mirrorGrib(glovar.lat, glovar.lon, 12)     # glovar.lat， glovar.lon为网格经纬度信息
    fsnow = preSnow()
    updateSnow(eleGrib, sktGrib, fsnow)
    clcsnow(eleGrib, sktGrib, fsnow)



