# 基本要素实况
import pymysql
import sqlalchemy
import numpy as np
import pandas as pd
import datetime as dt
from sqlalchemy import create_engine
from Model import cal2
import glovar
from Dataprocess import interp


def floodsk(dbmes, dbindex):
    # 6.15待测试
    # 计算实况淹没指数信息, dbmes为要素实况数据库信息，dbindex为指数实况数据信息, dataframe格式
    pretime = (dt.datetime.now() - dt.timedelta(hours=1)).strftime('%Y%m%d%H')   #前一时刻实况数据时间
    # 前一时刻降水
    dbmes.set_index(dbmes.Time, inplace=True)
    preraindf = dbmes.loc[pretime][['所包含的要素']]

    # 前一时刻积水
    dbindex.set_index(dbindex.Time, inplace=True)
    preflood = dbindex.loc[dbindex][['所包含的要素']]

    if len(preflood) == 0:
        preflood = np.ones_like(preraindf.rain.values, dtype='int32')

    rain = preraindf.rain.values[np.newaxis,:]       # 当前时刻降水
    fd = cal2(rain, preflood)
    return fd


def windsk(dbmes):
    # 6.15待测试
    # 计算实况横风指数信息，dbmes为要素实况数据库信息
    now_time = dt.datetime.now().strftime('%Y%m%d%H')
    # 取出当前风向风速
    dbmes.set_index(dbmes.Time, inplace=True)
    wind_mes = dbmes.loc[now_time][['所包含的要素']]
    ws, wd = wind_mes.ws.values, wind_mes.wd.values

    # 转化为当前u风v风，并构建uv
    u, v = -ws*np.sin((wd*np.pi)/180), -ws*np.cos((wd*np.pi)/180)
    value = [[uvalue, vvalue] for uvalue, vvalue in zip(u, v)]

    # 取出走向信息
    length = np.linalg.norm(dbmes.iloc[:, -2:], axis=1)
    ur = np.divide(dbmes['ur'], length)
    vr = np.divide(dbmes['vr'], length)
    uv = [[i, j] for i, j in zip(ur, vr)]

    # 计算风压等级
    dataset= np.cross(uv, value)
    w = 1 / 2 * 1.29 * (np.square(dataset)) * 1000
    w = np.piecewise(w, [w < 83, (w >= 83) & (w < 134), (w >= 134) & (w < 602), (w >= 602) & (w < 920), w >= 920],
                     [1, 2, 3, 4, 5])

    # 插入并更新
    time = dt.datetime.strftime('%Y%m%d%H')  # 此处应换为智能网格time
    alltime = np.repeat(time, len(w))
    df = pd.DataFrame(w[:, np.newaxis], columns=['windindex'])
    Time = pd.DataFrame(alltime[:, np.newaxis], columns=['Time'])
    dataframe = pd.concat([dt, df, Time], axis=1)
    return dataframe

'''
def icingsk(dbstations, dbmes, icgrid):
    # 6.15待测试， icgrid为实况结冰厚度网格
    # 实况结冰指数信息,由每天结冰厚度网格插值得到结冰厚度，对结冰厚度进行编码，再根据结冰天数得到结冰指数
    now_time = dt.datetime.now().strftime('%Y%m%d%H')
    # 插值至对应道路点
    icestation = np.nan_to_num(interp.interpolateGridData(icgrid, glovar.lat, glovar.lon, dbstations.Lat, dbstations.lom))
    icing = np.piecewise(icestation,[icestation>0.01, icestation<=0.01],[1, 0]) # 0-1编码，表征是否结冰

    # 连续结冰天数统计，获取实况结冰厚度信息
    dbmes.set_index(dbmes.Time, inplace=True)
    ice_mes = dbmes.loc[:now_time][['所包含的数据要素']]

    # 需按时间进行划分，得到每三个小时的结冰厚度序列，保证站点对应
    timeList = pd.to_datetime(ice_mes.Time.unique()).strftime('%Y%m%d%H%M%S')

    # 根据每三小时，构建一个ndarray，每列为每三个小时
    preIce = [ice_mes.loc[tl].icedepth.values for tl in timeList]
'''