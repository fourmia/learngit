# 此程序用来完成预报入库，取每个批次的前四个文件
import numpy as np
import pandas as pd
import datetime
import sklearn.neighbors import BallTree
from read_sql import Mysql
from collections import deque
import pickle
import interp, Write_to_csv, Writr_to_db, Writefile
import ectime
from DataInterface.Datainterface import micaps, GribData


def readdb():
    # 此处先从csv文件中读取并入库
    path = r'C:\Users\GJW\Desktop\trafficlist.csv'
    datas = pd.read_csv(path, index_col=0)
    return datas


def todbs(data, lat, lon, dt, name, numbs=4):
    # 此段代码完成预报要素入库
    # 当前代码段, i为0-56， data.shape(56,801,1381), point为道路点信息， dt为站点信息
    # 生成nc、csv的同时进行预报值的入库操作
    list1 = []
    for i in range(data.shape[0]):
        if i < numbs:
            # 对应到待检验站点上，有的话用in，没有则插值, name为df列名
            tmp = np.nan_to_num(interpolateGridData(data[i], lat, lon, dt.Lat, dt.Lon, isGrid=None)) # 采用插值方法，未想好站点对应
            print(tmp)
            time = (datetime.datetime.now() + datetime.timedelta(hours=i*3)).strftime('%Y%m%d%H')                   # 获取时间并填充至整列
            print(time)
            alltime = np.repeat(time, len(tmp))
            print(alltime)
            df = pd.DataFrame(tmp[:, np.newaxis], columns=[name])
            Time = pd.DataFrame(alltime[:, np.newaxis], columns=['Time'])
            dataframe = pd.concat([dt, df, Time], axis=1)
            # 需执行入库操作 df.todbs(),更新并入库，防止错误覆盖
            list1.append(dataframe)
        else:
            break
    return list1


def floodindex(rain, prerain):
    #  计算出实况淹没指数, rain为逐三小时降水， prerain为当前时刻前三小时降水
    pre = np.zeros_like(prerain)   # prerain的前一时刻积水深度
    preraindepth = cl2(prerain[np.newaxis, :], pre)
    rainnow = rain.reshape(-1, pre.shape[0], 1)  # 转换形状（分割天数）
    target = floodindex(rainnow, preraindepth)
    return target


def hfindex(roaddata, pdata, wdata):
    # 计算出实况横风指数，roaddata为带有道路走向的数据， df为交通沿线站点, wdata为风向、风速数据
    # 需根据df中的经纬度，寻找到距离其最近的道路走向
    tree = BallTree(roaddata.loc[:, ['Lon', 'Lat']].values, leaf_size=2)
    _, index = tree.query(pdata.loc[:,['Lon','Lat']].values,k=1)
    res = roaddata.iloc[index.ravel(), :]
    tmps = res.loc[:, ['ur', 'vr']]
    newstation = pd.concat([pdata, tmps], axis=1)     # 为道路附近站点添加道路走向信息
    roadd = newstation.loc[:, ['ur', 'vr']]           # 为站点最近道路走向(97, 2）
    # 须将风向、风速转化为UV风
    ws, wd = wdata
    v=-ws*np.cos((wd*np.pi)/180)
    u=-ws*np.sin((wd*np.pi)/180)
    # 计算出走向垂直方向的风速大小
    wind = np.array([u, v])
    crosswind = np.cross(roadd, wind)               # 计算出横风风速
    w = 1 / 2 * 1.29 * (np.square(crosswind)) * 1000
    index = np.piecewise(w, [w < 83, (w >= 83) & (w < 134), (w >= 134) & (w < 602), (w >= 602) & (w < 920), w >= 920],
                     [1, 2, 3, 4, 5])               # 计算出指数
    return index


def snow(ele, snowdeque, model):
    # script 2
    # ele 为计算积雪的实况要素
    # 计算出逐一小时积雪，假设初始时刻积雪数据已存在, snowdeque为五个容量的双端队列路径
    with open(snowdeque, 'rb')as f:
        dq = pickle.load(f)                          # dq为包含五个元素的双端队列
    newsnow = model.predict([ele, dq[-3]])           # 假定[ele, data1]符合模型输入方式
    dq.append(newsnow)                               # 新增积雪网格
    with open(snowdeque, 'wb')as f:
        pickle.dump(dq)
    return newsnow








def updatedq(prele, eightdata):
    # script 1
    # 取之后的EC数据补全之前EC, prele为实况格点场数据，主要包括气温、地温、湿度、降水
    # 降水应为累计降水， 目前资料为实况逐小时降水
    times = ectime()      #  获取ec时间,prele需保证满足[8-20/ 20-8],从实况格点场中获取8.， 11.， 14.， 17.， 20。
    grid = GribData()     #  实况格点场数据同步
    grid.mirror()
    *_, elements, ftp = Writefile.readxml(r'/home/cqkj/LZD/Product/Product/config/Traffic.xml', 5)
    element = elements.split(',')
    ftp = ftp.split(',')
    grib = GribData()
    now = datetime.datetime.now()
    remote_url = os.path.join(r'\\SPCC\\BEXN', now.strftime('%Y'), now.strftime('%Y%m%d'))
    grib.mirror(element[0], remote_url, element[1], element[2], ftp)
    #  同步四个类型每种类型同步五个，认为目前已经同步了
    dq = deque(maxlen=5)
    for pre in prele:
        # 取出每批数据
        dq.append(model.predict([pre, eightdata]))
    path = r''
    with open(path, 'wb') as f:
        pickle.dump(dq, f)
    return dq


def roadic(snow, rain, skint, model):
    # 计算出实况道路结冰指数，需先进行道路网格实况计算， 需先进行积雪深度（计算到逐小时）
    # 道路结冰逐小时网格积雪、降水、地温
    data = model.predict([snow, rain, skint])            # 认为输入符合模型结构
    return data


def




































def main():
    mysql = Mysql(host, port, user, pwd, db)







if __name__ == "__main__":
    data = np.random.randint(0, 2, size=(56, 801, 1381))
    todbs(data)
