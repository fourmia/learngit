import pickle
import numpy as np
import pandas as pd
import glovar
from DataInterface import Datainterface
from Dataprocess import Writefile, interp, ecmwf


class Roadic(object):
    def __init__(self):
        self._path = glovar.trafficpath
        self.mpath, self.roadpath = Writefile.readxml(self._path, 1)[:2]
        self.dics = Writefile.readxml(self._path, 2)[0].split(',')

    def icegrid(self, dataset, lat, lon):
        with open(self.mpath, 'rb')as f:
            model = pickle.load(f)
        temp = [data.reshape(-1, 1) for data in dataset]
        prediction = np.array(model.predict(temp))
        prediction.resize(56, len(lat), len(lon))
        icegrid = np.nan_to_num(prediction)
        icegrid[icegrid < 0] = 0
        return icegrid

    def depth2onezero(self, icegrid, lat,lon):
        with open(self.roadpath, 'rb') as f:
            roadmes = pickle.load(f)
        station_lat, station_lon = roadmes.lat, roadmes.lon
        icegrid = np.where(icegrid>0.05, 1, 0)
        iceindex = []
        for i in range(56):
            iceindex.append(interp.interpolateGridData(icegrid[i], lat, lon, station_lat, station_lon, isGrid=False))
        iceroad = np.concatenate(iceindex, axis=1)
        return iceroad

class RoadIceindex(object):
    def __init__(self, cmissdata, predata):
        self._cmdata = cmissdata
        self._predata = predata
        print(self._predata.shape)

    def iceday(self):
        # dfpre存储一个预测列表
        """
        判断结冰天数：分两部分，第一部分是判断出实况的天数
        第二部分则是判断预测时刻到预测开始的天数
        """
        # dfpre为未来七天结冰状况数组
        dfpre = self._predata
        # 实况影响
        # self._cmdata = self._cmdata.to_numpy()
        # 此处为替换代码
        predatas = self._cmdata[::, ::-1]
        pre_index = np.argmin(predatas, axis=1)
        # 此处为替换代码
        # pre_index = 0
        # 预报影响,总计生成56个结果
        # 生成一个用于拼接的数组形状，最后去掉
        middin_var = []
        for i in range(1, 57):
            tmp = dfpre[:, :i]  # 取出当前列和之前列
            tmp = tmp[::, ::-1]  # 转换位置从当前列开始数
            print('tmp shape is:{}'.format(tmp.shape))
            now_index = np.argmin(tmp, axis=1)  # 得到距离当前列最近的位置
            print('pre_index shape:{}'.format(pre_index.shape))
            print('now_index shape:{}'.format(now_index.shape))
            # index = np.where(now_index>0, now_index+pre_index,0)
            index = np.piecewise(now_index, [(now_index >= 0) & (now_index < i), now_index == i],
                                 [now_index, lambda now_index: now_index + pre_index])
            middin_var.append(index[:, np.newaxis])
        # 假设结冰状态不会突变
        middin_var = np.concatenate(middin_var, axis=1)
        print('middin_var shape:{}'.format(middin_var.shape))
        # 得到天数信息
        index = np.where(middin_var > 0, middin_var / 8 + 1, 0).astype(int)
        index = np.piecewise(index, [index == 0, (index > 0) & (index <= 2), (index > 2) & (index <= 5),
                                     (index > 5) & (index <= 10), index > 10], [1, 2, 3, 4, 5])
        print('index shape:{}'.format(index.shape))
        return index
        # return middin_var


def write(path, data, name, lat=None, lon=None, type=0):
    filetime = ecmwf.ecreptime()
    fh = range(3, 169, 3)
    fnames = ['_%03d' % i for i in fh]
    if type == 0:
        Writefile.write_to_nc(path, data, lat, lon, name, fnames, filetime)
    else:
        Writefile.write_to_csv(path, data, name, fnames, filetime)


def main():
    ice = Roadic()
    rep = ecmwf.ecreptime()
    fh = [i for i in range(12, 181, 3)]
    region = [float(i) for i in ','.join(Writefile.readxml(glovar.trafficpath, 0)).split(',')]
    new_lon = np.arange(region[0], region[2], region[-1])
    new_lat = np.arange(region[1], region[3], region[-1])
    lonlatset, dataset = [], []
    # 提取数据及经纬度(双重循环，看能否改进)
    for dic in ice.dics:
        lon, lat, data = Datainterface.micapsdata(rep, dic, fh)
        lonlatset.append((lon, lat))
        for i in range(data.shape[0] - 1):
            if (np.isnan(data[i]).all() == True) and (i + 1 <= data.shape[0]):
                data[i] = data[i + 1] / 2
                data[i+1] = data[i + 1] / 2
                interp.interpolateGridData(data,lat,lon,new_lat, new_lon)
            else:
                interp.interpolateGridData(data, lat, lon,new_lat, new_lon)
        dataset.append(data)                     # 保存插值后的数据集
    icgrid = ice.icegrid(dataset, new_lat, new_lon)
    savepath, indexpath = Writefile.readxml(glovar.trafficpath, 1)[2:]
    write(savepath, icgrid, 'Roadic', new_lat, new_lon)               # 先保存厚度网格数据
    iceroad = ice.depth2onezero(icgrid, new_lat, new_lon)
    ################################################################################
    # 获取cimiss数据集,此处仅为读取，实况数据获取及保存由另一程序实现
    cmissdata = np.loadtxt('/home/cqkj/QHTraffic/qhroadic/cmsk.csv', delimiter=',')
    icedays = RoadIceindex(cmissdata, iceroad)
    roadicing = icedays.iceday()
    write(indexpath, roadicing, 'RoadicIndex', type=1)


if __name__ == '__main__':
    main()
