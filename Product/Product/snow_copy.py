import gdal
import os
import pickle
import numpy as np
import datetime as dt
from Product import glovar
from Product.Dataprocess import interp, ecmwf, Writefile, Znwg
from Product.DataInterface import Datainterface


def saltepath(path):
    # 处理卫星数据，最后返回文件列表,path为卫星数据目录
    if not os.path.exists(path):
        print('The given path does not exist! Do not use modis data!')
        return None
    now = dt.datetime.now().strftime('%Y%m%d')
    pattern = r'(\w' + now + '.*?tif)'         # 匹配卫星文件名
    strings = os.listdir(path)
    tiflists = Znwg.regex(pattern, strings)    # 返回当前卫星数据文件列表
    if len(tiflists) == 0:
        print(r'MODIS tiff don\'t exist, call model_2!')
        return None
    else:
        print('MODIS exists, call model_1!')
        gdal.AllRegister()
        dataset = gdal.Open(tiflists)
        rtf = Datainterface.ReadTiff(dataset)
        px, py = rtf.imagexy2pro()
        pro2geov = np.vectorize(rtf.pro2geo)
        lon, lat = pro2geov(px, py)                                                # 此处执行很慢，循环操作
        # newlat =np.linspace(31.4,39.4,801)
        # newlon =np.linspace(89.3,103.1,1381)
        # *_, newdata = rtf.equalatlon('SnowDepth', dataset.ReadAsArray(), lat, lon, newlat, newlon)
        *_, newdata = rtf.equalatlon('SnowDepth', dataset.ReadAsArray(), lat, lon, glovar.lat, glovar.lon)
        return newdata


def snowData():
    # 获取ec数据信息(气温、降水、地温、湿度、积雪深度)
    ectime = ecmwf.ecreptime()
    fh = [i for i in range(12, 181, 3)]    # 20点的预报获取今天8:00的ec预报
    # *_, dics = Writefile.readxml(glovar.trafficpath, 0)
    *_, dics = Writefile.readxml(r'/home/cqkj/LZD/Product/Product/config/Traffic.xml', 0)
    dicslist = dics.split(',')[:-1]
    lonlatset, dataset = [], []
    for dic in dicslist:
        newdata = []
        lon, lat, data = Datainterface.micapsdata(ectime, dic, fh)
        lonlatset.append((lon, lat))
        for i in range(data.shape[0] - 1):
            if (np.isnan(data[i]).all() == True) and (i + 1 <= data.shape[0]):
                data[i] = data[i + 1] / 2
                data[i+1] = data[i + 1] / 2
                newdata.append(interp.interpolateGridData(data[i], lat, lon, glovar.lat, glovar.lon))
            else:
                newdata.append(interp.interpolateGridData(data[i], lat, lon, glovar.lat, glovar.lon))
        newdata = np.array(newdata)
        # newdata[newdata<0] = 0                    # 保证数据正确性
        dataset.append(newdata)                     # 保存插值后的数据集
    return np.array(dataset)

def presnow():
    # 得到前一时刻积雪深度
    ectime = ecmwf.ecreptime()
    fh = [0]
    dic = 'ECMWF_HR/SNOD'
    lon, lat, data = Datainterface.micapsdata(ectime, dic, fh)
    return interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon)

def reverse(saltedata, dataset, snowdepth):
    """
    # 加载模型生成积雪深度模型结果
    :param saltedata: 卫星数据
    :param dataset:   ec气象要素数据
    :return:
    """

    tmp = [data.reshape(-1, 1) for data in dataset]  # 转换基础要素
    ele = np.concatenate(tmp, axis=1)
    ele.resize(56, 801 * 1381, 4)                              # 转换形状，将上一时刻积雪输入
    temp = np.nan_to_num(ele)

    snowdepth = snowdepth.reshape(-1, 1)  # 积雪深度数据，仅包含前一时刻
    m1, m2, savepath, roadpath, indexpath, _ = Writefile.readxml(glovar.trafficpath, 0)[0].split(',')
    m2 = r'/home/cqkj/LZD/Product/Product/Source/snow.pickle'
    if saltedata is not None:
        with open(m1, 'rb') as f:
            model1 = pickle.load(f)
            #########################################
        saltedata.resize(801 * 1381, 1)
        typecode = 1
    else:
        with open(m2, 'rb') as f:
            model2 = pickle.load(f)
        typecode = 2
    alldata = []
    ################################################
    for i in range(56):
        # temp = [data.reshape(-1, 1) for data in dataset[i]]  # 仅包含基础要素
        # newdataset = np.concatenate([temp, snowdepth, saltedata], axis=1)
        if typecode == 1:
            newdataset = np.concatenate([temp[i], snowdepth, saltedata], axis=1)
            prediction = np.array(model1.predict(newdataset))  # 每轮结果
        if typecode == 2:
            #print(presnow.shape)
            # 此处预报结果不可用图像呈现出分块
            newdataset = np.concatenate([temp[i], snowdepth], axis=1)
            prediction = np.array(model2.predict(newdataset))  # 每轮结果
            predictions = np.nan_to_num(prediction)
            print(predictions.shape)
        snowdepth = predictions[:, np.newaxis]  # 结果作为下一次预测的输入
        predictions.resize(len(glovar.lat), len(glovar.lon))
        sdgrid = np.nan_to_num(predictions)
        sdgrid[sdgrid < 0] = 0
        alldata.append(sdgrid)
        sp = r'/data/traffic/snow//'
    Writefile.write_to_nc(sp, np.array(alldata), glovar.lat, glovar.lon, 'snowdepth', glovar.fnames, glovar.filetime)
    return np.array(alldata)  # 返回 [56, 801, 1381]网格数据

def main():
    dataset = snowData()
    snowdepth = presnow()
    reverse(None, dataset, snowdepth)

if __name__ == '__main__':
    main()



def index(sdgrid):
    with open(glovar.roadpath, 'rb') as f:
        roadmes = pickle.load(f)
        station_lat, station_lon = roadmes.lat, roadmes.lon
    sdgindex = np.piecewise(sdgrid, [sdgrid <= 0, (sdgrid > 0) & (sdgrid <= 5),
                                          (sdgrid > 5) & (sdgrid <= 10), sdgrid > 10], [0, 1, 2, 3])
    sdgindex = sdgindex.astype('int32')
    '''
    for sdgindex in sdgindex:
        sdgindex = interp.interpolateGridData(sdgindex, lat, lon, newlat=station_lat, newlon=station_lon, isGrid=False)
        sdindex.append(sdgindex)
    '''
    sdindex = [interp.interpolateGridData(sdg, glovar.lat, glovar.lon, newlat=station_lat, newlon=station_lon, isGrid=False) for sdg in sdgindex]
    return sdindex





















class SnowDepth(object):
    """积雪深度类，类方法包括省积雪深度计算，道路积雪深度指数"""
    def __init__(self, cpath):
        self.m1path, self.m2path, self.savepath, self.roadpath, self.indexpath = Writefile.readxml(cpath, 1)
        self.dics = Writefile.readxml(cpath, 2)[0].split(',')
        # 道路经纬度坐标
        self.new_lat = glovar.roadlat
        self.new_lon = glovar.roadlon
        #############################################################

    def clcsd(self,dataset, lat, lon, salte, snowdepth):
        """积雪深度计算，path为模型路径,新增卫星数据"""
        dataset.resize(56, 801*1381, 4)
        snowdepth = snowdepth.reshape(-1, 1)       # 积雪深度数据，仅包含初始时刻
        if salte is not None: 
            with open(self.m1path, 'rb') as f:
                model = pickle.load(f)
        #########################################
            salte.resize(801*1381, 1)
            alldata= []
        ################################################
            for i in range(56):
                temp = [data.reshape(-1, 1) for data in dataset[i]]  # 仅包含基础要素
                newdataset = np.concatenate([temp, snowdepth, salte], axis=1) 
                prediction = np.array(model.predict(newdataset))     # 每轮结果
                snowdepth = prediction                                # 结果作为下一次预测的输入
                prediction.resize(len(lat), len(lon))
                sdgrid = np.nan_to_num(prediction)
                sdgrid[sdgrid < 0] = 0
                alldata.append(sdgrid) 
                
        else:
            with open(self.m2path, 'rb') as f:
                model = pickle.load(f)
        ###############################################

            alldata= []
        ################################################
            for i in range(56):
                temp = [data.reshape(-1, 1) for data in dataset[i]]  # 仅包含基础要素
                newdataset = np.concatenate([temp, snowdepth], axis=1) 
                prediction = np.array(model.predict(newdataset))     # 每轮结果
                snowdepth = prediction                                #结果作为下一次预测的输入
                prediction.resize(len(lat), len(lon))
                sdgrid = np.nan_to_num(prediction)
                sdgrid[sdgrid < 0] = 0
                alldata.append(sdgrid)

        return alldata                                              # 返回 [56, 801, 1381]网格数据
    
    def clcindex(self, sdgrid, lat, lon):
        """
        积雪指数计算
        :param path:道路文件路径 
        :return: 道路指数数据
        """
        with open(self.roadpath, 'rb') as f:
            roadmes = pickle.load(f)
            station_lat, station_lon = roadmes.lat, roadmes.lon
        self.sdgindex = np.piecewise(sdgrid, [sdgrid<=0, (sdgrid>0)&(sdgrid<=5),
                                       (sdgrid>5)&(sdgrid<=10), sdgrid>10], [0,1,2,3])
        self.sdgindex = self.sdgindex.astype('int32')
        for sdgindex in self.sdgindex:
            sdgindex = interp.interpolateGridData(sdgindex, lat, lon, newlat=station_lat, newlon=station_lon, isGrid=False)
            self.sdindex.append(sdgindex)
        return self.sdindex

    def write(self, data, lat=None, lon=None, type=0):
        filetime = ecmwf.ecreptime()
        fh = range(3,169,3)
        fnames = ['_%03d' % i for i in fh]
        name = 'Snow'
        if type == 0:
            Writefile.write_to_nc(self.savepath,data, lat, lon, name,fnames, filetime)
        else :
            Writefile.write_to_csv(self.indexpath, data, 'SnowIndex',fnames,filetime)

'''
def saltedata(path):
    # path 为卫星数据目录
    pattern = dt.datetime.now().strftime('%Y%m%d')
    strings = os.listdir(path).sort()
    tifname = Znwg.regex(pattern, strings)
    
    if len(tifname) == 0:
        print('MODIS tiff don\'t exist, call model2!')
        return None
    else:
        print('call model1!(include MODIS)')
        gdal.AllRegister()
        dataset = gdal.Open(tifname)
        rtf = Datainterface.ReadTiff(dataset)
        px,py = rtf.imagexy2pro()
        pro2geov = np.vectorize(rtf.pro2geo)                         
        lon, lat = pro2geov(px, py)                                                # 此处执行很慢，循环操作
        newlat =np.linspace(31.4,39.4,801)
        newlon =np.linspace(89.3,103.1,1381)
        *_, newdata = rtf.equalatlon('SnowDepth', dataset.ReadAsArray(), lat, lon, newlat, newlon)
        return newdata
'''


def main(path):
    saltedata = saltepath(path)
    snowpre = np.random.randint(0, 1, size=(801*1381, 1))
    snow = SnowDepth()
    rep = ecmwf.ecreptime()
    fh = [i for i in range(12, 181, 3)]
    region = [float(i) for i in ','.join(Writefile.readxml(r'/home/cqkj/QHTraffic/Product/Traffic/SNOD/config.xml', 0)).split(',')]
    new_lon = np.arange(region[0], region[2], region[-1])
    new_lat = np.arange(region[1], region[3], region[-1])
    lonlatset, dataset = [], []
    # 提取数据及经纬度(双重循环，看能否改进)
    for dic in snow.dics:
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
    depthgrid = snow.clcsd(dataset, new_lat, new_lon, saltedata, snowpre)
    snow.write(depthgrid, new_lat, new_lon)
    dangerindex = snow.clcindex(depthgrid, new_lat, new_lon)
    snow.write(dangerindex, type=1)


def main():
    path = r'此处为MODIS遥感数据路径'
    saltedata = saltepath(path)
    snowpre = presnow()
    dataset = snowData()
    snowdepth = reverse(saltedata, dataset, snowpre)
    res = index(snowdepth)

if __name__ == '__main__':
    main()