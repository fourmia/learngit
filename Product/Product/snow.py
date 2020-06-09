import pickle
import numpy as np
import datetime as dt
#　包结构使用相对路径
from Product.Dataprocess import interp, ecmwf, Writefile
from Product.DataInterface import Datainterface


class SnowDepth(object):
    """积雪深度类，类方法包括省积雪深度计算，道路积雪深度指数"""
    def __init__(self, new_lat, new_lon):
        """
        初始化参数
        :param dics:计算积雪深度所需要素
        """
        cpath = r'/home/cqkj/QHTraffic/Product/Source/snowconfig.xml'
        self.m1path, self.m2path, self.savepath, self.roadpath, self.indexpath = Writefile.readxml(cpath, 1)
        self.dics = Writefile.readxml(cpath, 2)[0].split(',')
        #############################################################


    def clcsd(self,dataset, lat, lon, salte, snowdepth):
        """积雪深度计算，path为模型路径,新增卫星数据"""
        dataset.resize(56, 801*1381, 4)
        snowdepth  = snowdepth.reshape(-1, 1)
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
                snowdept = prediction                                #结果作为下一次预测的输入 
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
                snowdept = prediction                                #结果作为下一次预测的输入 
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
        if type == 0 :
            Writefile.write_to_nc(self.savepath,data, lat, lon, name,fnames, filetime)
        else :
            Writefile.write_to_csv(self.indexpath, data, 'SnowIndex',fnames,filetime)


def saltedata(path):
    # path 为卫星数据目录
    pattern = dt.datetime.now().strftime('%Y%m%d')
    strings = os.listdir(path).sort()
    tifname =regex(pattern, strings)
    
    if len(tifname) == 0:
        print('MODIS tiff don\'t exist, call model2!')
        return None
    else:
        print('call model1!(include MODIS)')
        gdal.AllRegister()
        dataset = gdal.Open(tifname)
        rtf = ReadTiff(dataset)
        px,py = rtf.imagexy2pro()
        pro2geov = np.vectorize(rtf.pro2geo)                         
        lon, lat = pro2geov(px, py)                                                # 此处执行很慢，循环操作
        newlat =np.linspace(31.4,39.4,801)
        newlon =np.linspace(89.3,103.1,1381)
        *_, newdata = rtf.equalatlon(ncname, dataset.ReadAsArray(), lat, lon, newlat, newlon)
        return newdata


def main():
    saltedata = saltedata(path)
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


if __name__ == '__main__':
    main()