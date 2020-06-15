import pickle
import numpy as np
import xarray as xr
import datetime
import os
from collections import deque
import glovar
from Product.DataInterface import Datainterface
from Product.Dataprocess import Znwg, Writefile, ecmwf, interp


def snowdepth(path):
    # 获取到青海积雪厚度网格产品预报(积雪厚度网格预报为003， 006， 009)需处理为逐日
    now = Znwg.znwgtime().strftime('%Y%m%d%H')
    pattern = r'(' + now + '.*?.nc)'
    strings = os.listdir(path)
    namelist = sorted(Znwg.regex(pattern, strings))
    os.chdir(r'/home/cqkj/QHTraffic/traffic')
    datasets = xr.open_mfdataset(namelist, concat_dim='time')

    data = datasets.SnowDepth.values
    newdata = []
    for i in range(0, 7):
        #print(i * 8, (i + 1) * 8)
        tmp = np.mean(data[i * 8:(i + 1) * 8], axis=0)
        #print(tmp.shape)
        newdata.append(tmp)
    newdata.extend(newdata[-3:])
    newdata = np.array(newdata)
    newdata = np.piecewise(newdata, [newdata<0.1, newdata>=0.1], [1, 0])
    return newdata


def landtype(greenpath, forestpath):
    # 是否需要处理土地类型数据（林地草地叠加）
    green = xr.open_dataset(greenpath).green.values
    forest = xr.open_dataset(forestpath).forgest.values
    return green, forest


def liverain(path, pklpath):
    # 获取青海的ZNWG实况降水数据并存储为pickle
    # 注意事项：目前生成的数据保留原始分辨率
    elements, _, localdir, historydir, freq, *ftp = Writefile.readxml(path, 1)
    now = datetime.datetime.now()
    ytd = now - datetime.timedelta(days=1)
    dir = r'/SCMOC/BEXN'
    remote_url = os.path.join(dir, ytd.strftime('%Y'), ytd.strftime('%Y%m%d'))
    grb = Datainterface.GribData()
    grb.mirror('ER24', remote_url, localdir, '24024', ftp)  # 同步昨天数据完成
    rainpath = sorted(os.listdir(localdir))[-1]
    os.chdir(localdir)
    rainlive, lat, lon, res = Znwg.arrange([grb.readGrib(rainpath, nlat=glovar.latt, nlon=glovar.lonn)][0])
    ####以上为获取当日青海数据
    with open(pklpath, 'rb') as f:
        data = pickle.load(f)
    data.append(rainlive)
    # 写入deque数据
    with open(pklpath, 'wb') as f:
        pickle.dump(data, f)
    return rainlive


def Weatherdata(path):
    # 获取森林火险所需气象数据
    elements, subdirs, localdir, _, freq, *ftp = Writefile.readxml(path, 1)
    now = datetime.datetime.now()
    elements = elements.split(',')
    subdirs = subdirs.split(',')
    remote_urls = [os.path.join(subdir, now.strftime('%Y'), now.strftime('%Y%m%d')) for subdir in subdirs]  # 待构造

    grib = Datainterface.GribData()
    '''
    [grib.mirror(element, remote_url, localdir, freq, ftp) for element, remote_url in
     zip(elements[:-1], remote_urls[:-1])]  # 同时同步大风、相对湿度、气温要素数据(24003)
     '''
    for element, remote_url in zip(elements[:-1], remote_urls[:-1]):
        grib.mirror(element, remote_url, localdir, freq, ftp)

    grib.mirror(elements[-1], remote_urls[-1], localdir, '24024', ftp)  # 同步出降水要素
    # 此处应改为提取出不同要素列表，目前简单实现，构造四个pattern
    strings = ','.join(os.listdir(localdir))
    patterns = [r'(\w+.EDA.*?.GRB2)', r'(\w+.ERH.*?.GRB2)', r'(\w+.TMP.*?.GRB2)', r'(\w+.ER24.*?.GRB2)']
    allpath = [localdir + sorted(Znwg.regex(pattern, strings), key=str.lower)[-1] for pattern in patterns] # allpath应为四个最新同步的文件列表
    ele14list = slice(1, 74, 8)  # （+2-1）前三个要素未来10天每天14时数据索引
    ####第一个要素wind包含u风和v风
    wind = grib.readGrib(allpath[0])[0]
    windu_v = np.array([v for _, v in wind.items()])
    windu, windv  = windu_v[::2][ele14list], windu_v[1::2][ele14list]
    data = np.array([Znwg.arrange(grib.readGrib(path))[0][ele14list] for path in allpath[1:-1]])  # 读取出前三项数据信息
    #er, lat, lon, size = Znwg.arrange(grib.readGrib(allpath[-1], nlat=glovar.lat, nlon=glovar.lon))  # 降水为国家级资料，先查看经纬度信息是否与前三者一致
    er, lat, lon, size = Znwg.arrange([grib.readGrib(allpath[-1], nlat=glovar.latt, nlon=glovar.lonn)][0])
    result = windu, windv, data, er  # 最终数据应为[4,10,181,277]矩阵
    return result, lat, lon


def firelevel(data, path, snow, landtype):
    """
    计算草原火险等级
    :param data: 气象要素外的其它条件
    :return: greenfirelevel
    """
    ############################################################
    # 采用双端队列存储实况降水数据数据量为八天左右，存储为pickle
    with open(path, 'rb') as f:
        predq = pickle.load(f)
    preres = np.argmax(np.array(predq)[::-1], axis=0)
    tem = np.add.reduce(predq)
    preres[tem == 0] = 8                       # 八天干旱情况
    ##############################################################
    # 计算连续无降水日数代码：假定已完成数据编码处理（有降水 0， 无降水1）
    *eda, erh, tmp, er = data                   # eda为U、V风速，需根据UV风速求得风速此处tmp为开氏温度，需转化成摄氏温度
    refertest = []
    for i in range(len(er)):
        if i == 0:
            test = np.piecewise(er[i], [er[i] < 0.1, er[i] >= 0.1], [1, 0])  # 结果为连续无降水日数
            # np.piecewise(test, [test == 0, test > 0], [0, test+preres])
            test = np.where(test>0, test+preres, 0)
        else:
            test = np.argmax(er[:i + 1][::-1], axis=0)
            refer = np.add.reduce(er[:i + 1])
            test[refer == 0] = i + 1
            # np.piecewise(test, [test < i+1, test >= i+1], [test, lambda x:x+preres])
            test = np.where(test>=i+1, test+preres, test)
        refertest.append(test)
    #############################################################
    # 此处根据火险气象因子查对应火险指数
    eda = np.sqrt(eda[0]**2 + eda[1]**2)                                 # 根据uv风求得风速大小
    edaindex = np.piecewise(eda, [(0<=eda)&(eda<1.6), (eda>=1.6)&(eda<3.5), (eda>=3.5)&
                                  (eda<5.6), (eda>=5.6)&(eda<8.1), (eda>=8.1)&(eda<10.9),
                                  (eda >=10.9)&(eda<14),(eda>14)&(eda<=17.2), eda>17.2],
                            [3.846, 7.692, 11.538, 15.382, 19.236, 23.076, 26.923, 30.9])

    tmp -= 273.15                                    # 转变为摄氏温度
    tmpindex = np.piecewise(tmp, [tmp<5, (tmp>=5)&(tmp<11), (tmp>=11)&
                                  (tmp<16), (tmp>=16)&(tmp<21), (tmp>=21)&(tmp<=25),
                                  tmp>25], [0, 4.61, 6.1, 9.23, 12.5, 15.384])

    erhindex = np.piecewise(erh, [erh>70, (erh>=60)&(erh<=70), (erh>=50)&
                                  (erh<60), (erh>=40)&(erh<50), (erh>=30)&(erh<=40),
                                  erh<30], [0, 3.076, 6.153, 9.23, 12.307, 15.384])
    refertest = np.array(refertest)
    mindex = np.piecewise(refertest, [refertest==0, refertest==1, refertest==2, refertest==3,
                                      refertest==4, refertest==5, refertest==6, refertest==7,
                                      refertest>=8], [0, 7.692, 11.538, 19.23, 23.076, 26.923,
                                                      30.7, 34.615, 38])
    u = edaindex + tmpindex + erhindex + mindex
    ###################################################################################
    # 订正部分计算（需要积雪深度矩阵、降水矩阵、地表状况矩阵, 先不使用积雪深度矩阵订正）
    rain = np.piecewise(er, [er<0.1, er>=0.1], [0, 1])
    rain = [interp.interpolateGridData(r, glovar.latt, glovar.lonn, glovar.lat, glovar.lon) for r in rain]
    rain = np.nan_to_num(np.array(rain))

    u = [interp.interpolateGridData(u_, glovar.latt, glovar.lonn, glovar.lat, glovar.lon) for u_ in u]
    u = np.nan_to_num(np.array(u))
    green = u*landtype[0]*rain*snow                                 #  草原火险产品
    forest = u*landtype[1]*rain*snow                                #  森林火险产品
    ###################################################################################
    # 进行森林火险气象等级划分
    gindex = np.piecewise(green, [green<=25, (green>25)&(green<51), (green>=51)&(green<73), (green>=73)&(green<91), green>=91], [1,2,3,4,5])
    findex = np.piecewise(forest, [forest <= 25, (forest > 25) & (forest < 51), (forest >= 51) & (forest < 73),
                                  (forest >= 73) & (forest < 91), forest >= 91], [1, 2, 3, 4, 5])
    mindex = np.maximum(gindex, findex)

    return gindex, findex, mindex


def main():
    snowpath, gpath, fpath, rainpath, savepath = Writefile.readxml(glovar.forestpath, 0)                       # 积雪nc数据存放位置
    snow = snowdepth(snowpath)           # 积雪数据[10, 801, 1381]
    data, *_ = Weatherdata(glovar.forestpath)   # 森林火险数据
    ldtype = landtype(gpath, fpath)
    gindex, findex, mindex = firelevel(data, rainpath, snow, ldtype)  # 最终生成指数
    filetime = ecmwf.ecreptime()
    fh = range(10)
    fnames = ['_%03d' % i for i in fh]
    Writefile.write_to_nc(savepath, gindex, filetime=filetime, fnames=fnames, lat=glovar.lat, lon=glovar.lon,
                          name='green')

    Writefile.write_to_nc(savepath, findex, filetime=filetime, fnames=fnames, lat=glovar.lat, lon=glovar.lon,
                          name='forest')

    Writefile.write_to_nc(savepath, mindex, filetime=filetime, fnames=fnames, lat=glovar.lat, lon=glovar.lon,
                          name='meteo')
if __name__ == '__main__':
    main()




