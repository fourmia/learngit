import pickle
import torch
import os
import pandas as pd
import numpy as np
from Model.lenet import LeNet
import glovar
from Dataprocess import interp, Znwg, Writefile, dataformat
from DataInterface import Datainterface

#from Product import glovar
#from Product.Dataprocess import interp, Znwg, Writefile, dataformat
#from Product.DataInterface import Datainterface


def windData(path):
    # 获取数据信息
    *_, elements, ftp = Writefile.readxml(path, 2)
    element = elements.split(',')
    ftp = ftp.split(',')
    grib = Datainterface.GribData()
    remote_url = os.path.join(r'\\SPCC\\BEXN', glovar.now.strftime('%Y'), glovar.now.strftime('%Y%m%d'))
    grib.mirror(element[0], remote_url, element[1], element[2], ftp)
    rname = sorted(os.listdir(element[1]))[-1]
    rpath = element[1] + rname
    dataset, lat, lon, _ = Znwg.arrange((grib.readGrib(rpath)))                       # result包含data,lat,lon,size
    return np.nan_to_num([interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon) for data in dataset])      # 返回插值后列表格式数据


def revise(path, message):
    # 订正解析结果，并插值到1km分辨率
    mdpath, _, gcpath, savepath, indexpath, *_ = Writefile.readxml(path, 2)
    data = [message[::2, :, :][:56], message[1::2, :, :][:56]]
    # data = [data[::2, :, :][:56], data[1::2, :, :][:56]]   # 分别取出来U风V风
    net = torch.load(mdpath)
    net.eval()
    dem = pd.read_csv(gcpath, index_col=0).values
    arrays = np.nan_to_num([np.array([i, j, dem]) for i, j in zip(data[0], data[1])])
    inputs = torch.from_numpy(arrays)
    torch.no_grad()
    outputs = [net(it[np.newaxis, :]).detach().numpy() for it in inputs]
    output = np.squeeze(np.nan_to_num(outputs))
    datau, datav = output[:, 0], output[:, 1]
    Writefile.write_to_nc(savepath, datau, glovar.lat, glovar.lon, 'U', glovar.fnames, glovar.filetime)
    Writefile.write_to_nc(savepath, datav, glovar.lat, glovar.lon, 'V', glovar.fnames, glovar.filetime)
    return indexpath, datau, datav


def press(savepath, windpath,datau, datav):
    # 计算出道路点风压大小
    mes = pd.read_csv(windpath, index_col=0)
    lon, lat = mes['Lon'], mes['Lat']  # 提取出道路坐标
    length = np.linalg.norm(mes.iloc[:, -2:], axis=1)
    ur = np.divide(mes['ur'], length)
    vr = np.divide(mes['vr'], length)
    uv = [[i, j] for i, j in zip(ur, vr)]
    for u, v in zip(datau, datav):
        uvalue = interp.interpolateGridData(u, glovar.latgrid, glovar.longrid, lat, lon, isGrid=False)
        vvalue = interp.interpolateGridData(v, glovar.latgrid, glovar.longrid, lat, lon, isGrid=False)
        UVvalue = [[i, j] for i, j in zip(uvalue, vvalue)]
        windcross = []
        for i,value in enumerate(UVvalue):
            uvalue = interp.interpolateGridData(u, glovar.latgrid, glovar.longrid, lat, lon, isGrid=False)
            vvalue = interp.interpolateGridData(v, glovar.latgrid, glovar.longrid, lat, lon, isGrid=False)
            datas = np.cross(uv, np.array(value).T)
            w = 1 / 2 * 1.29 * (np.square(datas)) * 1000
            w = np.piecewise(w, [w < 83, (w >= 83) & (w < 134), (w >= 134) & (w < 602), (w >= 602) & (w < 920), w >= 920],
                              [1, 2, 3, 4, 5])
            windcross.append(w[np.newaxis, :])
    Writefile.write_to_csv(savepath, windcross, 'wind', glovar.fnames, glovar.filetime)


def main():
    message = windData(glovar.trafficpath)
    savepath, u, v = revise(glovar.trafficpath, message)
    windpath = glovar.windpath        # 此处路径统一为服务器路径
    press(savepath, windpath, u, v)
    return None


if __name__ == '__main__':
    main()

