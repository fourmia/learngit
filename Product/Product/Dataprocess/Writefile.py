import xml.etree.cElementTree as ET
import pymysql
import pandas as pd
import xarray as xr
import numpy as np


def readxml(path, num):
    tree = ET.parse(path)
    root = tree.getroot()
    settings = [i.text for i in root[int(num)]]
    return settings


def readbs():
    host,uid,pwd,dbname,port,table_name = readxml()[1]
    # 有密码版本
    # conn = pymysql.connect(host=host, port=int(port), user=uid,
    #                   passwd=pwd, database=dbname)
    # 无密码版本
    conn = pymysql.connect(host=host, port=int(port), user=uid,
                       database=dbname)
    df = pd.read_sql('SELECT * FROM ' + table_name, conn)
    return df


def write_to_nc(path, data, lat, lon, name, fnames, filetime):
    cirnum = data.shape[0]
    print(cirnum)
    for i in range(cirnum):
        ds = xr.Dataset({name: (['lat', 'lon'], data[i])}, coords={'lat': lat, 'lon': lon})
        f = ''.join([path, name, filetime, fnames[i], '.nc'])
        ds.to_netcdf(f, format='NETCDF3_CLASSIC')


def write_to_csv(path, data, name, fnames, filetime):
    cirnum = len(data)
    for i in range(cirnum):
        allpath = path + name + '_' + filetime + fnames[i] + '.csv'
        np.savetxt(allpath, data[i], delimiter=',', fmt='%d')