# 此程序用来完成预报入库，取每个批次的前四个文件
import numpy as np
import pandas as pd
import datetime as dt
from read_sql import Mysql
import interp, Write_to_csv, Writr_to_db

def readdb():
    pass


def todbs(data, numbs=4, i, point, glovar, name):
    # 当前代码段, i为0-56， data.shape(56,801,1381), point为道路点信息， glovar为站点信息
    # 生成nc、csv的同时进行预报值的入库操作
    for i in range(data.shape[0]):
        if i < numbs:
            # 对应到待检验站点上，有的话用in，没有则插值, name为df列名
            tmp = np.nan_to_num(interp.interpolateGridData(data[i], point.lat, point.lon, glovar.lat, glovar.lon)) # 采用插值方法，未想好站点对应
            time = glovar.time.strptime() + dt.timedelta(hours=i*3)                   # 获取时间并填充至整列
            df = pd.DataFrame(tmp, columns=name)
            dataframe = pd.concat([glovar.lat, glovar.lon, df, time])
            # 需执行入库操作 df.todbs(),更新并入库，防止错误覆盖
            Writr_to_db(dataframe)
        Write_to_csv(tmp)
    return None


def main():
    mysql = Mysql(host, port, user, pwd, db)







if __name__ == "__main__":
    data = np.random.randint(0, 2, size=(56, 801, 1381))
    todbs(data)
