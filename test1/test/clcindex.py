#  此函数用来计算实况指数,先完成初步逻辑代码
import numpy as np
import pick1e
from Model import FloodModel


def readdbs():
    pass


def floodsk(predata, data):
    # 计算前一时刻积水深度, data为sk降水， predata为实况前一时刻降水, 返回各点数据
    dr = np.zeros_like(predata)  # predata前一时刻降水
    pre = FloodModel.cal2(predata[np.newaxis, ], dr)
    datas = FloodModel.cal2(data[np.newaxis, ], pre)
    return datas


def windcross(road, data):
    # 计算该站横风指数，road提供各站道路走向数据(道路走向存成新文件)，data提供风向风速数据
    length = np.linalg.norm(road.iloc[:, -2:], axis=1)
    ur = np.divide(road['ur'], length)
    vr = np.divide(road['vr'], length)
    uv = [[i, j] for i, j in zip(ur, vr)]
    datas = np.cross(uv, data)
    return datas


def roadicsk(data):
    # 计算道路结冰厚度实况，data为降水、积雪和地表温度数据
    model = r'test.pkl'
    with open(model, 'rb') as f:
        md = pickle.load(f)
    temp = [ds.reshape(-1, 1) for ds in data]
    prediction = np.array(md.predict(temp))
    return prediction


def icing(data):
    # 计算出结冰指数实况， data为道路结冰厚度实况数据，还需计算连续结冰天数
    pass


