# 基本要素实况
import pymysql
import sqlalchemy
from sqlalchemy import create_engine


class baseStdlive(object):
    """
    这个类用于解决基础站点实况值，包括降水、风速、地最高温、地最低温四类
    """
    def __init__(self, dataframe):
        self.df = dataframe


    def datetime(self):
        # 返回实例时间
        return sorted(self.time.unique())

    def rain