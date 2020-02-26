import math
import random
import numpy as np

class loader:
    def __init__(self,
                 dimensions = 2,
                 trainPercent: float = 90.0,
                 ):
        self.__tp = trainPercent
        self.__tr, self.__ts = self.__loadData(dimensions)

    def __loadData(self, dim):
        data = self.__get2DData() if dim==2 else self.__get3DData()
        self.__data = data.copy()
        ln = len(data)
        lnts = int(ln*(1-(self.__tp/100)))
        lntr = ln - lnts

        random.shuffle(data)

        tr = sorted(data[:lntr])
        ts = sorted(data[lntr:])
        return tr, ts

    def __get2DData(self):
        return [
            [[i / 10],
             [math.cos(i / 10) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-60, 61)
        ]

    def __get3DData(self):
        return [
            [[i/10, j/10],
             [math.sin(i/10)+math.cos(j/10)+random.random()*0.2-0.1]
            ]
            for i in range(-30, 31) for j in range(-30, 31)
        ]

    def getTrainInp(self):
        return np.array([i[0] for i in self.__tr])

    def getTrainOut(self):
        return np.array([i[1] for i in self.__tr])

    def getTestInp(self):
        return np.array([i[0] for i in self.__ts])

    def getTestOut(self):
        return np.array([i[1] for i in self.__ts])

    def getAllInp(self):
        return np.array([i[0] for i in self.__data])

    def getAllOut(self):
        return np.array([i[1] for i in self.__data])
