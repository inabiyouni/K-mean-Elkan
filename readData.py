import numpy as np

def readData(fileName, stoppingPnt = None):
    inputfile = open(fileName)

    line = inputfile.readline()
    strData = np.array([line.split()]).astype(float)
    list = np.empty((0,strData.size), float)
    list = np.append(list, strData, axis=0)

    cnt = 0
    for line in inputfile:
        strData = np.array([line.split()]).astype(float)
        list = np.append(list, strData, axis=0)
        cnt += 1
        if stoppingPnt != None and cnt >= stoppingPnt - 1: break
    return list