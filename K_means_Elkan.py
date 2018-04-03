import numpy as np
import copy
#from dataInfo import dataInfo
#from createNode import createNode
#import xml.etree.cElementTree as ET
from readData import readData
from dataInfo_Elkan import dataInfo_Elkan
from sys import argv, exit
from time import sleep
import os.path
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

def getDistOfPnts(subMat):
    dict = {}
    maxRep = 0
    for key in subMat:
        if key in dict:
            dict[key] += 1
            if dict[key] > maxRep: maxRep = dict[key]
        else:
            dict[key] = 1
            if maxRep == 0: maxRep = 1
    return dict, maxRep


def initial_Cntrds(inst, max_min):
    inst.c = np.mean(inst.x[:, :-1], axis=0).reshape(1, inst.k)
    next_mean = np.empty((1, inst.k))
    for j in range(inst.K - 1):
        max_Dist_to_Centrs = - np.inf
        rnd_Draw = np.random.randint(inst.N, size=inst.N)
        for i in rnd_Draw:
            dis = (inst.x[i, :-1]).reshape(1, inst.k) - inst.c
            dis_to_Cntrs = np.linalg.norm(dis, axis=1)
            min_dist = dis_to_Cntrs
            if dis_to_Cntrs.size > 1:
                dis_to_Cntrs.sort()
                min_dist = dis_to_Cntrs[0]
            if min_dist > max_Dist_to_Centrs:
                max_Dist_to_Centrs = min_dist
                next_mean = (inst.x[i, :-1]).reshape(1, inst.k)
        inst.c = np.append(inst.c, next_mean, axis=0)
    #return (np.random.rand(inst.K, inst.k) * (max_min[0,:] - max_min[1,:])) + max_min[1,:]


def normalizeData(inst):
    for i in range(0,len(inst.x[0,:])-1):
        attr = inst.x[:,i].astype(float)
        try:
            inst.x[:,i] = (attr) / np.std(attr)
        except Exception as e:
            print (e)

def cal_Distance(x, y, disType, p, inst):
    inst.dist_Cnt += 1
    if (disType == "eucli"):
        dist = ((np.power(abs(x - y), p)).sum()) ** (1.0 / p)
    elif (disType == 'fun1'):
        indx = x > y
        dist = (np.power((x[indx] - y[indx]).sum(), p) + np.power(
            (y[np.invert(indx)] - x[np.invert(indx)]).sum(), p)) ** (1.0 / p)
    elif (disType == 'fun2'):
        indx = x > y
        dist = (np.power((x[indx] - y[indx]).sum(), p) + np.power(
            (y[np.invert(indx)] - x[np.invert(indx)]).sum(), p)) ** (1.0 / p)
        temp = np.absolute(x - y).reshape(1, inst.k)
        temp = np.append(temp, np.absolute(x).reshape(1, inst.k), axis=0)
        temp = np.append(temp, np.absolute(y).reshape(1, inst.k), axis=0)
        denom = np.amax(temp, axis=0)
        dist = dist / denom.sum()
    elif (disType == 'cosine'):
        magX = abs((np.power(x, 2)).sum()) ** (1.0 / 2)
        magY = abs((np.power(y, 2)).sum()) ** (1.0 / 2)
        dist = 1 - sum(x[:] * y[:]) / magX / magY
    return dist

def calCentroids(inst, distFun, p):
    for j in range(0, inst.K):
        samData = inst.x[inst.a == j, :-1]
        if samData != []:
            samp_Mean = np.mean(samData, axis=0)
            inst.dlta[j] = cal_Distance(inst.c[j, :], samp_Mean, distFun, p, inst)
            inst.c[j, :] = samp_Mean

def updateBounds(inst):
    for i in range(0, inst.N):
        inst.u[i] += inst.dlta[inst.a[i]]
        for j in range(0, inst.K):
            inst.l[i, j] = np.amax([0, inst.l[i, j] - inst.dlta[j]])

def comput_S(inst, distFun, p):
    c_prox_mat = np.ones((inst.K,inst.K), float) * np.inf
    for j in range(0, inst.K - 1):
        for jper in range(j + 1, inst.K):
            dist = cal_Distance(inst.c[j, :], inst.c[jper, :], distFun, p, inst) / 2.0
            inst.cntrs_Dis[j, jper] = dist
            inst.cntrs_Dis[jper, j] = dist
            if c_prox_mat[j, jper] > dist:
                c_prox_mat[j, jper] = dist
                c_prox_mat[jper, j] = dist
    inst.s = np.amin(c_prox_mat, axis=1)

def updateGraph(fig, x, y, diffType, p = None):
    fig.set_xdata(x)#np.append(fig.get_xdata(), x))
    fig.set_ydata(y)#np.append(fig.get_ydata(), y))
    fig.set_label('%s %d' %(diffType, p)) if p != None else fig.set_label('%s' %(diffType))
    plt.draw()
    plt.pause(0.3)#1e-17)
    sleep(0.01)

def findMAxMin(inst):
    max_min = np.array(np.max(inst.x[:, :-1], axis=0)).reshape(1,inst.k)
    max_min = np.append(max_min, (np.min(inst.x[:, :-1], axis=0)).reshape(1,inst.k), axis=0)
    return max_min

def assignPnts(inst, distFun, p):
    cnt_Pnt_Updt = 0
    for i in range(0, inst.N):
        if inst.u[i] <= inst.s[inst.a[i]]:
            continue
        r = True
        for j in range(0, inst.K):
            z = np.amax([inst.l[i, j], inst.cntrs_Dis[inst.a[i], j]])
            if j == inst.a[i] or inst.u[i] <= z:
                continue
            if r:
                inst.u[i] = cal_Distance(inst.x[i,:-1], inst.c[inst.a[i], :], distFun, p, inst)
                r = False
                if inst.u[i] <= z:
                    continue
            inst.l[i, j] = cal_Distance(inst.x[i,:-1], inst.c[j, :], distFun, p, inst)
            if inst.l[i, j] < inst.u[i]:
                inst.a[i] = j
                cnt_Pnt_Updt += 1
    return cnt_Pnt_Updt

def K_means_Elkan(inst, distFun, epsl, hard_Stop, p):
    if (inst.clsDist == None):
        inst.clsDist, _ = getDistOfPnts(inst.x[:,-1])
        if (inst.K == 0): inst.K = len(inst.clsDist)
        #normalizeData(inst)
        inst.k = len(inst.x[0,:-1])
        max_min = findMAxMin(inst)
        if animate:
            axes.set_xlim(max_min[1,0]-1, max_min[0,0]+1)
            axes.set_ylim(max_min[1,1]-1, max_min[0,1]+1)  # 10000)
        inst.N = len(inst.x[:, 0])
        initial_Cntrds(inst, max_min)
        inst.a = np.zeros(inst.N, int)
        inst.u = np.ones(inst.N, float) * np.inf
        inst.l = np.zeros((inst.N, inst.K), float)
        inst.dlta = np.zeros(inst.K, float)
        inst.cntrs_Dis = np.ones((inst.K, inst.K), float)
        inst.itert = 0
        inst.dist_Cnt = 0
        print("iter | number of distance cal")
    print ('\r  {}  |     {:,}'.format(inst.itert, inst.dist_Cnt), end=""),
    if inst.itert  == hard_Stop:
        print("\nHard stop at the iteration number!")
        return
    inst.itert += 1
    comput_S(inst, distFun, p)
    cnt_Pnt_Updt = assignPnts(inst, distFun, p)
    if animate:
        updateGraph(figN1, inst.c[:,0], inst.c[:,1], "Centroids")
        for j in range(0, inst.K):
            samData = inst.x[inst.a == j, :-1]
            updateGraph(eval('figN%d'% (j + 2)), samData[:,0], samData[:,1], "Cluster", j)
            #lif j == 1: updateGraph(figN3, samData[:,0], samData[:,1], "Cluster", j)
    calCentroids(inst, distFun, p)
    updateBounds(inst)
    if (cnt_Pnt_Updt == 0):
        return
    K_means_Elkan(inst, distFun, epsl, hard_Stop, p)
    return

def calAccuracy(inst, clustrNo):
    purity = 0
    realClasses = inst.data[:,-1]
    newClasses = inst.clusters[:, 1]
    for j in range(0, clustrNo):
        cluster = realClasses[newClasses == j]
        _, majority = getDistOfPnts(cluster)
        purity += majority
    print("\nTotal purity comparing the real classes are:\n%0.2f" %(purity / len(inst.data[:, 0])))

#main body of the program for loading data and
# running a recursive tree function to learn from data

#arguments setting
args = iter(argv)
next(args)
clustrNo = 3
n= 10000
k = 3#1001
p = 2
epsl = 0.1
hard_Stop = 34
animate = False#True#
if animate:
    #plot setting
    plt.show()
    plt.xlabel('k')
    plt.ylabel('r')
    plt.title('Clustering by K-means')
    axes = plt.gca()
    figN2, = axes.plot([], [], 'gx', markersize=4)  # marker='x', linestyle='--', color='g')
    figN3, = axes.plot([], [], 'b*', markersize=4)  # marker='*', linestyle='--', color='b')
    figN4, = axes.plot([], [], 'c+', markersize=4)  # marker='*', linestyle='--', color='b')
    figN5, = axes.plot([], [], 'k-', markersize=4)  # marker='*', linestyle='--', color='b')
    figN6, = axes.plot([], [], 'yo', markersize=4)  # marker='*', linestyle='--', color='b')
    figN7, = axes.plot([], [], 'mx', markersize=4)  # marker='*', linestyle='--', color='b')
    figN1, = axes.plot([], [], 'r+', markersize=20)
    ax = plt.subplot(111)
    pos1 = ax.get_position() # get the original position
    pos2 = [pos1.x0 + 0.03, pos1.y0 + 0.02,  pos1.width / 1.0, pos1.height / 1.0]
    ax.set_position(pos2) # set a new position


inputFileTr = ""
inputClassesTr = ""
inputFileTs = ""
inputClassesTs = ""
distFun = "eucli"
for item in args:
    if item == "-K":
        clustrNo = int(next(args))
    elif item == "-itr1":
        inputFileTr = next(args)
    elif item == "-itr2":
        inputClassesTr = next(args)
    elif item == "-its1":
        inputFileTs = next(args)
    elif item == "-its2":
        inputClassesTs = next(args)
    elif item == "-i":
        hard_Stop = int(next(args))

if inputFileTr.lower() == "" or inputClassesTr == "":#or inputFileTs.lower() == "" or inputClassesTs.lower() == "" or
    print("You have NOT entered correct inputs!")
    exit()
if distFun.lower() is ("city"):
    p = 1
elif (distFun.lower() in ("fun1", "fun2", "cosine", "eucli", "")):
    p = 2

#inputFileTr = "X_cup.txt"#iris.txt"#X_data1_rand_2_150.txt"#X_train.txt"#banknote.txt"#
#inputClassesTr = "y_cup.txt"#iris.txt"#data1_class_1_150.txt"#y_train.txt"#banknote.txt"#
#inputFileTs = "X_data2_rand_2_150.txt"#"X_test.txt"#"
#inputClassesTs = "y_data2_class_1_150.txt"#"y_test.txt"#

if inputClassesTr != "":
    print("\nReading 1 set of data ...")
    clases = readData(inputClassesTr)
    data1 = readData(inputFileTr)
    data1 = np.append(data1, clases, axis=1)

if inputClassesTs != "":
    print("\nReading 2 set of data ...")
    clases = readData(inputClassesTs)
    data2 = readData(inputFileTs)
    data2 = np.append(data2, clases, axis=1)

if inputClassesTr != "":
    data = np.append(data1, data2, axis=0) if (inputClassesTs != "") else data1

if (distFun == ""): distFun = "eucli"

#print("\nRunning K-means algorithm ...")
#inst = dataInfo_Elkan(data)
#inst.K = clustrNo
#K_means_Elkan(inst, distFun.lower(), epsl, p)

result_file = open('K-mean-Elkan_%s_%d_%s.txt' %("rand", p, distFun),'a+')
#for clustrNo in [3, 20, 100]:
result_file.write('\n%sp = %d n = %d k = %d\n' % (distFun, p, n, clustrNo))
result_file.write("iters    num of dist cal\n")

for rept in range(6):
    #if (os.path.isfile("data.npy")):
    #    data = np.load("data.npy")
    #else:
    #    print("Generating new random data ...")
    #    data = np.random.random((n, k))
    #    np.save("data", data)
    if (distFun == ""): distFun = "eucli"

    print("\nRunning K-means_Elkan algorithm ...")
    inst = dataInfo_Elkan(data)
    inst.K = clustrNo
    K_means_Elkan(inst, distFun.lower(), epsl, hard_Stop, p)
    result_file.write("{} {:,}\n".format(inst.itert, inst.dist_Cnt))
    result_file.flush()

result_file.close()

print ("\n")
print("iterations: %d\nnumber of distance cal: %d" % (inst.itert, inst.dist_Cnt))
#calAccuracy(inst, clustrNo)
if animate:
    plt.legend()
    plt.savefig('result.png', bbox_inches='tight')
    plt.show()
#print("\nThe K-means algorithm has clustered the data,\nYou can look at it in a file named \"trained_Tree.xml\"\nwhich is created inside your project folder")
