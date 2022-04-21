import os
import numpy as np
import cv2
import sys
import math
import creat_HOD
from bs4 import BeautifulSoup as Bs
def validImg(fileName):
    validFile=['jpg','png']
    if(fileName[-3:] in validFile):
        return True
    return False

def getImgSize(folder):
    for fileName in os.listdir(folder):
        if(validImg(fileName)):
            img=cv2.imread(os.path.join(folder,fileName),0)
            break
    return img.shape[0],img.shape[1]
def readDataImg(folder):
    width, height = getImgSize(folder)
    res = np.empty((0,width*height))
    for fileName in sorted(os.listdir(folder)):
        if(validImg(fileName)):
            img = cv2.imread(os.path.join(folder,fileName),0)
            img = img.flatten().reshape(1,-1).astype(int)
            res = np.vstack((res,img))
    return res

def validTxt(fileName):
    if(fileName[-3:]=='txt'):
        return True
    return False

def getHodSize(folder):
    for fileName in os.listdir(folder):
        if(validTxt(fileName)):
            hod = np.loadtxt(os.path.join(folder,fileName), dtype=int, delimiter=',')
            hod = hod.reshape(1,-1)
            return hod.shape[0]*hod.shape[1]

def readHodImg(folder):
    size = getHodSize(folder)
    #print(size)
    res = np.empty((0,size))
    for fileName in sorted(os.listdir(folder)):
        if(validTxt(fileName)):
            hod = np.loadtxt(os.path.join(folder,fileName),dtype=int, delimiter=',')
            hod = hod.reshape(1,-1).astype(int)
            res = np.vstack((res,hod))
    return res

def FCM(folderU, folderV, ImgTraining, HodTraining, t, c, m, eps):
    #print(c)
    for i in range(HodTraining.shape[0]):
        V = np.zeros([t+2, c, 2])
        U = np.zeros([HodTraining.shape[1], c])
        rel_min = np.amin(ImgTraining[i])
        rel_max = np.amax(ImgTraining[i])
        img_min = np.amin(HodTraining[i])
        img_max = np.amax(HodTraining[i])
        for j in range(c):
            rel = np.random.uniform(rel_min, rel_max)
            img = np.random.uniform(img_min, img_max)
            V[0][j][0] = rel
            V[0][j][1] = img
        steps = 0
        diff = 1
        #print(i)
        while(steps <= t and diff > eps):
            print("steps: ",steps)
            steps += 1
            diff = 0
            for k in range(HodTraining.shape[1]):
                for j in range(c):
                    s = 0
                    for id1 in range(c):
                        s+=(math.sqrt(((ImgTraining[i][k]-V[steps-1][j][0])*(ImgTraining[i][k]-V[steps-1][j][0])+(HodTraining[i][k]-V[steps-1][j][1])*(HodTraining[i][k]-V[steps-1][j][1]))/((ImgTraining[i][k]-V[steps-1][id1][0])*(ImgTraining[i][k]-V[steps-1][id1][0])+(HodTraining[i][k]-V[steps-1][id1][1])*(HodTraining[i][k]-V[steps-1][id1][1]))))**(2/(m-1))
                    U[k][j]=1.0/s
            for j in range(c):
                sumU = 0
                s_rel = 0
                s_img = 0
                for k in range(HodTraining.shape[1]):
                    sumU = sumU+U[k][j]**m
                    s_rel=s_rel+(U[k][j]**m)*ImgTraining[i][k]
                    s_img=s_img+(U[k][j]**m)*HodTraining[i][k]
                V[steps][j][0]=s_rel/sumU
                V[steps][j][1]=s_img/sumU
                diff=diff+(V[steps][j][0]-V[steps-1][j][0])*(V[steps][j][0]-V[steps-1][j][0])+(V[steps][j][1]-V[steps-1][j][1])*(V[steps][j][1]-V[steps-1][j][1])
            diff=math.sqrt(diff)
            fileNameU=os.path.join(folderU,"U of image "+str(i)+".txt")
            np.savetxt(fileNameU, U, fmt="%.5f", delimiter=',')
            fileNameV=os.path.join(folderV,"V of image "+str(i)+".txt")
            np.savetxt(fileNameV, V[steps], fmt="%.5f", delimiter=',')

if __name__ == "__main__":
    # must insert progress bar
    with open('config.xml','r') as f:
        data=f.read()
    Bs_data=Bs(data, "xml")
    t = int(Bs_data.FCM.t.contents[0])
    c = int(Bs_data.FCM.c.contents[0])
    m = int(Bs_data.FCM.m.contents[0])
    eps = float(Bs_data.FCM.eps.contents[0])
    folderImg = os.path.join('Data',os.path.join('Training Data','gray'))
    folderHod = os.path.join('Data',os.path.join('Training Data','hod'))
    HodTraining = readHodImg(folderHod)/255
    ImgTraining = readDataImg(folderImg)/255
    FCM(ImgTraining, HodTraining, t, c, m, eps)