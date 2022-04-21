import os
import numpy as np
import cv2
import math
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
    res = np.empty((0,size))
    for fileName in sorted(os.listdir(folder)):
        if(validTxt(fileName)):
            hod = np.loadtxt(os.path.join(folder,fileName),dtype=int, delimiter=',')
            hod = hod.reshape(1,-1).astype(int)
            res = np.vstack((res,hod))
    return res

def setOfRule(U, V, ids, ImgTraining, HodTraining):
    tmp_rel_b=[V[:,0]]
    tmp_img_b=[V[:,1]]
    tmp_rel_a=np.zeros([1,V.shape[0]])
    tmp_img_a=np.zeros([1,V.shape[0]])
    tmp_rel_c=np.zeros([1,V.shape[0]])
    tmp_img_c=np.zeros([1,V.shape[0]])
    for j in range(V.shape[0]):
        sumU_a_rel=0
        sumU_a_img=0
        sumU_c_rel=0
        sumU_c_img=0
        sumUI_a_rel=0
        sumUI_a_img=0
        sumUI_c_rel=0
        sumUI_c_img=0
        for items in range(ImgTraining.shape[1]):
            if(ImgTraining[ids][items]<=tmp_rel_b[0][j]):
                sumU_a_rel+=U[items][j]
                sumUI_a_rel+=U[items][j]*ImgTraining[ids][items]
            if(HodTraining[ids][items]<=tmp_img_b[0][j]):
                sumU_a_img+=U[items][j]
                sumUI_a_img+=U[items][j]*HodTraining[ids][items]
            if(ImgTraining[ids][items]>=tmp_rel_b[0][j]):
                sumU_c_rel+=U[items][j]
                sumUI_c_rel+=U[items][j]*ImgTraining[ids][items]
            if(HodTraining[ids][items]>=tmp_img_b[0][j]):
                sumU_c_img+=U[items][j]
                sumUI_c_img+=U[items][j]*HodTraining[ids][items]
        tmp_rel_a[0][j]=sumUI_a_rel/sumU_a_rel
        tmp_img_a[0][j]=sumUI_a_img/sumU_a_img
        tmp_rel_c[0][j]=sumUI_c_rel/sumU_c_rel
        tmp_img_c[0][j]=sumUI_c_img/sumU_c_img
    return tmp_rel_a, tmp_rel_b, tmp_rel_c, tmp_img_a, tmp_img_b, tmp_img_c

def readUAndV(folderU, folderV, ImgTraining, HodTraining, c):
    ids=0
    rel_a=np.empty((0,c))
    img_a=np.empty((0,c))
    rel_b=np.empty((0,c))
    img_b=np.empty((0,c))
    rel_c=np.empty((0,c))
    img_c=np.empty((0,c))
    for fileName in sorted(os.listdir(folderV)):
        if validTxt(fileName):
            V=np.loadtxt(os.path.join(folderV,fileName),delimiter=',')
            fileNameU='U'+fileName[1:]
            U=np.loadtxt(os.path.join(folderU,fileNameU),delimiter=',')
            tmp_rel_a,tmp_rel_b,tmp_rel_c,tmp_img_a,tmp_img_b,tmp_img_c=setOfRule(U,V,ids,ImgTraining,HodTraining)
            ids+=1
            rel_a=np.vstack((rel_a,tmp_rel_a))
            rel_b=np.vstack((rel_b,tmp_rel_b))
            rel_c=np.vstack((rel_c,tmp_rel_c))
            img_a=np.vstack((img_a,tmp_img_a))
            img_b=np.vstack((img_b,tmp_img_b))
            img_c=np.vstack((img_c,tmp_img_c))
    return rel_a, rel_b, rel_c, img_a, img_b, img_c
if __name__ == "__main__":
    with open('config.xml','r') as f:
        data = f.read()
    Bs_data = Bs(data, 'xml')
    c = int(Bs_data.c.contents[0])
    folderImg = os.path.join('Data',os.path.join('Training Data','gray'))
    folderHod = os.path.join('Data',os.path.join('Training Data','hod'))
    HodTraining = readHodImg(folderHod)/255
    ImgTraining = readDataImg(folderImg)/255
    folderU = 'U'
    folderV = 'V'
    rel_a, rel_b, rel_c, img_a, img_b, img_c=readUAndV(folderU,folderV,ImgTraining,HodTraining,c)
    #final_rel_a=np.empty((0,1))
    #final_img_a=np.empty((0,1))
    #final_rel_b=np.empty((0,1))
    #final_img_b=np.empty((0,1))
    #final_rel_c=np.empty((0,1))
    #final_img_c=np.empty((0,1))
    #check=np.ones([rel_a.shape[0],rel_a.shape[1]])
    #for i in range(rel_a.shape[0]):
    #    for j in range(rel_a.shape[1]):
    #        if(check[i][j]==1):
    #            final_rel_a=np.vstack((final_rel_a,rel_a[i][j]))
    #            final_img_a=np.vstack((final_img_a,img_a[j][j]))
    #            final_rel_b=np.vstack((final_rel_b,rel_b[i][j]))
    #            final_img_b=np.vstack((final_img_b,img_b[i][j]))
    #            final_rel_c=np.vstack((final_rel_c,rel_c[i][j]))
    #            final_img_c=np.vstack((final_img_c,img_c[i][j]))
    np.savetxt(os.path.join('Rel','rel_a.txt'),rel_a,fmt="%.4f",delimiter=',')
    np.savetxt(os.path.join('Img','img_a.txt'),img_a,fmt="%.4f",delimiter=',')
    np.savetxt(os.path.join('Rel','rel_b.txt'),rel_b,fmt="%.4f",delimiter=',')
    np.savetxt(os.path.join('Img','img_b.txt'),img_b,fmt="%.4f",delimiter=',')
    np.savetxt(os.path.join('Rel','rel_c.txt'),rel_c,fmt="%.4f",delimiter=',')
    np.savetxt(os.path.join('Img','img_c.txt'),img_c,fmt="%.4f",delimiter=',')