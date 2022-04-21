from random import triangular
import numpy as np
import os
import math
import cv2
from shapely.geometry import Point,LineString
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from bs4 import BeautifulSoup as BS
def validImg(fileName):
    validFile=['jpg','png']
    if(fileName[-3:] in validFile):
        return True
    return False

def validTxt(fileName):
    if(fileName[-3:]=='txt'):
        return True
    return False

def isPointInside(point, polygon):
    return polygon.contains(point)

def calculateIntersecPoint(x1,y1,x2,y2,x3,y3,x4,y4):
    px = ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py = ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    DE = math.sqrt((px-x1)**2+(py-y1)**2)
    BE = math.sqrt((px-x2)**2+(py-y2)**2)
    return DE/BE

def readRule(a, b, c, a1, b1, c1):
    rel_a = np.loadtxt(a,delimiter=',')
    img_a = np.loadtxt(a1,delimiter=',')
    rel_b = np.loadtxt(b,delimiter=',')
    img_b = np.loadtxt(b1,delimiter=',')
    rel_c = np.loadtxt(c,delimiter=',')
    img_c = np.loadtxt(c1,delimiter=',')
    rel_a = rel_a.reshape(1,-1)
    img_a = img_a.reshape(1,-1)
    rel_b = rel_b.reshape(1,-1)
    img_b = img_b.reshape(1,-1)
    rel_c = rel_c.reshape(1,-1)
    img_c = img_c.reshape(1,-1)
    rel_rule = np.empty((0,3))
    img_rule = np.empty((0,3))
    for i in range(rel_a.shape[1]):
        tmp1 = np.asarray([rel_a[0][i],rel_b[0][i],rel_c[0][i]])
        rel_rule = np.vstack((rel_rule,tmp1))
        img_rule = np.vstack((img_rule,[img_a[0][i],img_b[0][i],img_c[0][i]]))
    return rel_rule, img_rule

def findPolygon(pointX, pointY, rel_rule, img_rule,j):
    res = 0
    polygon1 = Polygon([(0,rel_rule[j][0]),(0,rel_rule[j][2]),(img_rule[j][1],rel_rule[j][1])])
    polygon2 = Polygon([(0,rel_rule[j][0]),(img_rule[j][0],0),(img_rule[j][1],rel_rule[j][1])])
    polygon3 = Polygon([(img_rule[j][0],0),(img_rule[j][2],0),(img_rule[j][1],rel_rule[j][1])])
    if(isPointInside(Point([pointX,pointY]),polygon1)):
        res = calculateIntersecPoint(pointX,pointY,img_rule[j][1],rel_rule[j][1],0,rel_rule[j][0],0,rel_rule[j][2])
    elif(isPointInside(Point([pointX,pointY]),polygon2)):
        res = calculateIntersecPoint(pointX,pointY,img_rule[j][1],rel_rule[j][1],0,rel_rule[j][0],img_rule[j][0],0)
    elif(isPointInside(Point([pointX,pointY]),polygon3)):
        res = calculateIntersecPoint(pointX,pointY,img_rule[j][1],rel_rule[j][1],img_rule[j][2],0,img_rule[j][0],0)
    return res

def findAlpha(outsidePoint, polygon):
    left = 1.0 
    right = 10.0
    eps = 0.001
    alpha = 5
    while(abs(left-right)>eps):
        mid = (left+right)/2
        check = True
        for i in outsidePoint:
            if(isPointInside(Point([i[0]/mid,i[1]/mid]),polygon)==False):
                check = False
        if(check):
            right = mid
            alpha = right
        else:
            left = mid
    return alpha 

def calU(rel_rule, img_rule, last_image_path, last_hod_path):
    img = cv2.imread(last_image_path,0) #validation image
    width=img.shape[0]
    height=img.shape[1]
    img = img.flatten()
    img = img.reshape(1,-1)/255
    hod = np.loadtxt(last_hod_path, delimiter=',')/255 # load last hod 
    hod=hod.reshape(1,-1)
    U=np.zeros([img.shape[1],rel_rule.shape[0]])
    for j in range(rel_rule.shape[0]):
        res=0
        alpha=5
        polygon=Polygon([(0,rel_rule[j][0]),(img_rule[j][0],0),(img_rule[j][2],0),(img_rule[j][1],rel_rule[j][1]),(0,rel_rule[j][2])])
        outsidePoint=np.empty((0,2))
        for i in range(img.shape[1]):
            point=Point(hod[0][i],img[0][i])
            if(isPointInside(point,polygon)):
                U[i][j]=findPolygon(hod[0][i],img[0][i],rel_rule,img_rule,j)
            if U[i][j]==0:
                outsidePoint=np.vstack((outsidePoint,[(hod[0][i],img[0][i])]))
        if outsidePoint is not None:
            alpha=findAlpha(outsidePoint,polygon)
        for i in range(img.shape[1]):
            point=Point(hod[0][i],img[0][i])
            if(isPointInside(point,polygon)==False):
                point=Point(hod[0][i]/alpha, img[0][i]/alpha)
                U[i][j]=findPolygon(hod[0][i]/alpha,img[0][i]/alpha,rel_rule,img_rule,j)
    return U, width, height

def calDEF(rel_W, img_W, rel_rule, img_rule):
    return np.dot(rel_W,np.transpose(rel_rule)),np.dot(img_W,np.transpose(img_rule))

def calO(rel_def,img_def,U):
    return np.dot(rel_def,np.transpose(U)),np.dot(img_def,np.transpose(U)) 

def find_fit_param(now, prev):
    corr_alpha = np.dot(now[0], prev[0])/(np.linalg.norm(now[0])*np.linalg.norm(prev[0]))
    corr_beta = np.dot(now[1], prev[1])/(np.linalg.norm(now[1])*np.linalg.norm(prev[1]))
    corr_h = np.dot(now[2], prev[2])/(np.linalg.norm(now[2])*np.linalg.norm(prev[2]))
    corr_gamma = np.dot(now[3], prev[3])/(np.linalg.norm(now[3])*np.linalg.norm(prev[3]))
    param = []
    if(corr_alpha > 0.65):
        param += [now[0]]
    if(corr_beta > 0.65):
        param += [now[1]]
    if(corr_h > 0.65):
        param += [now[2]]
    if(corr_gamma > 0.65):
        param += [now[3]]
    return param


def FWAdam_rel(U, rel_rule, alpha, beta, h, gamma, y, w, s, learning_rate = 0.01, iterations = 150, threshold=0.0001, momentum = 0.1, beta1 = 0.9, beta2 = 0.999):
    m = y.shape[0]
    error = 0
    mt = np.zeros(3)
    vt = np.zeros(3)
    gradient= w
    e = 0.00000001
    Training_param = np.array([alpha], [beta], [h], [gamma])
    # define a for loop
    for j in range(iterations):
        error = math.sqrt((1/(2*m))*np.sum(np.square(y-np.transpose((255*np.dot(np.dot(w,np.transpose(rel_rule)),np.transpose(U))/(w[0]+w[1]+w[2])))/(s))))
        # stop iteration
        if(j%150==0):
            learning_rate/=10
        i=np.random.randint(0,U.shape[0])
        if abs(error) <= threshold:
            break
        #gradient = x[j] * (np.dot(x[j], theta) - y[j])
        for item in Training_param:
            gradient[0] = (255*np.dot(U[i],(w[1]*(rel_rule[:,0]-rel_rule[:,1])+w[2]*(rel_rule[:,0]-rel_rule[:,2]))/((w[0]+w[1]+w[2])**2)))/s[i]
            gradient[1] = (255*np.dot(U[i],(w[0]*(rel_rule[:,1]-rel_rule[:,0])+w[2]*(rel_rule[:,1]-rel_rule[:,2]))/((w[0]+w[1]+w[2])**2)))/s[i]
            gradient[2] = (255*np.dot(U[i],(w[0]*(rel_rule[:,2]-rel_rule[:,0])+w[1]*(rel_rule[:,2]-rel_rule[:,1]))/((w[0]+w[1]+w[2])**2)))/s[i]
        mt = beta1 * mt + (1 - beta1) * gradient
        vt = beta2 * vt + (1 - beta2) * (gradient ** 2)
        mtt = mt / (1 - (beta1 ** (i + 1)))
        vtt = vt / (1 - (beta2 ** (i + 1)))
        vtt_sqrt = np.array([math.sqrt(vtt[0]), math.sqrt(vtt[1]), math.sqrt(vtt[2])])  # sqrt func only works for scalar
        w = w - learning_rate * mtt / (vtt_sqrt + e)
        Training_param = find_fit_param(w, Training_param)

def FWAdam_img(U, img_rule, alpha, beta, h, gamma, y, w, s, learning_rate , iterations , threshold, momentum , beta1 , beta2, validationImage):
    m = y.shape[0]
    error = 0
    mt = np.zeros(3)
    vt = np.zeros(3)
    gradient= w
    e = 0.00000001
    Training_param = np.array([alpha], [beta], [h], [gamma])
    #ValidationFolder=os.path.join('Data','Validation')
    #img = cv2.imread(os.path.join(ValidationFolder,os.listdir(ValidationFolder)[0]))
    img = cv2.imread(validationImage,0)
    img=img.flatten()
    img=img.reshape(1,-1)/255
    for i in range(iterations):
        error = math.sqrt((1/(2*m))*np.sum(np.square(y-np.transpose((np.multiply((1+255*np.dot(np.dot(w,np.transpose(img_rule)),np.transpose(U))),img)/(w[0]+w[1]+w[2])))/(s))))
        if abs(error) <= threshold:
            break
        if(sum(U[i])!=0):
            for item in Training_param:
                gradient[0] = (255*np.dot(U[i],(w[1]*(img_rule[:,0]-img_rule[:,1])+w[2]*(img_rule[:,0]-img_rule[:,2]))/((w[0]+w[1]+w[2])**2)))/s[i]
                gradient[1] = (255*np.dot(U[i],(w[0]*(img_rule[:,1]-img_rule[:,0])+w[2]*(img_rule[:,1]-img_rule[:,2]))/((w[0]+w[1]+w[2])**2)))/s[i]
                gradient[2] = (255*np.dot(U[i],(w[0]*(img_rule[:,2]-img_rule[:,0])+w[1]*(img_rule[:,2]-img_rule[:,1]))/((w[0]+w[1]+w[2])**2)))/s[i]
            mt = beta1 * mt + (1 - beta1) * gradient
            vt = beta2 * vt + (1 - beta2) * (gradient ** 2)
            mtt = mt / (1 - (beta1 ** (i + 1)))
            vtt = vt / (1 - (beta2 ** (i + 1)))
            vtt_sqrt = np.array([math.sqrt(vtt[0]), math.sqrt(vtt[1]), math.sqrt(vtt[2])])  # sqrt func only works for scalar
            w = w - learning_rate * mtt / (vtt_sqrt + e)
            Training_param = find_fit_param(w, Training_param)
    return w

if __name__ == "__main__":
    with open('config.xml', 'r') as f:
        data = f.read()
    Bs_data = BS(data, 'xml')
    lr = float(Bs_data.Adam.LearningRate.contents[0])
    iterations = int(Bs_data.Adam.iteration.contents[0])
    threshold = float(Bs_data.Adam.threshold.contents[0])
    momentum = float(Bs_data.Adam.momentum.contents[0])
    beta1 = float(Bs_data.Adam.beta1.contents[0])
    beta2 = float(Bs_data.Adam.beta2.contents[0])
    path_rel_a = os.path.join('Rel','rel_a.txt')
    path_img_a = os.path.join('Img','img_a.txt')
    path_rel_b = os.path.join('Rel','rel_b.txt')
    path_img_b = os.path.join('Img','img_b.txt')
    path_rel_c = os.path.join('Rel','rel_c.txt')
    path_img_c = os.path.join('Img','img_c.txt')
    rel_rule,img_rule=readRule(path_rel_a, path_rel_b, path_rel_c, path_img_a, path_img_b, path_img_c)
    last_image_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','gray'))))[-1]
    if(not validImg(last_image_name)):
        last_image_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','gray'))))[-2]
    last_image_path = os.path.join('Data',os.path.join('Training Data',os.path.join('gray',last_image_name)))
    last_hod_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','hod'))))[-1]
    if(not validTxt(last_hod_name)):
        last_hod_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','hod'))))[-2]
    last_hod_path = os.path.join('Data',os.path.join('Training Data',os.path.join('hod',last_hod_name)))
    U, width, height=calU(rel_rule, img_rule, last_image_path, last_hod_path)
    np.savetxt("U_all_rule.txt",U,fmt='%.4f',delimiter=',')
    s=np.ones([U.shape[0],1])
    for i in range(U.shape[0]):
        if(np.sum(U[i]!=0)):
            s[i]=np.sum(U[i])
    theta1 = np.ones(3)
    first_image_testing = sorted(os.listdir(os.path.join('Data','Testing')))[0]
    if(not validImg(first_image_testing)):
        first_image_testing = sorted(os.listdir(os.path.join('Data','Testing')))[1]
    first_image_testing_path = os.path.join('Data',os.path.join('Testing',first_image_testing))
    y=cv2.imread(first_image_testing_path,0) #first image of testing
    y=y.flatten()
    y=y.reshape(-1,1)
    w=Adam_img(U,rel_rule, y, theta1, s, lr, iterations, threshold, momentum, beta1, beta2, last_image_path)
    for fileName in sorted(os.listdir(os.path.join('Data','Testing'))):
        if(validImg(fileName)):
            rel_W=np.asarray([[1,2,1]])
            img_W=w.reshape(1,3)
            rel_def,img_def=calDEF(rel_W, img_W, rel_rule, img_rule)
            O_rel,O_img=calO(rel_def,img_def,U)
            O_rel=((O_rel/4)*255)
            O_img=((O_img/(np.sum(img_W)))*255)
        #img=cv2.imread('Hawaii_100x100_crop/Testing/rs_10.jpg',0)
            img = cv2.imread(os.path.join('Data',os.path.join('Testing',fileName)),0)
        #img=cv2.imread(os.path.join('Data',os.path.join('Testing',...)),0)
            img=img.flatten().reshape(1,-1)/255
            for i in range(U.shape[0]):
                if(np.sum(U[i]!=0)):
                    O_rel[0][i]=O_rel[0][i]/np.sum(U[i])
                    O_img[0][i]=O_img[0][i]/np.sum(U[i])
                    O_img[0][i]=img[0][i]*(1+O_img[0][i])
            O_img=O_img.reshape(width,height).astype(int)
            cv2.imwrite(os.path.join('Result',fileName),O_img)
            #cv2.imwrite('ver1_'+fileName,O_img)