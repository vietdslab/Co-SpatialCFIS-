import creat_HOD
import FCM
import MakeRule
import TrainingAndPredict
import RMSE
import SplitData
import os
import numpy as np
import cv2
import time
from bs4 import BeautifulSoup as Bs
import argparse
def Spatial_CFIS(configFile, DataFolder):
    #ap = argparse.ArgumentParser()
    # ap.add_argument("-c", "--config", required = True, help = "path to config")
    # ap.add_argument("-d", "--data", required = True, help= "path to Data")
    # ap.add_argument("-o", "--output", required = True, help = "path to predict image")
    # ap.add_argument("-a", "--accuracy", required = True, help = "path to accuracy folder") 
    # #ap.add_argument("-id","--userID", required = True, help = "User ID")
    # args = vars(ap.parse_args())
    try:   
        with open(configFile,'r') as f:
            data=f.read()
        Bs_data=Bs(data, "xml")
    except:
        print("Please choose *.xml or format config xml file same as tutorial")
    t = int(Bs_data.FCM.t.contents[0])
    c = int(Bs_data.FCM.c.contents[0])
    m = int(Bs_data.FCM.m.contents[0])
    eps = float(Bs_data.FCM.eps.contents[0])
    lr = float(Bs_data.Adam.LearningRate.contents[0])
    iterations = int(Bs_data.Adam.iteration.contents[0])
    threshold = float(Bs_data.Adam.threshold.contents[0])
    momentum = float(Bs_data.Adam.momentum.contents[0])
    beta1 = float(Bs_data.Adam.beta1.contents[0])
    beta2 = float(Bs_data.Adam.beta2.contents[0])
    all_data_folder = DataFolder
    SplitData.splitImg(all_data_folder)
    training_data_folder = os.path.join('Data','Training Data')
    creat_HOD.readImage(training_data_folder)
    start_time = time.time()
    folderImg = os.path.join('Data',os.path.join('Training Data','gray'))
    folderHod = os.path.join('Data',os.path.join('Training Data','hod'))
    HodTraining = FCM.readHodImg(folderHod)/255
    ImgTraining = FCM.readDataImg(folderImg)/255
    print("hod: ",HodTraining.shape)
    for iter in range(0,30):
        print('iter:  ',iter) 
        oPath = "Result" + str(iter)
        try:  
            outputPath = os.mkdir(oPath)  
        except OSError as error:  
            print(error)
        try:    
            folderU = 'U'+ str(iter)
            folderV = 'V' + str(iter)
            os.mkdir(folderU)
            os.mkdir(folderV)
        except OSError as error:  
            print(error)
        FCM.FCM(folderU, folderV, ImgTraining, HodTraining, t, c, m, eps)
        rel_a, rel_b, rel_c, img_a, img_b, img_c=MakeRule.readUAndV(folderU,folderV,ImgTraining,HodTraining,c)
        np.savetxt(os.path.join('Rel','rel_a.txt'),rel_a,fmt="%.4f",delimiter=',')
        np.savetxt(os.path.join('Img','img_a.txt'),img_a,fmt="%.4f",delimiter=',')
        np.savetxt(os.path.join('Rel','rel_b.txt'),rel_b,fmt="%.4f",delimiter=',')
        np.savetxt(os.path.join('Img','img_b.txt'),img_b,fmt="%.4f",delimiter=',')
        np.savetxt(os.path.join('Rel','rel_c.txt'),rel_c,fmt="%.4f",delimiter=',')
        np.savetxt(os.path.join('Img','img_c.txt'),img_c,fmt="%.4f",delimiter=',')
        path_rel_a = os.path.join('Rel','rel_a.txt')
        path_img_a = os.path.join('Img','img_a.txt')
        path_rel_b = os.path.join('Rel','rel_b.txt')
        path_img_b = os.path.join('Img','img_b.txt')
        path_rel_c = os.path.join('Rel','rel_c.txt')
        path_img_c = os.path.join('Img','img_c.txt')
        rel_rule,img_rule = TrainingAndPredict.readRule(path_rel_a, path_rel_b, path_rel_c, path_img_a, path_img_b, path_img_c)
        last_image_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','gray'))))[-1]
        if(not FCM.validImg(last_image_name)):
            last_image_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','gray'))))[-2]
        last_image_path = os.path.join('Data',os.path.join('Training Data',os.path.join('gray',last_image_name)))
        last_hod_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','hod'))))[-1]
        if(not FCM.validTxt(last_hod_name)):
            last_hod_name = sorted(os.listdir(os.path.join('Data',os.path.join('Training Data','hod'))))[-2]
        last_hod_path = os.path.join('Data',os.path.join('Training Data',os.path.join('hod',last_hod_name)))
        U, width, height=TrainingAndPredict.calU(rel_rule, img_rule, last_image_path, last_hod_path)
        np.savetxt("U_all_rule.txt",U,fmt='%.4f',delimiter=',')
        alpha = np.zeros((U.shape[0], 1))
        beta = np.zeros((U.shape[0] + 1, 1))
        h = np.zeros((U.shape[0]*3, 1))
        gamma = np.zeros((1, 1))
        s=np.ones([U.shape[0],1])
        for i in range(U.shape[0]):
            if(np.sum(U[i]!=0)):
                s[i]=np.sum(U[i])
        theta1 = np.ones(3)
        first_image_testing = sorted(os.listdir(os.path.join('Data','Testing')))[0]
        if(not FCM.validImg(first_image_testing)):
            first_image_testing = sorted(os.listdir(os.path.join('Data','Testing')))[1]
        first_image_testing_path = os.path.join('Data',os.path.join('Testing',first_image_testing))
        y=cv2.imread(first_image_testing_path,0) #first image of testing
        y=y.flatten()
        y=y.reshape(-1,1)

        w = TrainingAndPredict.FWAdam_img(U,rel_rule, alpha, beta, h, gamma, y, theta1, s, lr, iterations, threshold, momentum, beta1, beta2, last_image_path)
        for fileName in sorted(os.listdir(os.path.join('Data','Testing'))):
            if(FCM.validImg(fileName)):
                rel_W=np.asarray([[1,2,1]])
                img_W=w.reshape(1,3)
                rel_def,img_def=TrainingAndPredict.calDEF(rel_W, img_W, rel_rule, img_rule)
                O_rel,O_img=TrainingAndPredict.calO(rel_def,img_def,U)
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
                cv2.imwrite(os.path.join(oPath,fileName),O_img)
                #cv2.imwrite('ver1_'+fileName,O_img)
        result = RMSE.Accuracy(os.path.join('Data','Testing'), oPath)
        np.savetxt("./RMSE"+ str(iter), result, fmt = '%.4f', delimiter = ',')
        end_time = time.time()
        #print(end_time - start_time)

#def main():
if __name__ == '__main__':
    
    Spatial_CFIS('config.xml', './Data')    


