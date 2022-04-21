import cv2
import numpy as np
import os
import math

def validImg(fileName):
    validFile=['jpg','png']
    if(fileName[-3:] in validFile):
        return True
    return False

def Accuracy(testImgFolder, predictImgFolder):
    result = []
    for fileName in sorted(os.listdir(testImgFolder)):
        if validImg(fileName): 
            testImg = cv2.imread(os.path.join(testImgFolder, fileName),0)
            predictImg = cv2.imread(os.path.join(predictImgFolder, fileName),0)
            result+=[math.sqrt(np.sum(np.square(testImg - predictImg))/(2*testImg.shape[0]*testImg.shape[1]))]
    return result

if __name__ == "__main__":
    testImgFolder = os.path.join('Data','Testing')
    predictImgFolder = 'Result'
    result = Accuracy(testImgFolder, predictImgFolder)