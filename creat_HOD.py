import os
import cv2
import numpy as np

def valid(fileName):
    validImg=['jpg','png']
    if(fileName[-3:] in validImg):
        return True
    return False

def readImage(folder):
    items=0
    for fileName in sorted(os.listdir(folder)):
        if(valid(fileName)):
            tmp = cv2.imread(os.path.join(folder,fileName),0)
            cv2.imwrite(os.path.join(folder,os.path.join('gray',fileName)),tmp)
            #cv2.remove((os.path.join(folder,fileName)))
            tmp=tmp.flatten().reshape(1,-1)
            if(items >= 1):
                hod = abs(img-tmp)
                np.savetxt(os.path.join(folder,os.path.join('hod',fileName[:-3]+'txt')),hod, fmt='%d', delimiter=',')
            items+=1
            img=tmp
if __name__ == "__main__":
    folder = os.path.join('Data','Training Data')
    readImage(folder)