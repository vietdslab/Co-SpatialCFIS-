import os
import cv2
def validImg(fileName):
    validFile=['jpg','png']
    if(fileName[-3:] in validFile):
        return True
    return False

def splitImg(folder):
    cnt = 0
    numFile = len(os.listdir(folder)) - 1
    check = True
    for fileName in sorted(os.listdir(folder)):
        if(validImg(fileName) and check):
            cnt+=1
            img = cv2.imread(os.path.join(folder,fileName),0)
            os.rename(os.path.join(folder,fileName), os.path.join('Data',os.path.join('Training Data',fileName)))
            cv2.imwrite(os.path.join(folder,fileName),img)
        if (validImg(fileName) and check == False ):
            img = cv2.imread(os.path.join(folder,fileName),0)
            os.rename(os.path.join(folder,fileName), os.path.join('Data',os.path.join('Testing',fileName)))
            cv2.imwrite(os.path.join(folder,fileName),img)
        if(cnt>=0.7*numFile):
            check = False
if __name__ == "__main__":
    folder = 'All Data'
    splitImg(folder)