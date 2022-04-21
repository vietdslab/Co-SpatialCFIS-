import cv2
import numpy as np
import os

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

if __name__ == "__main__":
    testFolder = 'Test PCO'
    predictFolder = 'Final Res Spatial CFIS/Result30'
    file = predictFolder[-8:]
    f = open("RMSE.txt", "a")
    #f1 = open("R2.txt","a")
    print(file)
    res_R2 = []
    res_RMSE = []
    for items in sorted(os.listdir(predictFolder)):
        print(items)
        #if items.find('8') == True or items.find('9') == True or items.find('10') == True:
        testImg = cv2.imread(os.path.join(testFolder,items),0)
        predictImg = cv2.imread(os.path.join(predictFolder,items),0)
        #print(predictImg)
        #print('color_'+items)
        mean = np.mean(testImg)
        sstot = np.sum((testImg - mean)**2)
        ssres = np.sum((testImg-predictImg)**2)
        res_R2 += [1-(ssres)/(sstot)]
        res_RMSE += [np.round(rmse(testImg, predictImg),3)]
        print(1-(ssres)/(sstot))
        print('RMSE', rmse(testImg, predictImg))
    s1 = str(sorted(res_RMSE)).replace("[", "").replace("]", "")
    #s2 = str(sorted(res_R2)).replace("[", "").replace("]", "")
    f.write(s1 + "\n")
    #f1.write(s2 + "\n")
    f.close()
    #f1.close()
    np.savetxt(os.path.join("R2 Result", file + ".txt"), sorted(res_R2), fmt = "%.3f")
    np.savetxt(os.path.join("RMSE Result", file + ".txt"), sorted(res_RMSE), fmt = "%.3f")
    #testImg = cv2.imread(os.path.join(testFolder,'color_rs_8.jpg'),0)
    #predictImg = cv2.imread(os.path.join(predictFolder,'color_rs_8.jpg'),0)
    #mean = np.mean(testImg)
    #sstot = np.sum((testImg - mean)**2)
    #ssres = np.sum((testImg-predictImg)**2)
    #print(1-(ssres)/(sstot))
