import os
import cv2
path = "Data"
pathSave = "MyData"
count = 0
for i in range(0, 2):
    os.mkdir(pathSave +"/"+str(i))
myList = os.listdir(path)
for i in range (0, len(myList)):
    z=0
    folder = os.listdir(path+"/"+str(count))
    for j in  folder:
        CurImg = cv2.imread(path +"/"+str(count)+"/"+j)
        Img = cv2.resize(CurImg,(56,56))
        cv2.imwrite(pathSave +"/"+str(count)+"/"+ str(z)+".jpg", Img)
        z += 1
    count+=1