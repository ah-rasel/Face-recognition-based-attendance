import cv2
import os
import numpy as nump
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
def check_if_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    count = 0
    for imagePath in imagePaths:
        count+=1
        pilImage=Image.open(imagePath).convert('L')
        imageNp=nump.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[0])
        # print(os.path.split(imagePath)[-1].split(".")[0] + " - " + str(count))
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('C:\\Users\\Rasel\\Desktop\\Attendance Project\\dataset')
s = recognizer.train(faces, nump.array(Ids))
print(faces)
print("Successfully trained")
check_if_path_exists("C:\\Users\\Rasel\\Desktop\\Attendance Project\\train\\train.yml")
recognizer.write('C:\\Users\\Rasel\\Desktop\\Attendance Project\\train\\train.yml')