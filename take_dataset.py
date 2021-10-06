import os

import cv2


def check_if_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

user_id = input('enter your id ')
check_if_path_exists("C:\\Users\\Rasel\\Desktop\\Attendance Project\\dataset")
cap = cv2.VideoCapture(0)
sample_taken = 0
sample_to_take = 10

cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detections = cascade_classifier.detectMultiScale(gray,1.3,5)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        sample_taken+= 1
        cv2.imwrite("C:\\Users\\Rasel\\Desktop\\Attendance Project\\dataset\\" + str(user_id) + '.' + str(sample_taken) + ".jpg", frame)
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        cv2.putText(frame, "Successfully taken " + str(sample_taken) + " Samples", (x, y - 10), font, 0.55, (120, 255, 120), 1)  
    cv2.imshow('My First Window to check image', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sample_taken >=sample_to_take:
        break

cap.release()
cv2.destroyAllWindows()
