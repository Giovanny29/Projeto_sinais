import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

img = cv2.imread('img/semaforo8.jpg')

def Pre_processing(img): 
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    imgPre = cv2.Canny(imgPre,50,100)
    kernel = np.ones((2,2),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre



    #_,img = video.read()
img = cv2.resize(img,(640,480))
imgPre = Pre_processing(img)
countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for cnt in countors:
      area = cv2.contourArea(cnt)
      if area > 2000 :
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
while True:   
  cv2.imshow('IMG',img)
  cv2.imshow('IMG PRE', imgPre)
  cv2.waitKey(1)
