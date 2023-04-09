import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

Width, Hight = 640, 480

img = cv2.imread('img/semaforo9.jpg')

def Pre_processing(img):
    img = cv2.resize(img, (Width , Hight))
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    imgPre = cv2.Canny(imgPre,90,140)
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    # Aplicar a FFT na imagem processada
    imgPre_fft = cv2.dft(np.float32(imgPre), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Aplicar um filtro passa-alta
    rows, cols = imgPre.shape
    crow, ccol = rows // 2, cols // 2
    filtro_pa = np.ones((rows, cols, 2), np.float32)
    filtro_pa[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
    imgPre_fft_filtrada = imgPre_fft * filtro_pa

    # Aplicar a IFFT na matriz de frequência filtrada
    imgPre_filtrada = cv2.idft(imgPre_fft_filtrada, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # Converter a imagem filtrada para o tipo uint8 e normalizar os valores de pixel
    imgPre_filtrada = cv2.normalize(imgPre_filtrada, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
  
    return imgPre



    #_,img = video.read()
img = cv2.resize(img,(640,480))
imgPre = Pre_processing(img)
countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for cnt in countors:
      area = cv2.contourArea(cnt)
      if area > 2000:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
while True:   
  cv2.imshow('IMG',img)
  cv2.imshow('IMG PRE', imgPre)
  cv2.waitKey(1)
