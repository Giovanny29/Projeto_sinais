import cv2
import numpy as np
from keras.models import load_model

# carrega a imagem
# model = load_model('keras_model.h5',compile=False)
# data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)
# classes = ["0 semaforo","1 nuvem","2 arvore"]

# def DetectarSemeforo(img):
#     imgMoeda = cv2.resize(img,(224,224))
#     imgMoeda = np.asarray(imgMoeda)
#     imgMoedaNormalize = (imgMoeda.astype(np.float32)/127.0)-1
#     data[0] = imgMoedaNormalize
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     percent = prediction[0][index]
#     classe = classes[index]
#     return classe,percent
def filtroAntiGlimmering(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] > 128):
                img[i][j] = max(128, img[i][j]*0.85)
            else:
                img[i][j] = min(128, img[i][j]*1.15)
    return img

img = cv2.imread('img\semaforo4.jpg')
img = filtroAntiGlimmering(img)
# aplica a transformada de Fourier à imagem
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# exibe a imagem após a transformada de Fourier
#magnitude_spectrum = 20 * np.log(np.abs(fshift))

# define o filtro passa-alta para realçar a região de alta frequência da imagem
rows, cols, channels = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

# aplica a transformada inversa de Fourier à imagem filtrada
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = filtroAntiGlimmering(gray)

# aplica a detecção de bordas usando o filtro de Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_8U)

# aplica a detecção de contornos para encontrar os semáforos na imagem
contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# aplica a detecção de contornos para encontrar os semáforos na imagem
contours, hierarchy = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
semaphore_contours = []
for contour in contours:
    # obtém o retângulo mínimo que contém o contorno
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area > 523 :  # define os limites de área dos semáforos
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        semaphore_contours.append(contour)


# exibe a imagem com os retângulos detectados
while True:   
  cv2.imshow('IMG',img)
  cv2.imshow('Imagem após filtro de laplace',laplacian )
  cv2.waitKey(1)