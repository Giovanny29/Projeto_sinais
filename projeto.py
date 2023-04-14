import cv2
import numpy as np

# carrega a imagem
img = cv2.imread('img\semaforo1.jpg')
img_resized = cv2.resize(img, (266, 266))

# aplica a transformada de Fourier à imagem
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# exibe a imagem após a transformada de Fourier
magnitude_spectrum = 20 * np.log(np.abs(fshift))


# define o filtro passa-alta para realçar a região de alta frequência da imagem
rows, cols, channels = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

# aplica a transformada inversa de Fourier à imagem filtrada
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    if area > 300 :  # define os limites de área dos semáforos
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        semaphore_contours.append(contour)

# exibe a imagem com os retângulos detectados
while True:   
  cv2.imshow('IMG',img)
  cv2.imshow('Imagem após filtro de laplace',laplacian )
  cv2.waitKey(1)
