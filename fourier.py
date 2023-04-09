import numpy as np
import cv2 
from PIL import Image

# Carrega a imagem
img = Image.open("img\semaforo1.jpg")

# Converte a imagem para escala de cinza
img_gray = img.convert("L")

# Converte a imagem para um array NumPy
img_array = np.array(img_gray)

# Aplica a FFT na imagem
img_fft = np.fft.fft2(img_array)

# Aplica o shift nas frequências para o centro da imagem
img_fft_shift = np.fft.fftshift(img_fft)

# Cria um filtro passa-baixas
filtro = np.zeros(img_array.shape)
altura, largura = img_array.shape
centro_altura, centro_largura = altura // 2, largura // 2
raio = 50
for i in range(altura):
    for j in range(largura):
        if ((i - centro_altura)**2 + (j - centro_largura)**2) < raio**2:
            filtro[i, j] = 1

# Aplica o filtro na imagem
img_fft_filt = img_fft_shift * filtro

# Aplica o shift inverso nas frequências
img_fft_filt_shift = np.fft.ifftshift(img_fft_filt)

# Aplica a inversa da FFT na imagem filtrada
img_filtrada = np.abs(np.fft.ifft2(img_fft_filt_shift))

# Salva a imagem filtrada
Image.fromarray(img_filtrada.astype(np.uint8)).save("img/semafaro1_fft.jpg")
# Aplica o operador Sobel
sobelx = cv2.Sobel(img_filtrada, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_filtrada, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = np.sqrt(sobelx**2 + sobely**2)

# Normaliza a imagem
img_sobel = cv2.normalize(img_sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Aplica a limiarização
img_filtrada = cv2.convertScaleAbs(img_filtrada)
_, img_limiarizada = cv2.threshold(img_filtrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Salva a imagem limiarizada
cv2.imwrite("img/semafaro1_limiarizada.jpg", img_limiarizada)


# Salva a imagem com as bordas detectadas
cv2.imwrite("img/imagem_bordas_sobel.png", img_sobel)