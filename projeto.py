import cv2
import numpy as np

# Definir a nova largura e altura da imagem
new_width = 1600
new_height = 800

# Carregar a imagem, redimensionar e converter para escala de cinza
img = cv2.imread('img/semaforo1.jpg')
rsd_img = cv2.resize(img, (new_width, new_height))
gray = cv2.cvtColor(rsd_img, cv2.COLOR_BGR2GRAY)

# Aplicar filtro passa alta
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
gray_filtered = cv2.filter2D(gray, -1, kernel)

# Aplicar a Transformada de Fourier
img_fft = np.fft.fft2(gray_filtered)

# Centralizar as frequências baixas
img_fft_shifted = np.fft.fftshift(img_fft)

# Aplicar um filtro passa-baixa no domínio da frequência para remover o ruído
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
r = 50
cv2.circle(mask, (ccol, crow), r, 255, -1)
img_fft_shifted_filtered = img_fft_shifted * mask

# Centralizar novamente as frequências baixas
img_fft_filtered = np.fft.ifftshift(img_fft_shifted_filtered)

# Aplicar a Transformada Inversa de Fourier
img_final = np.abs(np.fft.ifft2(img_fft_filtered))

# Normalizar a imagem resultante para o intervalo [0, 255]
img_final = cv2.normalize(img_final, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Aplicar filtro de Sobel na direção x e y
sobelx = cv2.Sobel(img_final , cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_final , cv2.CV_64F, 0, 1, ksize=3)

# Combinar os resultados
sobel = np.sqrt(sobelx**2 + sobely**2)

# Converter para escala de cinza
sobel = np.uint8(sobel)
    
# Aplicar o detector de bordas Canny
edges = cv2.Canny(sobel, 100, 200)

# Exibir a imagem original e a imagem filtrada
cv2.imshow('Original', img)
cv2.imshow('Final', img_final)
cv2.imshow('Sobel', sobel)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
