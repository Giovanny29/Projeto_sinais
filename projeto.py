import cv2
import numpy as np

# Definir a nova largura e altura da imagem
new_width = 1600
new_height = 800

# Carregar a imagem, redimensionar e converter para escala de cinza
img = cv2.imread('img/semaforo1.jpg')
rsd_img = cv2.resize(img, (new_width, new_height))
gray = cv2.cvtColor(rsd_img, cv2.COLOR_BGR2GRAY)

# Aplicar filtro de Gauss
img_gauss = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplicar filtro de Sobel na direção x e y
sobelx = cv2.Sobel(img_gauss , cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gauss , cv2.CV_64F, 0, 1, ksize=3)

# Combinar os resultados
sobel = np.sqrt(sobelx**2 + sobely**2)

# Converter para escala de cinza
sobel = np.uint8(sobel)
    
# Aplicar o detector de bordas Canny
edges = cv2.Canny(sobel, 100, 200)

# Exibir a imagem original e a imagem filtrada
cv2.imshow('Original', img)
cv2.imshow('Gauss', img_gauss)
cv2.imshow('Sobel', sobel)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
