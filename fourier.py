import cv2
import numpy as np

# Leitura da imagem
img = cv2.imread('img/semaforo3.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicação do filtro de Gauss
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Cálculo do gradiente com o operador Sobel
grad_x = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)

# Cálculo da magnitude e direção do gradiente
mag, ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

# Aplicação do limiar OTSU na magnitude do gradiente
mag_8bit = cv2.convertScaleAbs(mag)
_, thresh = cv2.threshold(mag_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Cálculo da transformada de Fourier
dft = cv2.dft(np.float32(img_gaussian), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Filtro passa-alta com a transformada de Fourier
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols, 2), np.uint8)
r = 50
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
mask[mask_area] = 0

# Aplicação do filtro passa-alta na transformada de Fourier
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Exibição das imagens
cv2.imshow('Original', img)
cv2.imshow('Gradiente', mag_8bit)
cv2.imshow('Limiar', thresh)
cv2.imshow('Passa-alta', img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()
