import cv2
import numpy as np

# Load the image
imagePath = 'img/semaforo9.jpg'

img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
colorImage = cv2.imread(imagePath)

# Apply Fourier transform to the image
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Define a high-pass filter to enhance edges
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

# Apply inverse Fourier transform to get the filtered image
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

# Normalize the filtered image to 0-255 range
iimg = cv2.normalize(iimg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Apply thresholding to convert the image to binary
ret, thresh = cv2.threshold(iimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
img_contours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)

semaphore_contours = []
for contour in contours:
    # obtém o retângulo mínimo que contém o contorno
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area > 523 and (h/w > 1.5 or w/h > 1.5):  # define os limites de área dos semáforos
        cv2.drawContours(colorImage, [contour], 0, (0, 0, 255), 2)
        semaphore_contours.append(contour)

        # Find the circles inside the rectangle
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        circle_contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for circle_contour in circle_contours:
            (x, y), radius = cv2.minEnclosingCircle(circle_contour)
            center = (int(x), int(y))
            radius = int(radius)
            # Check if the circle is entirely inside the rectangle
            if x - radius > contour[0][0][0] and x + radius < contour[0][2][0] \
                    and y - radius > contour[0][0][1] and y + radius < contour[0][2][1]:
                cv2.circle(colorImage, center, radius, (255, 0, 0), 2)

# Display the results
cv2.imshow('Original', colorImage)
#cv2.imshow('Filtered', iimg)
#cv2.imshow('Binary', thresh)
#cv2.imshow('Contours', img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
