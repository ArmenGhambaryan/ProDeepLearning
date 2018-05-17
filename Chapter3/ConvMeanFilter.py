import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import numpy as np

np.random.seed(0)
img = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure('Original image')
plt.imshow(gray,cmap='gray')
mean = 0
var = 100
sigma = var**0.5
row, col = gray.shape
gauss = np.random.normal(mean,sigma,(row,col))
gray = gray.reshape(row,col)
gray_noisy = gray + gauss
plt.figure('Gaussian filter')
plt.imshow(gray_noisy,cmap='gray')
## Mean Filter 
Hm = np.ones((3,3))/float(9)
Gm = convolve2d(gray_noisy,Hm,mode='same')
plt.figure('Mean filter')
plt.imshow(Gm,cmap='gray')

## Median filter
gray_sp = gray*1
sp_indices = np.random.randint(0,21,[row,col])
for i in range(row):
	for j in range(col):
		if sp_indices[i,j]==0:
			gray_sp[i,j] = 0
		if sp_indices[i,j]==20:
			gray_sp[i,j] = 255
plt.figure('Salt and Pepper image')
plt.imshow(gray_sp,cmap='gray')

gray_sp_removed = cv2.medianBlur(gray_sp,3)
plt.figure('Median filter')
plt.imshow(gray_sp_removed,cmap='gray')


## Gaussian filter
Hg = np.zeros((20,20))
for i in range(20):
	for j in range(20):
		Hg[i,j] = np.exp(-((i-10)**2 + (j-10)**2)/10)
plt.figure('Gaussian IRF')
plt.imshow(Hg,cmap='gray')
gray_blur = convolve2d(gray, Hg, mode='same')
plt.figure('Gaussian filter')
plt.imshow(gray_blur,cmap='gray')
gray_high = gray - gray_blur
gray_enhanced = gray + 0.025*gray_high
plt.figure('Sharper image')
plt.imshow(gray_enhanced,cmap='gray')
plt.show()



