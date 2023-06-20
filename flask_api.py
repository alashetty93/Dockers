# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:14:12 2023

@author: Ashvini Alashetty
"""

import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import sobel
from scipy import ndimage as nd
import pandas as pd


img = cv2.imread('C:/Users/Ashvini Alashetty/Downloads/archive/COVID-19_Radiography_Dataset/COVID/images/COVID-990.png')
img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Pixels Values'] = img2

entropy_img = entropy(img, disk(1))
entropy1 = entropy_img.reshape(-1)
df['Entropy'] = entropy1



gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian1 = entropy_img.reshape(-1)
df['gaussian'] = gaussian1

sobel_img = sobel(img)
sobel1 = entropy_img.reshape(-1)
df['sobel'] = sobel1


print(df)
cv2.imshow('Original Image', img)
cv2.imshow('sobel', sobel_img)
cv2.waitKey()
cv2.destroyAllWindows()