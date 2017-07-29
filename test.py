import numpy as np
from numpy import *
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import xlrd
import matplotlib.pyplot as plt

img = array(Image.open('true_value.png'))
img1 = []
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        img1.append(img[i, j])
def rgb2gray(R, G, B):
    return  3 * R + 6 * G + 1 * B
img2 = []
for i in range(img1.__len__()):
    img2.append(rgb2gray(img1[i][0], img1[i][1], img1[i][2]))

# print img2

for i in range(img1.__len__()):
    if img2[i] == 1785:
        img2[i] = 1
    elif img2[i] == 2295:
        img2[i] = 2
    elif img2[i] == 765:
        img2[i] = 3
    elif img2[i] == 1530:
        img2[i] = 4
    else:
        img2[i] = 5
print img2