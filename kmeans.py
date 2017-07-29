# -*- coding: utf-8 -*-
from scipy.cluster.vq import *
from scipy.misc import imresize
from PIL import Image,ImageDraw
from numpy import *
import matplotlib.pyplot as plt
steps = 500
im = array(Image.open('man.png'))
dx = im.shape[0] / steps
dy = im.shape[1] / steps
features = []
for x in range(steps):
    for y in range(steps):
        R = mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,0])
        G = mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,1])
        B = mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,2])
        features.append([R,G,B])
features = array(features,'f')
centroids,variance = kmeans(features,10)
code,distance = vq(features,centroids)
codeim = code.reshape(steps,steps)
codeim = imresize(codeim,im.shape[:2],interp='nearest')

plt.imshow(codeim)
plt.show()
