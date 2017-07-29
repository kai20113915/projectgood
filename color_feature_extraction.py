# -*- coding: utf-8 -*-
from numpy import *
from PIL import Image
# 定义函数输入图像img与分割间隔steps，输出矩阵feature
# 其中feature矩阵格式每个单元像素[r,g,b,c]的矩阵
def featrure_extaction(img,steps):
    im = array(Image.open(img))                          #打开图片，并将图片转化为array数组
    dx = im.shape[0] / steps                             #数组x长度除以分割间隔steps，代表每个block的x方向像素个数dx(即block的x方向长度)
    dy = im.shape[1] / steps                             #数组y长度除以分割间隔steps，代表每个block的y方向像素个数dy
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,0])  #mean代表取均值，im矩阵，从x开始遍历每隔dx，dy，0代表r值，1代表g值，2代表b值
            G = mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,1])  #即对每个dx*dy的block的众多像素rgb取均值，得出一个rgb作为该block的rgb
            B = mean(im[x*dx:(x+1)*dx,y*dy:(y+1)*dy,2])
            features.append([R,G,B])
    features = array(features,'f')                       #f代表浮点型数据
    return features

# import color_feature_extraction
# img = color_feature_extraction.featrure_extaction('post.png', 100)
# print img