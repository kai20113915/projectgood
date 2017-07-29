# -*- coding: utf-8 -*-
# from PIL import Image,ImageDraw
# from numpy import *
# steps = 100
# im = array(Image.open('hf_local_2016.png'))
# dx = im.shape[0] / steps
# dy = im.shape[1] / steps
# m = im.size
# print im, im.shape[0], im.shape[1], dx, dy, m
# print im[0, 0, 0], im[0, 0, 1], im[0, 0, 2]

# -----------------------------------

#
# import color_feature_extraction
# img = color_feature_extraction.featrure_extaction('post.png', 100)
# print img


# ----------------------------
# from numpy import *
# import xlrd
# data = xlrd.open_workbook('train_data.xls')
# table = data.sheets()[0]
# # table = data.sheet_by_index(0)
# # table = data.sheet_by_name(u'Sheet1')
# nrows = table.nrows
# ncols = table.ncols
# # table.row_values()
# # table.col_values()

# X = []
# for i in range(0, nrows):
#     R = table.cell(i, 1).value
#     G = table.cell(i, 2).value
#     B = table.cell(i, 3).value
#     X.append([R, G, B])
# # print X[0][1]
# print X
#
# y = []
# for i in range(0, nrows):
#     label = table.cell(i, 0).value
#     y.append(label)
# print y

# ------------------------------
#
# from PIL import Image,ImageDraw
# from numpy import *
# im = array(Image.open('hf_local_2016.png'))
#
# hf_dataset = []
# for i in range(0, im.shape[0]):
#     for j in range(0, im.shape[1]):
#         hf_dataset.append(im[i, j])
# print im, im.shape[0], im.shape[1]
# print im[0, 0, 0], im[0, 0, 1], im[0, 0, 2]
# print im[0, 0]
# print hf_dataset[1]



# ----------------------------

from numpy import *
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import xlrd

# ---------------------------------

img = array(Image.open('hf_local_2016.png'))
x_test = []
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        x_test.append(img[i, j])

# x_test = map(eval, x_test)
# print type(x_test[0][0])
# x_test = array(x_test, 'f')

data = xlrd.open_workbook('train_data.xls')
table = data.sheets()[0]

# table = data.sheet_by_index(0)
# table = data.sheet_by_name(u'Sheet1')

nrows = table.nrows
ncols = table.ncols

# table.row_values()
# table.col_values()

X = []
for i in range(0, nrows):
    R = table.cell(i, 1).value
    G = table.cell(i, 2).value
    B = table.cell(i, 3).value
    X.append([R, G, B])
# print X[0][1]
# X = array(X, 'f')
print type(X[0][0])

y = []
for i in range(0, nrows):
    label = table.cell(i, 0).value
    y.append(label)
# y = array(y, 'f')
print y

# rf = RandomForestClassifier()
# rf.fit(X, y)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
#
# print rf.predict_proba(x_test)

# ## -------------------
# data=[[0,0,0],[1,1,1],[2,2,2],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[4,4,4]]
# arget=[0,1,2,1,2,3,1,4]
# data = array(data, 'f')
# rf = RandomForestClassifier()
# rf.fit(data, arget)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
#
# print rf.predict_proba([[1,1,1]])
# print type(data[0][0])


# -----------------

from numpy import *
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import xlrd
import numpy as np
import matplotlib.pyplot as plt

train_data = xlrd.open_workbook('train_data.xls')
table = train_data.sheets()[0]
nrows = table.nrows
ncols = table.ncols

X = []
for i in range(0, nrows):
    R = table.cell(i, 1).value
    G = table.cell(i, 2).value
    B = table.cell(i, 3).value
    X.append([R, G, B])
X = array(X, 'f')

y = []
for i in range(0, nrows):
    label = table.cell(i, 0).value
    y.append(label)
X = array(X, 'f')

# data=[[0,0,0],[1,1,1],[2,2,2],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[4,4,4]]
# arget=[0,1,2,1,2,3,1,4]
# data = array(data, 'f')
# arget = array(arget, 'f')

rf = RandomForestClassifier()
rf.fit(X, y)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

img = array(Image.open('hf_local_2016.png'))
x_test = []
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        x_test.append(img[i, j])
x_test = array(x_test, 'f')

label_set = rf.predict(x_test)


print label_set
# print rf.predict_proba(x_test)

# print X
# print y
# print type(X[0][0]), type(y[0])ko
# print X.__len__()
# print y.__len__()
#
# print type(data[0][0]), type(arget[0])
# print data.__len__()
# print arget.__len__()

label_set = label_set.tolist()
print type(label_set)
print label_set.__len__()


label_set = np.array(label_set)
image_result = label_set.reshape(400, 400)
print image_result
plt.imshow(image_result)
plt.show()


def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print "confusion_matrix(left labels: y_true, up labels: y_pred):"
    print "labels\t",
    for i in range(len(labels)):
        print labels[i], "\t",
    print
    for i in range(len(conf_mat)):
        print i, "\t",
        for j in range(len(conf_mat[i])):
            print conf_mat[i][j], '\t',
        print
    print


def my_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    print "classification_report(left: labels):"
    print classification_report(y_true, y_pred)


# mm = my_confusion_matrix()

# from sklearn import metrics
# precision = metrics.precision_score(x_test, image_result)
# print precision

# img1 = array(Image.open('true_value.png'))
#
# plt.imshow(img1)
# plt.show()
# print img1[399][399]


from scipy.cluster.vq import *
from scipy.misc import imresize
from PIL import Image,ImageDraw
from numpy import *
import matplotlib.pyplot as plt
steps = 400
im = array(Image.open('hf_local_2016.png'))
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
centroids,variance = kmeans(features,5)
code,distance = vq(features,centroids)
codeim = code.reshape(steps,steps)
codeim = imresize(codeim,im.shape[:2],interp='nearest')

plt.imshow(codeim)
plt.show()

