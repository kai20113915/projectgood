# -*- coding: utf-8 -*-
from numpy import *
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import xlrd
import matplotlib.pyplot as plt

train_set = xlrd.open_workbook('train_set.xls')#从制作好的excel表格里导入训练数据（训练集）
table = train_set.sheets()[0]#从excel表第一个sheet导入
nrows = table.nrows#读取训练集行数nrows
ncols = table.ncols#读取训练集列数ncols

X = []#读取到的234列分别代表RGB值，读取并生成二维训练集数组
for i in range(0, nrows):
    R = table.cell(i, 1).value
    G = table.cell(i, 2).value
    B = table.cell(i, 3).value
    X.append([R, G, B])
X = array(X, 'f')

y = []#读取到的第1列代表训练集label，该数组元素个数与X的维数相同
for i in range(0, nrows):
    label = table.cell(i, 0).value
    y.append(label)
X = array(X, 'f')

img = array(Image.open('hf_local_2016.png'))#读取遥感图像，并生成三维数组[[[R,G,B]]]
x_test = []#将读取的图像生成的三维数组改化成二维测试集数组[[R,G,B]],即一排[R,G,B]
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        x_test.append(img[i, j])
x_test = array(x_test, 'f')

rf = RandomForestClassifier()#定义一个随机森林类
rf.fit(X, y)#导入训练集X，及其对应label
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)#设置随机森林参数
label_set = rf.predict(x_test)#根据导入训练集对测试集进行预测
image_result = label_set.reshape(400, 400)#生成的label为原图像大小的数组即一维，需将一维数组改化成二维数组以成图
print image_result
# plt.imshow(image_result)
# plt.show()

# ---------------------以上随机森林分类结果-------------------------


# ---------------------转化真值图-------------------------

true_value = array(Image.open('true_value.png'))
img_true = []
for i in range(0, true_value.shape[0]):
    for j in range(0, true_value.shape[1]):
        img_true.append(true_value[i, j])
def rgb2gray(R, G, B):
    return  3 * R + 6 * G + 1 * B
img_label = []
for i in range(img_true.__len__()):
    img_label.append(rgb2gray(img_true[i][0], img_true[i][1], img_true[i][2]))

for i in range(img_true.__len__()):
    if img_label[i] == 1785:#湖泊
        img_label[i] = 4
    elif img_label[i] == 2295:
        img_label[i] = 5
    elif img_label[i] == 765:#绿地
        img_label[i] = 3
    elif img_label[i] == 1530:
        img_label[i] = 2
    else:
        img_label[i] = 1
print img_label

# ---------------------精度评定-------------------------


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

print my_confusion_matrix(img_label, label_set)
print my_classification_report(img_label, label_set)

plt.imshow(image_result)
plt.show()