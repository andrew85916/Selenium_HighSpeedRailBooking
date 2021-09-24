import cv2
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

WIDTH = 140
HEIGHT = 48

def imgDenoise(filename):
    img = cv2.imread(filename)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)
    return dst

def img2Gray(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

def findRegression(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:, 14:WIDTH - 7] = 0
    imagedata = np.where(img == 255)

    X = np.array([imagedata[1]])
    Y = HEIGHT - imagedata[0]

    poly_reg = PolynomialFeatures(degree=2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)
    return regr

def dePolynomial(img, regr):
    X2 = np.array([[i for i in range(0, WIDTH)]])
    poly_reg = PolynomialFeatures(degree=2)
    X2_ = poly_reg.fit_transform(X2.T)
    offset = 4
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for ele in np.column_stack([regr.predict(X2_).round(2), X2[0]]):
        pos = HEIGHT - int(ele[0])
        img[pos - offset:pos + offset,
               int(ele[1])] = 255 - img[pos - offset:pos + offset, int(ele[1])]
    return img

def predict_file_preprocessing(from_filename, to_filename):
    if not os.path.isfile(from_filename):
        return
    img = imgDenoise(from_filename)
    img = img2Gray(img)
    regr = findRegression(img)
    img = dePolynomial(img, regr)
    cv2.imwrite(to_filename, img)
    return