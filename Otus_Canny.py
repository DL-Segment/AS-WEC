# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: BTEF
File Name: test.py
Author: ZhouBaoxian
Create Date: 2024/3/26
Description：
-------------------------------------------------
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np


def Otus_thresh(img_gray):
    # Otus 分割求解
    assert img_gray.ndim == 2, "must input a gary_img"  # shape有几个数字, ndim就是多少
    img_gray = np.array(img_gray).ravel().astype(np.uint8)
    th = 0.0
    GrayScale = 255
    # 总的像素数目
    PixSum = img_gray.size
    # 各个灰度值的像素数目
    PixCount = np.zeros(GrayScale)
    # 各灰度值所占总像素数的比例
    PixRate = np.zeros(GrayScale)
    # 统计各个灰度值的像素个数
    for i in range(PixSum):
        # 默认灰度图像的像素值范围为GrayScale
        Pixvalue = img_gray[i]
        PixCount[Pixvalue] = PixCount[Pixvalue] + 1

    # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
    for j in range(GrayScale):
        PixRate[j] = PixCount[j] * 1.0 / PixSum
    Max_var = 0
    # 确定最大类间方差对应的阈值
    for i in range(1, GrayScale):  # 从1开始是为了避免w1为0.
        u1_tem = 0.0
        u2_tem = 0.0
        # 背景像素的比列
        w1 = np.sum(PixRate[:i])
        # 前景像素的比例
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            pass
        else:  # 背景像素的平均灰度值
            for m in range(i):
                u1_tem = u1_tem + PixRate[m] * m
            u1 = u1_tem * 1.0 / w1
            # 前景像素的平均灰度值
            for n in range(i, GrayScale):
                u2_tem = u2_tem + PixRate[n] * n
            u2 = u2_tem / w2
            # 类间方差公式：G=w1*w2*(u1-u2)**2
            tem_var = w1 * w2 * np.power((u1 - u2), 2)
            # 判断当前类间方差是否为最大值
            if Max_var < tem_var:
                Max_var = tem_var  # 深拷贝，Max_var与tem_var占用不同的内存空间。
                th = i
    return th


def Linear_trans(y, interval=[-3,3]):
    # 按照比例缩放
    y=np.array(y)
    y_min = y.min()
    y_max = y.max()
    a = interval[0]
    b = interval[-1]
    y_new = (b-a)*(y-y_min)/(y_max-y_min)+a
    return y_new


def SLinear_trans(y, interval=[-3,3]):
    # 按照比例缩放
    y = np.array(y)
    y_min = y.min()
    y_max = y.max()
    y_ = 20*(y-y_min)/(y_max-y_min)-10
    y_new = 254/(1+np.exp(-y_))
    return y_new


def Trans_liner(val, interval=[-3, 3]):
    # 按原比例还原
    a = interval[0]
    b = interval[-1]
    scale = [0, 254]
    val_new = (b-a)*(val-scale[0])/(scale[-1]-scale[0])+a
    return val_new


def Grad_calc(img_blur):
    # 计算梯度和方向
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_dir = np.arctan2(sobel_y, sobel_x)

    # 非极大值抑制
    grad_mag_max = np.zeros(grad_mag.shape)
    for i in range(1, grad_mag.shape[0] - 1):
        for j in range(1, grad_mag.shape[1] - 1):
            # 如果角度小于0，使其加上一个pi,量化至四个方向
            if grad_dir[i, j] < 0:
                grad_dir[i, j] += np.pi
            if np.pi / 8 <= grad_dir[i, j] < 3 * np.pi / 8:
                if grad_mag[i, j] > grad_mag[i - 1, j - 1] and grad_mag[i, j] > grad_mag[i + 1, j + 1]:
                    grad_mag_max[i, j] = grad_mag[i, j]
            elif 3 * np.pi / 8 <= grad_dir[i, j] < 5 * np.pi / 8:
                if grad_mag[i, j] > grad_mag[i - 1, j] and grad_mag[i, j] > grad_mag[i + 1, j]:
                    grad_mag_max[i, j] = grad_mag[i, j]
            elif 5 * np.pi / 8 <= grad_dir[i, j] < 7 * np.pi / 8:
                if grad_mag[i, j] > grad_mag[i - 1, j + 1] and grad_mag[i, j] > grad_mag[i + 1, j - 1]:
                    grad_mag_max[i, j] = grad_mag[i, j]
            else:
                if grad_mag[i, j] > grad_mag[i, j - 1] and grad_mag[i, j] > grad_mag[i, j + 1]:
                    grad_mag_max[i, j] = grad_mag[i, j]
    return grad_mag_max


def Edge_pro(grad_original, high_threshold, low_threshold):
    # 边缘分析与计算
    edges = np.zeros(grad_original.shape)
    strong_edges = (grad_original > high_threshold)
    weak_edges = (grad_original >= low_threshold) & (grad_original <= high_threshold)
    edges[strong_edges] = 1
    while np.sum(weak_edges) > 0:
        i, j = np.unravel_index(weak_edges.argmax(),weak_edges.shape)  # .argmax()返回第一个参数   .shanpe返回矩阵各维数量（512*512）  np.unravel_index求索引值
        weak_edges[i, j] = 0
        # 强边缘连通性判断
        if strong_edges[i - 1:i + 2, j - 1:j + 2].any():  # .any() 判断迭代参数是否全是false
            edges[i, j] = 1
            strong_edges[i, j] = 1
        # 弱边缘的连通性判断
        if weak_edges[i - 1:i + 2, j - 1:j + 2].any():
            edges[i, j] = 1
            strong_edges[i, j] = 1
        else:
            edges[i, j] = 0
    return edges


def Pre_process(img):
    # 灰度化处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 1.4)

    return blur_img


if __name__ == '__main__':
    # 读取需要处理的图像
    img = cv2.imread("data/Edges/2.jpg")
    pre_img = Pre_process(img)

    # Otus自适应阈值求解
    grad_oral = Grad_calc(pre_img)

    # 第一次分割的Max阈值
    grad_resize = Linear_trans(grad_oral, interval=[0, 254])
    max_val = Otus_thresh(grad_resize)
    max_val = Trans_liner(max_val, interval=[grad_oral.min(), grad_oral.max()])

    # 第二次分割的Min阈值
    grad_del = np.zeros(grad_oral.shape)
    for i in range(0, grad_oral.shape[0]):
        for j in range(0, grad_oral.shape[1]):
            if grad_oral[i][j] > max_val:
                grad_del[i][j] = 0
            else:
                grad_del[i][j] = grad_oral[i][j]

    grad_resize = Linear_trans(grad_del, interval=[0, 254])
    min_val = Otus_thresh(grad_resize)
    min_val = Trans_liner(min_val, interval=[grad_del.min(), grad_del.max()])

    # 边缘连接处理
    edge = Edge_pro(grad_oral, max_val, min_val * 0.6)
    plt.imshow(edge, cmap='gray')
    cv2.imwrite('data/Edges/2_edges.png', edge * 255)
