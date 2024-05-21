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
import numpy as np
from sklearn.metrics import f1_score, recall_score, mean_absolute_error
import cv2
import numpy as np

def preprocess_grayscale_image(image, threshold=128):
    _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_mask

# 读取灰度图像
original_image = cv2.imread('data/results/TCGA_CS_6668_20011025_17.png', cv2.IMREAD_GRAYSCALE)
mask_image = cv2.imread('data/Test_Labels/TCGA_CS_6668_20011025_17_mask.png', cv2.IMREAD_GRAYSCALE)
# 预处理图像，创建二进制分割掩模
threshold_value = 128  # 选择合适的阈值

binary_mask = preprocess_grayscale_image(original_image, threshold_value)
predict_mask = preprocess_grayscale_image(mask_image, threshold_value)
# 生成 y_true 和 y_pred
y_true = binary_mask / 255  # 将二进制掩模转为 {0, 1} 形式
y_pred = predict_mask / 255  # 生成一个随机的 y_pred，模拟分割结果

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2.0 * intersection) / (union + 1e-7)  # Adding a small epsilon to avoid division by zero
    return dice

def sensitivity(y_true, y_pred):
    recall = recall_score(y_true.flatten(), y_pred.flatten())
    return recall


dice = dice_coefficient(y_true, y_pred)
sen = sensitivity(y_true, y_pred)
mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())

print(f'Dice Coefficient: {dice}')
print(f'Sensitivity: {sen}')
print(f'Mean Absolute Error: {mae}')
