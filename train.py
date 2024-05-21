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
from model.MySeg_Model import MySegNet
from model.MySeg_Model import weights_init
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        total_loss = 0
        # 训练模式
        net.train()
        with tqdm(total=per_epoch_num, desc=f'Epoch {epoch+1} / {epochs}', postfix=dict, mininterval=0.3) as pbar:
            # 按照batch_size开始训练
            for iteration, batch in enumerate(train_loader):
                if iteration >= len(train_loader):
                    break
                image, label = batch
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 计算loss
                loss = criterion(pred, label)
                # print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())

                # 更新参数
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(**{
                    'total_loss': total_loss / (iteration + 1),
                    'lr': get_lr(optimizer)
                })
                pbar.update(1)
        # 保存loss值最小的网络参数
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), 'best_model.pth')


if __name__ == "__main__":

    # 加载网络，图片单通道1，分类为1。
    net = MySegNet(1,1)  # todo edit input_channels n_classes
    # 载入模型权重
    model_path = 'weights/best_model.pth'
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = net.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    else:
        # 初始化权重
        weights_init(net)

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "data"  # todo 修改为你本地的数据集位置
    train_net(net, device, data_path, epochs=30, batch_size=1)
