# -*- coding: utf-8 -*-
"""
step5_train_best.py —— 第五步（上）：训练并保存最佳模型

工程惯例：训练过程中记录验证集最佳准确率，只保存 best 模型（早停思想）。
保存方式：只保存参数 state_dict（文件小、跨版本稳定），加载时先建同结构模型再 load。

运行：python step5_train_best.py
产物：checkpoints/best_model.pth
"""
import os

import torch
import torch.nn as nn
from torchvision import models

from common import CHECKPOINTS_DIR, DATASETS_DIR, get_device, get_loaders

BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")


def train_with_best_save(net, train_loader, val_loader,
                         epochs=5, lr=1e-3, save_path=BEST_MODEL_PATH, device=None):
    if device is None:
        device = get_device()

    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()

        # 验证
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                correct += (net(X).argmax(1) == Y).sum().item()
                total += Y.size(0)
        acc = correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] Val Acc: {acc:.4f}")

        if acc > best_acc:                      # 只保留历史最佳
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            print(f"  ^ 最佳模型已保存到 {save_path}")

    print(f"最佳准确率: {best_acc:.4f}")
    return best_acc


def build_resnet18(num_classes=2):
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True
    return net


def main():
    device = get_device()
    train_loader, val_loader = get_loaders(DATASETS_DIR, batch_size=16)

    net = build_resnet18(num_classes=2)
    print(f"设备: {device} | 模型: torchvision resnet18 + 预训练，冻结卷积只训练 fc")
    train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3, device=device)


if __name__ == "__main__":
    main()
