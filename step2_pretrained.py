# -*- coding: utf-8 -*-
"""
step2_pretrained.py —— 第二步：加载 ImageNet 预训练参数（迁移学习）

核心：先建 1000 类的完整 AlexNet 结构 -> 加载官方预训练权重 -> 把分类头换成 2 类
      -> 冻结卷积层，只训练最后的分类器。
预期：准确率跃升到 0.90~1.00。小数据集上「知识从哪来」比「训练多久」重要得多。

运行：python step2_pretrained.py

进阶：删掉冻结的 for 循环，改用 lr=1e-5、epochs=5 微调全部层，
      对比「冻结 vs 微调」的准确率和训练时间。
"""
import torch
import torch.nn as nn
from torchvision import models

from common import DATASETS_DIR, get_device, get_loaders, train_and_eval
from models import AlexNetManual

# 是否冻结卷积层（只训练分类头）。设为 False 即"全模型微调"的进阶实验。
FREEZE_BACKBONE = True


def build_alexnet(num_classes=2, freeze=True):
    """构建加载了 ImageNet 预训练权重的 AlexNet。"""
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)       # 1. 先建 1000 类的完整结构
    net.load_state_dict(official.state_dict())  # 2. 形状完全一致，strict=True 也能加载
    net.classifier[6] = nn.Linear(4096, num_classes)  # 3. 分类头换成 2 类

    if freeze:
        # 冻结：只训练 classifier（也可以只训练最后一层）
        for name, param in net.named_parameters():
            param.requires_grad = name.startswith("classifier")
    return net


def main():
    device = get_device()
    train_loader, val_loader = get_loaders(DATASETS_DIR, batch_size=16)

    net = build_alexnet(num_classes=2, freeze=FREEZE_BACKBONE)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"设备: {device} | 冻结卷积层: {FREEZE_BACKBONE}")
    print(f"可训练参数: {trainable / 1e6:.2f}M / 总计 {total / 1e6:.2f}M")

    acc = train_and_eval(net, train_loader, val_loader,
                         epochs=5, lr=1e-3, device=device)
    print(f"\n【第二步 预训练迁移】最终准确率: {acc:.4f}")


if __name__ == "__main__":
    main()
