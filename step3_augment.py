# -*- coding: utf-8 -*-
"""
step3_augment.py —— 第三步：加入数据增强

核心：在「预训练迁移」基础上，只给【训练集】加随机增强，逼模型学本质特征而非背图。
      - RandomHorizontalFlip      猫狗朝左朝右都是猫狗
      - RandomResizedCrop         模拟拍摄距离 / 构图变化
      - ColorJitter               模拟光线 / 白平衡变化
      - RandomRotation            模拟相机角度轻微倾斜

注意：增强只作用于训练集，验证集必须用干净的 Resize + Normalize。

运行：python step3_augment.py
"""
import torch.nn as nn
from torchvision import models, transforms

from common import DATASETS_DIR, IMAGENET_MEAN, IMAGENET_STD, get_device, get_loaders, train_and_eval
from models import ResNet18Manual


def build_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机裁剪缩放
        transforms.RandomHorizontalFlip(),                    # 随机水平翻转
        transforms.ColorJitter(0.2, 0.2, 0.2),                # 随机亮度/对比度/饱和度
        transforms.RandomRotation(10),                        # 随机小角度旋转
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_resnet18(num_classes=2, pretrained=True, freeze=True):
    """手动 ResNet18 + 官方预训练权重 + 替换分类头。"""
    net = ResNet18Manual(num_classes=num_classes)
    if pretrained:
        official = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # strict=False：分类头 fc 形状不同（1000 vs 2），跳过它即可
        net.load_state_dict(official.state_dict(), strict=False)
    if freeze:
        for name, param in net.named_parameters():
            param.requires_grad = "fc" in name
    return net


def main():
    device = get_device()
    train_loader, val_loader = get_loaders(
        DATASETS_DIR, batch_size=16, train_transform=build_train_transform())

    net = build_resnet18(num_classes=2, pretrained=True, freeze=True)
    print(f"设备: {device} | 模型: ResNet18Manual + 预训练 + 数据增强")

    acc = train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3, device=device)
    print(f"\n【第三步 预训练 + 数据增强】最终准确率: {acc:.4f}")


if __name__ == "__main__":
    main()
