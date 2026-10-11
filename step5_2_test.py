# -*- coding: utf-8 -*-
"""
step5_2_test.py —— 第五步（下）：加载最佳模型，对单张图片做推理

演示「训练与推理分离」：推理时不需要数据集，只要权重 + 一张图。

用法：
    python step5_2_test.py
    python step5_2_test.py checkpoints/best_model.pth datasets/dog.60.jpg
"""
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common import CHECKPOINTS_DIR, DATASETS_DIR, IMAGENET_MEAN, IMAGENET_STD, get_device

CLASSES = ["cat 猫", "dog 狗"]


def load_model(weight_path=os.path.join(CHECKPOINTS_DIR, "best_model.pth")):
    # 结构必须与训练时一致；这里用 resnet18，weights=None 表示不下载官方权重
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()
    return net


def predict(image_path, net, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)   # 增加 batch 维 -> [1,3,224,224]
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]  # 得分转概率
    pred = probs.argmax().item()
    return CLASSES[pred], probs[pred].item(), probs


def main():
    device = get_device()
    weight_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    image_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATASETS_DIR, "cat.1.jpg")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"找不到权重文件 {weight_path}，请先运行 python step5_train_best.py")

    net = load_model(weight_path).to(device)
    label, conf, probs = predict(image_path, net, device)
    print(f"图片: {image_path}")
    print(f"预测: {label}，置信度 {conf:.2%}")
    print(f"各类概率: 猫 {probs[0]:.2%} | 狗 {probs[1]:.2%}")


if __name__ == "__main__":
    main()
