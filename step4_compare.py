# -*- coding: utf-8 -*-
"""
step4_compare.py —— 第四步：调用 torchvision 封装好的模型横向对比

对比 resnet18 / mobilenet_v3_small / shufflenet_v2_x1_0：
小数据集上准确率可能相差不大（都在 0.9+），此时推理速度与模型体积才是选型关键
—— 这正好为第六步部署做铺垫。

运行：python step4_compare.py
"""
import os
import time

import torch
import torch.nn as nn
from torchvision import models

from common import CHECKPOINTS_DIR, DATASETS_DIR, get_device, get_loaders, train_and_eval

# 展示名 -> (torchvision 构造函数, 权重枚举, 参数量)
MODEL_ZOO = {
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "mobilenet_v3_small": lambda: models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT),
    "shufflenet_v2_x1_0": lambda: models.shufflenet_v2_x1_0(
        weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
}


def build_model(name, num_classes=2, freeze=True):
    """统一接口：加载预训练模型并替换分类头（不同模型头名字不一样）。"""
    net = MODEL_ZOO[name]()

    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "mobilenet_v3_small":
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head_params = net.classifier[3].parameters()
    elif name == "shufflenet_v2_x1_0":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    else:
        raise ValueError(f"未知模型: {name}")

    if freeze:  # 冻结特征提取层，只训练分类头（CPU 友好）
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return net


def main():
    device = get_device()
    train_loader, val_loader = get_loaders(DATASETS_DIR, batch_size=16)

    results = {}
    for name in MODEL_ZOO:
        print(f"\n===== 模型: {name} =====")
        net = build_model(name)
        n_param = sum(p.numel() for p in net.parameters())

        t0 = time.time()
        acc = train_and_eval(net, train_loader, val_loader,
                             epochs=5, lr=1e-3, device=device)
        elapsed = time.time() - t0

        torch.save(net.state_dict(), os.path.join(CHECKPOINTS_DIR, f"model_{name}.pth"))
        results[name] = {"acc": acc, "time": elapsed, "params": n_param}
        print(f"参数量 {n_param / 1e6:.2f}M | 5 轮训练用时 {elapsed:.1f}s")

    print("\n===== 横向对比结果 =====")
    print(f"{'模型':<24}{'参数量':>10}{'准确率':>10}{'用时(s)':>10}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        print(f"{name:<24}{r['params'] / 1e6:>8.2f}M{r['acc']:>10.4f}{r['time']:>10.1f}")
    print("\n提示：小数据下三者差距小，选型看推理速度与模型体积（为第六步部署铺垫）。")


if __name__ == "__main__":
    main()
