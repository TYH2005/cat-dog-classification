# -*- coding: utf-8 -*-
"""
step1_scratch.py —— 第一步：手动搭建 AlexNet，从零训练

核心：随机初始化权重，不借助任何预训练知识。
预期：小数据集上从零训练学不出通用视觉特征，准确率约 0.65~0.80。

运行：python step1_scratch.py
"""
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import DATASETS_DIR, OUTPUTS_DIR, get_device
from models import AlexNet, init_weight


# --------------------------------------------------------------------------- #
# 索引式数据集：先用 txt 记录「图片路径 标签」，再按 txt 读取
# （与 common.py 的 FileNameDataset 是两种思路，本步演示这种更通用的做法）
# --------------------------------------------------------------------------- #
class DatasetLoader(Dataset):
    def __init__(self, dataset_path):
        self.images_path = []
        self.labels = []
        with open(dataset_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    self.images_path.append(parts[0])
                    self.labels.append(int(parts[1]))

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return image_transform(image)

    def __getitem__(self, index):
        image = self.preprocess_image(self.images_path[index])
        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)


def dataloader(base_path, batch_size, seed=42):
    """划分数据集并生成 txt 索引文件，返回训练 / 测试 DataLoader。"""
    random.seed(seed)  # 固定种子，保证各实验划分一致（控制变量）
    imgs_abs_path = [
        os.path.join(base_path, i)
        for i in os.listdir(base_path)
        if i.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    test_size = int(0.2 * len(imgs_abs_path))
    test_files = set(random.sample(imgs_abs_path, test_size))

    train_txt = os.path.join(OUTPUTS_DIR, "trainDatasets.txt")
    test_txt = os.path.join(OUTPUTS_DIR, "testDatasets.txt")

    with open(test_txt, "w", encoding="utf-8") as f1, \
            open(train_txt, "w", encoding="utf-8") as f2:
        for i in imgs_abs_path:
            label = 0 if "cat" in i.lower() else 1
            (f1 if i in test_files else f2).write(f"{i} {label}\n")

    train_dataset = DatasetLoader(train_txt)
    test_dataset = DatasetLoader(test_txt)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def main():
    batch_size = 16
    lr = 1e-4      # 从头训练自定义网络，建议用较小的学习率
    epochs = 30
    device = get_device()

    net = AlexNet(num_classes=2)
    net.apply(init_weight)
    net.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    base_path = DATASETS_DIR
    train_iter, test_iter = dataloader(base_path, batch_size=batch_size)
    print(f"设备: {device} | 训练 {len(train_iter.dataset)} 张 / 测试 {len(test_iter.dataset)} 张")
    print("开始训练（从零训练，无预训练）...")

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_iter:
            X, Y = X.to(device), Y.to(device)
            y_hat = net(X)
            loss = loss_fn(y_hat, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        # 验证与评价指标
        net.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in test_iter:
                X, Y = X.to(device), Y.to(device)
                y_hat = net(X)
                val_loss += loss_fn(y_hat, Y).item() * X.size(0)
                all_preds.extend(y_hat.argmax(1).cpu().numpy())
                all_labels.extend(Y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0)

        print(f"EPOCH: {epoch + 1}/{epochs}")
        print(f"  -> Train Loss: {running_loss / len(train_iter.dataset):.4f} | "
              f"Test Loss: {val_loss / len(test_iter.dataset):.4f}")
        print(f"  -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | F1: {f1:.4f}\n")


if __name__ == "__main__":
    main()
