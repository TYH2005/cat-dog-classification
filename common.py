# -*- coding: utf-8 -*-
"""
common.py —— 公共代码：数据集 / 数据加载 / 训练与评估

对应实验指导书 "0.2 数据加载" 与 "0.3 训练 + 评估"。
所有实验脚本都从这里 import，保证「控制变量」：

    ① 同一份数据划分（固定 seed=42）
    ② 同一套评价指标（Accuracy / Precision / Recall / F1）

每次实验只改动一个变量（网络结构 / 是否预训练 / 是否数据增强），对比才公平。
"""
import os

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
# ImageNet 预训练模型使用的均值 / 标准差。
# 迁移学习时输入必须用同一套归一化参数，否则预训练特征会"对不上"。
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 类别索引 -> 名称（标签由文件名解析而来：cat.* -> 0，dog.* -> 1）
CLASSES = ["cat 猫", "dog 狗"]

# 项目内的绝对路径，保证从任意工作目录运行都能找到文件
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

for _d in (CHECKPOINTS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_device():
    """有 CUDA 就用 CUDA，否则退回 CPU（本实验默认 CPU 即可跑通）。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class FileNameDataset(Dataset):
    """从文件名解析标签：cat.x.jpg -> 0，dog.x.jpg -> 1

    文件名里就藏着标签，所以不需要额外的标注文件。
    """

    def __init__(self, image_dir=DATASETS_DIR, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not self.images:
            raise FileNotFoundError(f"在 {image_dir} 下没有找到任何图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        label = 0 if "cat" in img_name.lower() else 1
        if self.transform:
            image = self.transform(image)
        return image, label


# --------------------------------------------------------------------------- #
# 数据加载器
# --------------------------------------------------------------------------- #
def get_loaders(data_dir=DATASETS_DIR, batch_size=16,
                train_transform=None, val_transform=None, seed=42):
    """统一划分：80% 训练 / 20% 验证，固定随机种子保证各实验可比。

    Args:
        data_dir:        图片目录
        batch_size:      批大小
        train_transform: 训练集预处理（可含数据增强）；None 时用 Resize+Normalize
        val_transform:   验证集预处理；None 时用干净的 Resize+Normalize
        seed:            随机种子
    """
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if val_transform is None:
        # 验证集必须"干净"：只做 Resize + Normalize，绝不做随机增强，
        # 否则评估指标不可信。
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # 分别给训练 / 验证套上不同的 transform（同一份索引，不同预处理）
    train_set = FileNameDataset(image_dir=data_dir, transform=train_transform)
    val_set = FileNameDataset(image_dir=data_dir, transform=val_transform)

    n_total = len(train_set)
    train_size = int(0.8 * n_total)
    val_size = n_total - train_size

    g = torch.Generator().manual_seed(seed)
    idx_train, idx_val = random_split(range(n_total), [train_size, val_size], generator=g)

    train_loader = DataLoader(Subset(train_set, idx_train.indices),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(val_set, idx_val.indices),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(net, loader, device="cpu"):
    """在给定数据集上评估，返回 Accuracy / Precision / Recall / F1（macro）。"""
    net.eval()
    all_preds, all_labels = [], []
    for X, Y in loader:
        preds = net(X.to(device)).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(Y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    return {"acc": acc, "precision": p, "recall": r, "f1": f1}


# --------------------------------------------------------------------------- #
# 训练 + 评估
# --------------------------------------------------------------------------- #
def train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3, device=None):
    """训练并在验证集上评估，返回准确率。

    只对 requires_grad=True 的参数做优化，因此天然支持"冻结卷积层"的迁移学习。
    """
    if device is None:
        device = get_device()

    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {running_loss / len(train_loader.dataset):.4f}")

    metrics = evaluate(net, val_loader, device)
    print(f"准确率: {metrics['acc']:.4f} | 精确率: {metrics['precision']:.4f} | "
          f"召回率: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    return metrics["acc"]


if __name__ == "__main__":
    # 自检：打印数据集规模与一个 batch 的形状
    train_loader, val_loader = get_loaders(batch_size=8)
    X, Y = next(iter(train_loader))
    print(f"设备: {get_device()}")
    print(f"训练集: {len(train_loader.dataset)} 张 | 验证集: {len(val_loader.dataset)} 张")
    print(f"一个 batch: X={tuple(X.shape)}, Y={Y.tolist()}")
