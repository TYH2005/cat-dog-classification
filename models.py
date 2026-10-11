# -*- coding: utf-8 -*-
"""
models.py —— 手动搭建的网络结构（实验第一 ~ 第三步使用）

包含：
    AlexNet         : 第一步用的精简版 AlexNet（从零训练）
    AlexNetManual   : 第二步用的 AlexNet，结构与 torchvision 一致，
                      可以直接 load_state_dict 官方预训练权重
    BasicBlock /
    ResNet18Manual  : 第三步用的手动 ResNet18，结构与 torchvision 一致，
                      同样可以加载官方预训练权重

设计原则：这些手写模型的「子模块命名 / 层级顺序」都刻意与 torchvision 保持一致，
所以能 strict=True / strict=False 地加载官方 ImageNet 权重——这正是第二步
"迁移学习"的关键。
"""
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 第一步：精简版 AlexNet（从零训练）
# --------------------------------------------------------------------------- #
class AlexNet(nn.Module):
    """精简版 AlexNet（去掉了当年双 GPU 的设计）。

    输入 224x224x3 经过 5 个卷积 + 3 个池化后得到 256x5x5，
    展平后是 6400 维，接两个 4096 的全连接层，最后输出 2 类。
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            # 用 11*11 的大窗口捕捉对象；步幅 4 快速降低分辨率
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，padding=2 保持高宽不变，增大通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 三个连续的小卷积层
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 全连接层用 Dropout 减轻过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, X):
        return self.net(X)


def init_weight(m):
    """Xavier 初始化：从零训练时让各层输出方差稳定。"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


# --------------------------------------------------------------------------- #
# 第二步：可与 torchvision 权重对齐的 AlexNet
# --------------------------------------------------------------------------- #
class AlexNetManual(nn.Module):
    """结构与 torchvision.models.alexnet 完全一致。

    - 卷积部分必须命名为 features，且层顺序与 torchvision 相同；
    - 分类部分必须命名为 classifier，索引 6 是最后一层（换成 2 类）。
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),   # 索引 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # 索引 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # 索引 6 <- 分类头，换这里
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# 第三步：手动 ResNet18（残差连接）
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """ResNet18/34 使用的基础残差块：两层 3x3 卷积 + 恒等(或下采样)捷径。"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：形状不一致时用 1x1 卷积把捷径投影到相同形状
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity          # "+x" 让梯度可以抄近道回传
        out = self.relu(out)
        return out


class ResNet18Manual(nn.Module):
    """手动实现的 ResNet18，子模块命名与 torchvision.models.resnet18 一致。"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 权重初始化（若之后加载官方预训练权重会被覆盖）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # 自检：打印三个模型的输出形状
    for name, model in [("AlexNet", AlexNet()),
                        ("AlexNetManual", AlexNetManual()),
                        ("ResNet18Manual", ResNet18Manual())]:
        out = model(torch.zeros(2, 3, 224, 224))
        n_param = sum(p.numel() for p in model.parameters())
        print(f"{name:16s} 输出 {tuple(out.shape)}  参数量 {n_param / 1e6:.2f}M")
