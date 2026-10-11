# -*- coding: utf-8 -*-
"""
step6_app.py —— 第六步：封装成可拍照分类的小程序（Gradio）

部署三要素：加载训练好的权重 -> 接收用户输入（拍照 / 上传）-> 返回预测结果。

运行：python step6_app.py
    - 本机访问:            http://127.0.0.1:7860
    - 手机(同一 Wi-Fi)访问: http://<电脑IP>:7860  例如 http://192.168.1.100:7860

（延伸）要做成真正的微信小程序：把 predict 逻辑换成 FastAPI 的 /predict 接口，
       小程序端用 wx.uploadFile 调用即可，思路完全一致。
"""
import os

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common import CHECKPOINTS_DIR, IMAGENET_MEAN, IMAGENET_STD, get_device

CLASSES = {0: "🐱 猫", 1: "🐶 狗"}
WEIGHT_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")

# ---------- 1. 加载模型（启动时只做一次） ----------
device = get_device()
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError(
        f"找不到权重文件 {WEIGHT_PATH}\n请先运行: python step5_train_best.py")

net = models.resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, 2)
net.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
net.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------- 2. 定义预测函数（每次拍照 / 上传都会调用） ----------
def predict(img: Image.Image):
    if img is None:
        return "请先拍照或上传一张图片", {}
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0].cpu()
    conf, pred = probs.max(0)
    detail = {CLASSES[i]: float(p) for i, p in enumerate(probs)}
    return f"识别结果：{CLASSES[int(pred)]}（置信度 {conf:.2%}）", detail


# ---------- 3. 搭建界面并启动 ----------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", sources=["webcam", "upload"], label="拍照或上传"),
    outputs=[gr.Textbox(label="识别结果"), gr.Label(label="各类别概率")],
    title="🐱🐶 猫狗分类器",
    description="基于 ResNet18 迁移学习训练。对准猫或狗拍照，或上传图片即可识别。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    # 本机访问: http://127.0.0.1:7860
    # 手机(同一局域网)访问: http://<电脑IP>:7860
