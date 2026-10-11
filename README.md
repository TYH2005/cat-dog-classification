# 🐱🐶 猫狗图像分类实验：从零搭建到部署上线

一个完整的计算机视觉入门实验项目，覆盖 **手动搭建 CNN → 迁移学习 → 数据增强 → 多模型对比 → 模型保存与推理 → Gradio 部署** 六步。

数据集 100 张图片（`cat.1~50.jpg` / `dog.51~100.jpg`），标签藏在文件名里。

> 参考实验指导书：`docs/猫狗分类实验指导书.md`（原始文档见 `docs/`）

---

## 目录结构

```
cat-dog/
├── datasets/              # 100 张猫狗图片（cat.*.jpg / dog.*.jpg）
├── docs/
│   ├── 猫狗分类实验指导书.md
│   └── alexnet.png        # AlexNet 结构图
├── .vscode/               # VSCode 配置（解释器、调试、推荐插件）
├── common.py              # 公共代码：数据集 / 数据加载 / 训练与评估
├── models.py              # 手动搭建的网络：AlexNet / AlexNetManual / ResNet18Manual
├── step1_scratch.py       # 第一步：手动 AlexNet，从零训练
├── step2_pretrained.py    # 第二步：加载 ImageNet 预训练（迁移学习）
├── step3_augment.py       # 第三步：数据增强
├── step4_compare.py       # 第四步：torchvision 多模型横向对比
├── step5_train_best.py    # 第五步(上)：训练并保存最佳模型
├── step5_2_test.py        # 第五步(下)：单张图片推理
├── step6_app.py           # 第六步：Gradio 拍照分类小程序
├── run_all.py             # 一键跑完六步
├── requirements.txt
├── install.ps1 / install.sh
└── checkpoints/           # 训练产出的权重（已 gitignore）
```

---

## 快速开始

### 1. 创建环境并安装依赖

**Windows (PowerShell)：**
```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

**macOS / Linux / Git Bash：**
```bash
bash install.sh
```

**手动安装（等价）：**
```bash
python -m venv .venv
# Windows
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS / Linux
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

> ⚠️ 一定要用 `--index-url https://download.pytorch.org/whl/cpu`，否则 `pip` 会下载 2GB+ 的 CUDA 版本，本实验只需要 CPU。

### 2. 在 VSCode 中打开

1. `文件 → 打开文件夹` 选择 `cat-dog`；
2. `Ctrl+Shift+P` → `Python: Select Interpreter` → 选择 `./.venv/Scripts/python.exe`；
3. 打开任意 `stepX_*.py`，按 **F5** 即可调试运行（已预置各步调试配置），或按 `Ctrl+F5` 直接运行。

### 3. 依次运行

```bash
python step1_scratch.py      # 第一步（较慢）
python step2_pretrained.py   # 第二步
python step3_augment.py      # 第三步
python step4_compare.py      # 第四步
python step5_train_best.py   # 第五步：产出 checkpoints/best_model.pth
python step5_2_test.py       # 单张图片推理
python step6_app.py          # 第六步：浏览器访问 http://127.0.0.1:7860
```

或一键运行：`python run_all.py`（可指定步骤，如 `python run_all.py 3 4`）。

---

## 六步实验对比

| 步骤 | 核心改动 | 预期准确率 | 核心结论 |
|---|---|---|---|
| 1. 手动搭建 + 从零训练 | 随机权重 | 0.65~0.80 | 小数据集从零训练学不出通用视觉特征 |
| 2. + ImageNet 预训练 | 灌入预训练权重、冻结卷积 | 0.90~1.00 | 迁移学习是小数据集的最优解 |
| 3. + 数据增强 | 训练集随机翻转/裁剪/色彩 | ≈步骤 2 或更稳 | 增强提升泛化，数据越多越明显 |
| 4. 多模型对比 | resnet18 / mobilenet / shufflenet | 都在 0.9+ | 小数据下差距小，选型看速度与体积 |
| 5. 保存与推理 | state_dict 保存 + 单图测试 | — | 训练与推理分离是工程标配 |
| 6. 部署小程序 | Gradio 摄像头界面 | — | 模型只有被用起来才有价值 |

**控制变量**：全程固定 ①同一份数据划分（seed=42）②同样的评价指标，每次只改一个变量。

---

## 关键知识点

- **卷积 / 池化 / BatchNorm / 残差连接 / 全连接**：见 `models.py` 中的手写实现与注释；
- **迁移学习**：`step2_pretrained.py` —— 先建 1000 类完整结构 → `load_state_dict` 官方权重 → 换 2 类分类头 → 冻结卷积；
- **数据增强只作用于训练集**：验证集必须用干净的 `Resize + Normalize`（见 `common.get_loaders`）；
- **保存最佳模型**：`torch.save(net.state_dict(), ...)`，只存参数、跨版本稳定。

---

## 进阶练习

1. 把冻结改为全模型微调（`lr=1e-5`），准确率和训练时间如何变化？
2. 数据增强再加强（如 `RandomErasing`），观察置信度分布变化。
3. 用 FastAPI 把 `predict` 封装成 HTTP 接口，写一个真正的微信小程序前端调用它。

---

## 常见问题

- **下载预训练权重慢/失败**：需要联网下载 torchvision 官方权重（首次运行自动下载到 `~/.cache/torch`）。如网络受限，可配置代理后再运行。
- **`ModuleNotFoundError: torch`**：确认 VSCode 右下角解释器选的是 `.venv`，而不是系统 Python。
- **手机访问不到 Gradio**：确认手机与电脑连的是同一个 Wi-Fi，并用电脑的局域网 IP（`ipconfig` 查看）。
