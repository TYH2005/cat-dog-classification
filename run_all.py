# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑完六步实验（可选）

按顺序执行各步骤脚本，方便一次性复现全部结果。
注意：第一、二步较慢（从零训练 / 迁移学习各 5~30 轮，CPU 上需要一些时间）。

用法：
    python run_all.py            # 跑全部
    python run_all.py 3 4        # 只跑第 3、4 步
"""
import runpy
import sys
import time
from pathlib import Path

STEPS = {
    "1": ("第一步 手动 AlexNet 从零训练", "step1_scratch.py"),
    "2": ("第二步 加载 ImageNet 预训练", "step2_pretrained.py"),
    "3": ("第三步 数据增强", "step3_augment.py"),
    "4": ("第四步 多模型横向对比", "step4_compare.py"),
    "5": ("第五步 保存最佳模型", "step5_train_best.py"),
    "6": ("第五步 单张图片推理", "step5_2_test.py"),
    "7": ("第六步 Gradio 小程序（会阻塞，需手动关闭）", "step6_app.py"),
}


def main():
    targets = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    root = Path(__file__).resolve().parent

    for key in targets:
        if key not in STEPS:
            print(f"跳过未知步骤: {key}")
            continue
        title, script = STEPS[key]
        print("\n" + "=" * 70)
        print(f"【{key}】{title}  ({script})")
        print("=" * 70)
        t0 = time.time()
        runpy.run_path(str(root / script), run_name="__main__")
        print(f"--- 用时 {time.time() - t0:.1f}s ---")


if __name__ == "__main__":
    main()
