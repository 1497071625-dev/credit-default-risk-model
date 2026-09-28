# -*- coding: utf-8 -*-
# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
report_figures.py —— 生成报告排版用的合成图（不改变原始图）

为什么需要：最终报告和周报要把"混淆矩阵 / ROC / PR"三张方图放在一起，
竖着排会占掉一页多。这里把它们等高并排拼成一张宽图，正文里一行就能放下。

输入（原始图不动）
    outputs/W4/confusion_matrix_test.png
    outputs/W4/roc_curve_test.png
    outputs/W4/pr_curve_test.png
输出
    outputs/W4/fig_perf_row.png    三张方图并排（每张间隔 16px 白边）

运行
    .venv/bin/python scripts/W4/report_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BASE = next((p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
             if (p / "outputs").is_dir()), Path(__file__).resolve().parents[2])
W4 = BASE / "outputs" / "W4"

GAP = 16          # 每张图之间的白边（像素）
BG = (255, 255, 255)


def load(name: str) -> Image.Image:
    img = Image.open(W4 / name).convert("RGB")
    return img


def stack_row(images: list[Image.Image], gap: int = GAP) -> Image.Image:
    """等高并排：以最高者为基准等比放大其余图，避免缩放带来的模糊。"""
    h = max(im.height for im in images)
    scaled = [im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
              if im.height != h else im for im in images]
    w = sum(im.width for im in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (w, h), BG)
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def main() -> None:
    row = stack_row([load("confusion_matrix_test.png"),
                     load("roc_curve_test.png"),
                     load("pr_curve_test.png")])
    out = W4 / "fig_perf_row.png"
    row.save(out, dpi=(200, 200))
    print(f"已生成 {out}（{row.width}x{row.height}，{out.stat().st_size/1024:.0f} KB）")


if __name__ == "__main__":
    main()
