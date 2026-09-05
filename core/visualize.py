# -*- coding: utf-8 -*-
"""
检测结果可视化：在示例图片上叠加真实框（红）与预测框（绿+置信度），
供前端直观展示人员目标检测效果。
"""
from PIL import Image, ImageDraw, ImageFont
import os

GT_COLOR = (220, 38, 38)      # 红：真实框
PRED_COLOR = (34, 197, 94)    # 绿：预测框
MAX_WIDTH = 720

_FONT = None


def _label_font(size: int = 13):
    """加载支持中文的标签字体（Windows 优先微软雅黑/黑体），失败时回退默认字体。"""
    global _FONT
    if _FONT is None:
        for fp in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                   "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                   "/System/Library/Fonts/PingFang.ttc"):
            if os.path.exists(fp):
                try:
                    _FONT = ImageFont.truetype(fp, size)
                    break
                except Exception:  # noqa: BLE001
                    continue
        if _FONT is None:
            _FONT = ImageFont.load_default()
    return _FONT


def draw_detections(img_source, gt_boxes, pred_boxes, max_width: int = MAX_WIDTH):
    """
    绘制单张图的检测结果叠加图。

    参数:
        img_source: 图片绝对路径 或 PIL.Image（自定义上传的内存图片）
        gt_boxes:   list[xyxy]            真实 person 框
        pred_boxes: list[xyxy + conf]    预测 person 框（含置信度）

    返回:
        PIL.Image（RGB，已等比缩放）
    """
    if isinstance(img_source, Image.Image):
        img = img_source.convert("RGB")
    else:
        img = Image.open(img_source).convert("RGB")
    ratio = 1.0
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * ratio))), Image.LANCZOS)

    d = ImageDraw.Draw(img)
    f = _label_font()
    for b in gt_boxes:
        x1, y1, x2, y2 = [v * ratio for v in b[:4]]
        d.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)

    for p in pred_boxes:
        x1, y1, x2, y2 = [v * ratio for v in p[:4]]
        conf = float(p[4])
        d.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=2)
        # 标签只显示置信度（不标注类别名：上传图片不限于数据集固定类别）
        label = f"{conf:.2f}"
        tw = d.textlength(label, font=f)
        d.rectangle([x1, max(y1 - 16, 0), x1 + tw + 4, max(y1 - 16, 0) + 14],
                    fill=PRED_COLOR)
        d.text((x1 + 2, max(y1 - 15, 0)), label, fill=(0, 0, 0), font=f)
    return img
