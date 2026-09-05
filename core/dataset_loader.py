# -*- coding: utf-8 -*-
"""
多数据集加载模块。

统一输出结构:
    Dataset(
        name    数据集名称
        version 版本
        scope   测评范围描述（供测评报告记录）
        samples list[(image_path, boxes)]
            image_path: 图片绝对路径
            boxes:      list[xyxy] person 真实框
    )

支持的数据集:
    Pascal VOC       data/voc/JPEGImages/ + data/voc/Annotations/*.xml（自动过滤 person）
    USC Pedestrian   data/usc/（bmp + gt.xml）
    Penn-Fudan       data/PennFudanPed/（PNG + 对应标注）
    NYC 景点地标     data/archive/nyc_landmarks/（YOLO txt 格式）
    自定义 YOLO      内存图片 / images/ + labels/*.txt
"""
import glob
import os
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

import config


@dataclass
class Dataset:
    name: str
    version: str
    scope: str
    samples: List[Tuple[str, List[List[float]]]] = field(default_factory=list)
    classes: Optional[List[str]] = None  # 多类数据集的 GT 类别名（如景点检测）

    @property
    def num_images(self):
        return len(self.samples)

    @property
    def num_person(self):
        return sum(len(b) for _, b in self.samples)

    def summary(self):
        return {
            "name": self.name,
            "version": self.version,
            "scope": self.scope,
            "num_images": self.num_images,
            "num_person": self.num_person,
        }


def _parse_voc_xmls(ann_dir: str, jpeg_dir: str,
                    max_images: Optional[int] = None,
                    prefixes: Optional[tuple] = None,
                    include: Optional[set] = None):
    """解析 VOC 格式标注目录，返回 [(image_path, boxes)]，仅保留 person 类。

    优先使用 XML 中的 <path> 定位图片（兼容 MPII 图片不复制到 JPEGImages 的情况），
    否则回退到 jpeg_dir + filename。
    prefixes: 文件名前缀过滤（如 ("2007te",) 只取 VOC2007 test 子集），None 取全部。
    include: 仅保留的 XML 文件名集合（不含扩展名），官方 val 划分用。
    """
    xmls = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
    if prefixes:
        xmls = [p for p in xmls
                if os.path.basename(p).startswith(tuple(prefixes))]
    if include:
        xmls = [p for p in xmls
                if os.path.splitext(os.path.basename(p))[0] in include]
    samples = []
    for xml_path in xmls:
        tree = ET.parse(xml_path)
        root_ = tree.getroot()
        filename = root_.findtext("filename")
        boxes = []
        for obj in root_.findall("object"):
            if obj.findtext("name") != "person":
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            boxes.append([
                float(b.findtext("xmin")), float(b.findtext("ymin")),
                float(b.findtext("xmax")), float(b.findtext("ymax")),
            ])
        if not boxes:
            continue
        path = root_.findtext("path") or ""
        if not (path and os.path.exists(path)):
            path = os.path.join(jpeg_dir,
                                filename or os.path.basename(xml_path)[:-4] + ".jpg")
        if os.path.exists(path):
            samples.append((path, boxes))
        if max_images and len(samples) >= max_images:
            break
    return samples


def load_voc(root: str, max_images: Optional[int] = None,
             prefixes: Optional[tuple] = None,
             ids_file: Optional[str] = None) -> Dataset:
    """Pascal VOC，自动过滤 person 类。

    prefixes 用于选取 2007/2012 子集；
    ids_file 为官方 ImageSets 划分清单（每行一个 XML 文件名），
    如 VOC2012 val 只取官方 val 子集。
    """
    jpeg_dir = os.path.join(root, "JPEGImages")
    ann_dir = os.path.join(root, "Annotations")
    include = None
    if ids_file and os.path.exists(ids_file):
        with open(ids_file, "r", encoding="utf-8") as f:
            include = {ln.strip() for ln in f if ln.strip()}
    samples = _parse_voc_xmls(ann_dir, jpeg_dir, max_images, prefixes,
                              include)
    if not samples:
        raise FileNotFoundError(
            f"未找到 VOC 标注或图片: {ann_dir}")

    if prefixes:
        friendly = {"2007te": "2007 test", "2007tv": "2007 trainval",
                    "2012tv": "2012 trainval"}
        version = "+".join(friendly.get(p, p) for p in prefixes)
    else:
        version = "2007+2012"
    if include and ids_file and os.path.basename(ids_file) == "voc2012_val_ids.txt":
        version = "2012 val（官方划分）"
    return Dataset(
        name="Pascal VOC",
        version=version,
        scope=f"Pascal VOC {version} person 类（{len(samples)} 张，{sum(len(b) for _, b in samples)} 个实例）",
        samples=samples,
    )


def load_custom_yolo(root: str, max_images: Optional[int] = None,
                     person_cls: int = 0) -> Dataset:
    """
    自定义 YOLO 格式数据集：
        data/custom_yolo/images/*.jpg|png|...
        data/custom_yolo/labels/*.txt  （cls xc yc w h，归一化坐标，0 表示 person）
    """
    img_dir = os.path.join(root, "images")
    label_dir = os.path.join(root, "labels")
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    images = sorted(images)
    if not images:
        raise FileNotFoundError(f"未找到自定义数据集图片: {img_dir}\n请参考 README 放置数据集。")

    samples = []
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        boxes = []
        if os.path.exists(label_path):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 1, 1
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    if cls != person_cls:
                        continue
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
        if not boxes:
            continue  # 无 person 标注的图跳过
        samples.append((img_path, boxes))
        if max_images and len(samples) >= max_images:
            break

    return Dataset(
        name="自定义YOLO",
        version="自定义",
        scope=f"自定义 YOLO 格式 person 类（{len(samples)} 张）",
        samples=samples,
    )


def _parse_usc_gt(xml_path: str) -> List[List[float]]:
    """USC gt.xml：<ObjectList><Object><Rect x= y= width= height=/></Object>..."""
    boxes = []
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return boxes
    for obj in tree.getroot().findall(".//Object"):
        rect = obj.find("Rect")
        if rect is None:
            continue
        x = float(rect.get("x", 0))
        y = float(rect.get("y", 0))
        w = float(rect.get("width", 0))
        h = float(rect.get("height", 0))
        if w > 0 and h > 0:
            boxes.append([x, y, x + w, y + h])
    return boxes


def load_usc(root: str, max_images: Optional[int] = None) -> Dataset:
    """USC 行人数据集 Set A/B/C：bmp 图片 + 同名 gt.xml。"""
    xmls = sorted(glob.glob(os.path.join(root, "**", "*.gt.xml"), recursive=True))
    if not xmls:
        raise FileNotFoundError(
            f"未找到 USC 标注: {root}\\**\\*.gt.xml")
    samples = []
    for xml_path in xmls:
        boxes = _parse_usc_gt(xml_path)
        if not boxes:
            continue
        stem = xml_path[:-7] if xml_path.endswith(".gt.xml") else os.path.splitext(xml_path)[0]  # xxx.gt.xml -> xxx
        img_path = stem + ".bmp"                       # 同目录（A/B）
        if not os.path.exists(img_path):
            # Set C：GT/xxx.gt.xml，图片在上级目录
            cand = os.path.join(os.path.dirname(os.path.dirname(xml_path)),
                                os.path.basename(stem) + ".bmp")
            if os.path.exists(cand):
                img_path = cand
        if not os.path.exists(img_path):
            continue
        samples.append((img_path, boxes))
        if max_images and len(samples) >= max_images:
            break
    if not samples:
        raise FileNotFoundError(f"USC 图片与标注未配对: {root}")
    return Dataset(
        name="USC",
        version="Set A+B+C",
        scope=f"USC Pedestrian A/B/C person 框（{len(samples)} 张，{sum(len(b) for _, b in samples)} 个实例）",
        samples=samples,
    )


def load_pennfudan(root: str, max_images: Optional[int] = None) -> Dataset:
    """Penn-Fudan Pedestrian：
        data/PennFudanPed/PennFudanPed/{Annotation,PNGImages,PedMasks}
    Annotation/*.txt 为 PASCAL 风格，解析 "Bounding box ... : (x1, y1) - (x2, y2)"。
    """
    ann_dir = os.path.join(root, "Annotation")
    img_dir = os.path.join(root, "PNGImages")
    if not os.path.isdir(ann_dir):
        root = os.path.join(root, "PennFudanPed")
        ann_dir = os.path.join(root, "Annotation")
        img_dir = os.path.join(root, "PNGImages")
    anns = sorted(glob.glob(os.path.join(ann_dir, "*.txt")))
    if not anns:
        raise FileNotFoundError(f"未找到 PennFudanPed 标注: {ann_dir}")

    box_re = re.compile(
        r"Bounding box for object \d+ \"[^\"]+\" "
        r"\(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)")
    samples = []
    for ann in anns:
        stem = os.path.splitext(os.path.basename(ann))[0]
        img_path = os.path.join(img_dir, stem + ".png")
        if not os.path.exists(img_path):
            continue
        boxes = []
        with open(ann, "r", encoding="utf-8") as f:
            for line in f:
                m = box_re.search(line)
                if m:
                    x1, y1, x2, y2 = map(int, m.groups())
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
        if boxes:
            samples.append((img_path, boxes))
        if max_images and len(samples) >= max_images:
            break

    return Dataset(
        name="Penn-Fudan",
        version="Pedestrian",
        scope=f"Penn-Fudan Pedestrian（{len(samples)} 张，{sum(len(b) for _, b in samples)} 个实例）",
        samples=samples,
    )


def load_landmarks(root: str, max_images: Optional[int] = None) -> Dataset:
    """NYC 景点数据集（YOLO txt 格式，直接平铺在 train/ 目录下）：
        root/train/*.jpg   ↔  root/train/*.txt
        txt 格式：cls_id xc yc w h（归一化坐标，cls_id 0-9 对应 10 种 NYC 景点）
    无 data.yaml，类别 ID 映射从 config.LANDMARK_CLASSES 读取。
    评测保留全部景点类框（GT 聚合计算 AP@0.5）。
    """
    img_dir = os.path.join(root, "train")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"未找到景点数据集图片目录: {img_dir}")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(img_dir, ext)))
    images = sorted(images)
    if not images:
        raise FileNotFoundError(f"景点数据集目录无图片: {img_dir}")

    # YOLO txt: cls_id xc yc w h (归一化)
    samples = []
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(img_dir, stem + ".txt")
        boxes = []
        if os.path.exists(label_path):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
        if boxes:
            samples.append((img_path, boxes))
        if max_images and len(samples) >= max_images:
            break

    if not samples:
        raise FileNotFoundError(f"景点数据集图片与标注未配对: {img_dir}")

    class_names = list(config.LANDMARK_CLASSES.values())
    cls_cn = "、".join(config.LANDMARK_CLASS_ZH.values())
    return Dataset(
        name="NYC 景点检测",
        version="纽约地标（944 图 / 10 类）",
        scope=f"NYC 景点类目标（类别: {cls_cn}，{len(samples)} 张，{sum(len(b) for _, b in samples)} 个实例）",
        samples=samples,
        classes=class_names,
    )


_SAMPLE_SEED = 42  # 固定随机抽样种子，保证同参数多次测评结果一致、可复现


def _subsample(ds: Dataset, max_images: Optional[int]) -> Dataset:
    """从全量数据集中随机抽样 max_images 张（固定种子，保持原顺序）。

    不采用"按文件名排序后截取前 N 张"：数据集内图片按名称排列时场景分布
    不均匀（难样本/密集行人集中在某段），顺序截取的小样本 AP 严重失真；
    随机抽样更能代表全量分布。抽样后同步更新 scope 中的张数/实例数描述。
    """
    n = ds.num_images
    if not max_images or n <= max_images:
        return ds
    rng = random.Random(_SAMPLE_SEED)
    idx = sorted(rng.sample(range(n), max_images))
    ds.samples = [ds.samples[i] for i in idx]
    ds.scope = re.sub(r"\d[\d,]*\s*张", f"{max_images}/{n} 张", ds.scope, count=1)
    ds.scope = re.sub(r"\d[\d,]*\s*个实例", f"{ds.num_person} 个实例",
                      ds.scope, count=1)
    return ds


def load_dataset(kind: str, root: str, max_images: Optional[int] = None) -> Dataset:
    """按 kind 分发加载（先全量加载，再统一随机抽样 max_images 张，0/None=全量）。"""
    kind = kind.lower()
    if kind in ("voc", "voc2007", "voc2007c"):
        ds = load_voc(root, prefixes=("2007te",) if kind != "voc" else None)
    elif kind == "voc2012":
        ds = load_voc(root, prefixes=("2012tv",),
                      ids_file=os.path.join(root, "voc2012_val_ids.txt"))
    elif kind == "custom":
        ds = load_custom_yolo(root)
    elif kind == "usc":
        ds = load_usc(root)
    elif kind == "pennfudan":
        ds = load_pennfudan(root)
    elif kind == "landmarks":
        ds = load_landmarks(root)
    else:
        raise ValueError(f"未知数据集类型: {kind}")
    return _subsample(ds, max_images)
