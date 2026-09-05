# -*- coding: utf-8 -*-
"""
YOLO 推理评估器。

流程：
    加载 ultralytics YOLO 模型 -> 对数据集逐批推理（仅保留 person 类预测）
    -> 与真实框做 IoU>=0.5 贪心匹配 -> 计算 Person AP@0.5 (VOC/COCO 两种口径)
"""
import os
import shutil
import time
from typing import Callable, List, Optional

from ultralytics import YOLO

import config
from . import metrics
from .dataset_loader import Dataset

ProgressCb = Optional[Callable[[int, int, str], None]]


def resolve_model_path(model_name: str) -> str:
    """解析模型权重路径，不存在时自动下载到 models/ 目录。

    查找顺序：
    1. 项目根相对路径直接存在（如 models/yolov8x.pt）；
    2. models/ 目录下（模型权重）；
    3. 均不存在时触发 ultralytics 自动下载并落到 models/ 目录。
    """
    file = config.MODEL_CONFIGS[model_name]["file"]
    root_rel = os.path.join(config.BASE_DIR, file)
    if os.path.exists(root_rel):
        return root_rel
    local = os.path.join(config.MODELS_DIR, file)
    if os.path.exists(local):
        return local
    # 触发 ultralytics 自动下载（下载到当前工作目录）
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    YOLO(file)
    src = os.path.join(os.getcwd(), file)
    if os.path.exists(src):
        shutil.move(src, local)
    return local


class Evaluator:
    def __init__(self, model_name: str, device: str = "0",
                 keep_all_classes: bool = False):
        self.model_name = model_name
        self.device = device
        # keep_all_classes=True 时保留模型全部类别预测（供景点检测等
        # 多类模型测评使用，GT 聚合口径与统一匹配规则一致）
        self.keep_all_classes = keep_all_classes
        path = resolve_model_path(model_name)
        self.model = YOLO(path)

    def evaluate(
        self,
        dataset: Dataset,
        conf: float = config.DEFAULT_CONF,
        nms_iou: float = config.DEFAULT_IOU,
        progress_cb: ProgressCb = None,
        cancel_event: Optional[Callable[[], bool]] = None,
        sample_paths: Optional[List[str]] = None,
        imgsz: Optional[int] = None,
    ) -> dict:
        """
        对数据集执行完整测评。

        参数:
            sample_paths: 需要额外保存预测框的图片路径列表（用于结果可视化）。
            imgsz: 推理分辨率，None 时用 config.DEFAULT_IMGSZ
                   （宽幅小目标数据集由 config.DATASET_IMGSZ 指定更高分辨率）。

        返回 dict:
            model, device, conf, nms_iou,
            ap_voc, ap_coco, recall, precision,
            num_gt, num_pred, num_images,
            avg_time_ms, fps, total_time
            sample_preds: {path: [[x1,y1,x2,y2,conf], ...]} 仅 sample_paths 中的图
        """
        if imgsz is None:
            imgsz = config.DEFAULT_IMGSZ
        all_gt = [b for _, b in dataset.samples]
        paths = [p for p, _ in dataset.samples]
        all_pred: List[List[List[float]]] = []

        sample_set = set(sample_paths or [])
        sample_preds = {}

        t0 = time.time()
        total = len(paths)
        done = 0
        for start in range(0, total, config.BATCH_SIZE):
            if cancel_event and cancel_event():
                break
            batch = paths[start:start + config.BATCH_SIZE]
            results = self.model(
                batch,
                conf=conf,
                iou=nms_iou,
                imgsz=imgsz,
                verbose=False,
                device=config.resolve_device(self.device),
            )
            for r, path in zip(results, batch):
                preds = []
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for b in boxes:
                        cls = int(b.cls.item())
                        if not self.keep_all_classes and cls != config.PERSON_CLASS_ID:
                            continue  # 只统计 person（景点检测模型保留全部类别）
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        c = float(b.conf.item())
                        preds.append([x1, y1, x2, y2, c])
                all_pred.append(preds)
                if path in sample_set:
                    sample_preds[path] = preds
            done += len(batch)
            if progress_cb:
                progress_cb(done, total, self.model_name)

        total_time = time.time() - t0
        avg_ms = (total_time / done) * 1000 if done else 0

        ap = metrics.compute_ap50(all_gt, all_pred, iou_thr=config.MATCH_IOU)

        return {
            "model": self.model_name,
            "device": self.device,
            "conf": conf,
            "nms_iou": nms_iou,
            "ap_voc": ap["ap_voc"],
            "ap_coco": ap["ap_coco"],
            "recall": ap["recall"],
            "precision": ap["precision"],
            "num_gt": ap["num_gt"],
            "num_pred": ap["num_pred"],
            "num_images": done,
            "avg_time_ms": avg_ms,
            "fps": done / total_time if total_time > 0 else 0.0,
            "total_time": total_time,
            "matched": sum(ap["tp"]),
            "sample_preds": sample_preds,
        }

    def predict_images(
        self,
        images: dict,
        conf: float = config.DEFAULT_CONF,
        nms_iou: float = config.DEFAULT_IOU,
        progress_cb: ProgressCb = None,
        cancel_event: Optional[Callable[[], bool]] = None,
        imgsz: Optional[int] = None,
    ) -> dict:
        """对内存图片做纯推理（无 GT，自定义上传识别 / 测试集选图检测用），图片不落盘。

        images: {名称: PIL.Image}
        返回 {名称: [[x1,y1,x2,y2,conf], ...]}；
        人员模型仅保留 person 类预测，景点模型保留全部类别预测。"""
        if imgsz is None:
            imgsz = config.DEFAULT_IMGSZ
        preds_by_name = {}
        names = list(images.keys())
        total = len(names)
        done = 0
        for start in range(0, total, config.BATCH_SIZE):
            if cancel_event and cancel_event():
                break
            batch_names = names[start:start + config.BATCH_SIZE]
            batch = [images[n] for n in batch_names]
            results = self.model(
                batch,
                conf=conf,
                iou=nms_iou,
                imgsz=imgsz,
                verbose=False,
                device=config.resolve_device(self.device),
            )
            for r, name in zip(results, batch_names):
                preds = []
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for b in boxes:
                        if not self.keep_all_classes and int(b.cls.item()) != config.PERSON_CLASS_ID:
                            continue  # 人员模型只保留 person；景点模型保留全部类别
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        preds.append([x1, y1, x2, y2, float(b.conf.item())])
                preds_by_name[name] = preds
            done += len(batch)
            if progress_cb:
                progress_cb(done, total, self.model_name)
        return preds_by_name

    def get_model_info(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
        }
