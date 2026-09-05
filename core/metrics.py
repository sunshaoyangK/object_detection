# -*- coding: utf-8 -*-
"""
AP@0.5 指标计算模块（与 pycocotools 解耦，纯 numpy 实现）。

支持：
- IoU 计算
- 预测框与真实框在 IoU>=0.5 条件下的贪心匹配（置信度降序）
- Precision-Recall 曲线
- VOC 11-point AP 与 COCO 101-point AP（All-point，梯形积分）
"""
import numpy as np


def compute_iou(box1, box2):
    """两个 xyxy 框的 IoU。"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_detections(all_gt, all_pred, iou_thr=0.5):
    """
    全局贪心匹配。

    参数:
        all_gt : list[list[xyxy]]         每张图的真实 person 框
        all_pred: list[list[xyxy + conf]] 每张图的预测框（已按 person 过滤）

    返回:
        tp, fp, num_gt
        tp/fp 按全局置信度降序排列（长度 = 预测总数），num_gt 为总真实框数。
    """
    items = []  # (conf, image_idx, box)
    for img_idx, preds in enumerate(all_pred):
        for p in preds:
            items.append((float(p[4]), img_idx, [float(v) for v in p[:4]]))
    items.sort(key=lambda x: -x[0])

    matched = [set() for _ in range(len(all_gt))]
    tp, fp = [], []
    for conf, img_idx, box in items:
        gts = all_gt[img_idx]
        best_iou, best_gi = iou_thr, -1
        for gi, gt in enumerate(gts):
            if gi in matched[img_idx]:
                continue
            iou = compute_iou(box, gt)
            if iou > best_iou:  # 严格大于阈值
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            matched[img_idx].add(best_gi)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    num_gt = sum(len(g) for g in all_gt)
    return tp, fp, num_gt


def pr_curve(tp, fp, num_gt):
    """由 tp/fp 累计序列计算 PR 曲线（recall, precision），首点补 (0, 1)。"""
    tp_cum = np.cumsum(tp, dtype=np.float64)
    fp_cum = np.cumsum(fp, dtype=np.float64)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return np.concatenate(([0.0], recall)), np.concatenate(([1.0], precision))


def _precision_envelope(recall, precision):
    """
    计算单调不增的最大精度包络线（VOC/COCO 标准做法）。
    返回按 recall 升序排列的 (recall, envelope_precision)。
    """
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    # 从高 recall 向低 recall 累积最大值
    env = np.maximum.accumulate(p[::-1])[::-1]
    return r, env


def ap_11point(recall, precision):
    """
    VOC 2007 风格：11 个 recall 阈值 (0, 0.1, ..., 1.0) 的均值。
    每个阈值处取 "recall >= 阈值" 中的最大 precision。
    """
    _, env = _precision_envelope(recall, precision)
    ap = 0.0
    for t in np.arange(0.0, 1.0001, 0.1):
        mask = recall >= t
        ap += float(np.max(env[mask])) if mask.any() else 0.0
    return ap / 11.0


def ap_allpoint(recall, precision, n_points=101):
    """
    COCO/VOC2010+ 风格：在 [0,1] 上取 101 个等距 recall 阈值，
    对精度包络线做插值后取平均（等价于梯形积分）。
    """
    r, env = _precision_envelope(recall, precision)
    thresholds = np.linspace(0.0, 1.0, n_points)
    # 左端点 (recall=0) 处精度为 1（首点已补），右侧用包络插值
    ps = np.interp(thresholds, r, env, left=float(env[0]), right=0.0)
    return float(np.mean(ps))


def compute_ap50(all_gt, all_pred, iou_thr=0.5):
    """
    一站式计算 Person AP@0.5。

    返回 dict:
        ap_voc       VOC 11-point AP@0.5
        ap_coco      COCO 101-point AP@0.5
        recall / precision / num_gt / num_pred / tp / fp
    """
    tp, fp, num_gt = match_detections(all_gt, all_pred, iou_thr=iou_thr)
    if not tp:
        r = np.array([0.0])
        p = np.array([1.0])
    else:
        r, p = pr_curve(tp, fp, num_gt)
    return {
        "ap_voc": ap_11point(r, p),
        "ap_coco": ap_allpoint(r, p),
        "recall": r,
        "precision": p,
        "num_gt": num_gt,
        "num_pred": len(tp),
        "tp": tp,
        "fp": fp,
    }
