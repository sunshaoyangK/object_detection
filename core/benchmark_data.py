# -*- coding: utf-8 -*-
"""
各模型在各测评数据集上的全量测评综合指标。

数据来源：全量测评记录（AP@0.5 / Recall / Precision，均为 0-100 百分比），
按 检测任务(person/landmark) → 数据集 kind（与 config.DATASET_DIRS 的 key 一致）
→ 模型名称（与 config.MODEL_CONFIGS 的 key 一致）组织。

界面在固定位置展示当前所选「模型 + 数据集」的整体测评结果，
并提供同一数据集下各模型的横向对比。
"""

# 每条记录：(AP@0.5%, Recall%, Precision%)
BENCHMARK = {
    "person": {
        "usc": {
            "name": "USC",
            "models": {
                "RT-DETR-l": (97.75, 99.02, 78.98),
                "YOLOv5l":   (97.87, 99.02, 86.88),
                "YOLOv5x":   (98.19, 99.02, 86.88),
                "YOLOv8m":   (97.08, 98.90, 87.43),
                "YOLOv8x":   (96.77, 98.65, 87.50),
                "YOLO11m":   (97.20, 98.90, 87.24),
                "YOLO11l":   (98.22, 99.14, 87.08),
                "YOLO11x":   (96.95, 98.90, 86.59),
            },
        },
        "pennfudan": {
            "name": "Penn-Fudan",
            "models": {
                "RT-DETR-l": (97.67, 100.00, 41.96),
                "YOLOv5l":   (96.08, 99.76, 60.03),
                "YOLOv5x":   (96.15, 99.53, 59.38),
                "YOLOv8m":   (95.45, 99.53, 60.32),
                "YOLOv8x":   (96.04, 99.76, 60.46),
                "YOLO11m":   (95.95, 99.53, 60.58),
                "YOLO11l":   (96.25, 99.76, 60.20),
                "YOLO11x":   (96.16, 99.76, 59.77),
            },
        },
        "voc2007c": {
            "name": "VOC2007",
            "models": {
                "RT-DETR-l": (92.42, 96.68, 55.06),
                "YOLOv5l":   (90.17, 93.16, 78.57),
                "YOLOv5x":   (90.39, 93.51, 79.26),
                "YOLOv8m":   (90.12, 93.16, 78.56),
                "YOLOv8x":   (90.25, 93.58, 78.38),
                "YOLO11m":   (90.25, 93.00, 79.38),
                "YOLO11l":   (90.20, 93.39, 78.48),
                "YOLO11x":   (90.17, 93.60, 78.38),
            },
        },
        "voc2012": {
            "name": "VOC2012",
            "models": {
                "RT-DETR-l": (93.19, 96.28, 58.63),
                "YOLOv5l":   (90.26, 92.32, 84.31),
                "YOLOv5x":   (91.23, 93.21, 84.82),
                "YOLOv8m":   (90.11, 92.26, 84.36),
                "YOLOv8x":   (91.13, 93.41, 83.55),
                "YOLO11m":   (90.20, 92.30, 84.85),
                "YOLO11l":   (91.10, 93.21, 84.17),
                "YOLO11x":   (92.04, 94.05, 84.01),
            },
        },
    },
    "landmark": {
        "landmarks": {
            "name": "NYC 景点",
            "models": {
                "YOLO11n-景点": (96.91, 97.40, 93.44),
                "YOLOv8n-景点": (96.69, 97.56, 93.97),
                "YOLOv5n-景点": (96.91, 97.40, 93.44),
            },
        },
    },
}


def get_dataset_benchmark(task_key: str, kind: str):
    """返回某任务下某数据集的综合测评块 {name, models:{model:(ap,rec,prec)}}，无记录返回 None。"""
    return BENCHMARK.get(task_key, {}).get(kind)


def get_model_metrics(task_key: str, kind: str, model_name: str):
    """返回某模型在某数据集上的 (AP@0.5, Recall, Precision) 百分比元组，无记录返回 None。"""
    block = get_dataset_benchmark(task_key, kind)
    if block is None:
        return None
    return block["models"].get(model_name)
