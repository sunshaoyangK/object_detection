# -*- coding: utf-8 -*-
"""
核心模块自测（不依赖 torch/ultralytics）：
    1. metrics：IoU / 匹配 / PR / AP 计算正确性
    2. dataset_loader：合成 YOLO 格式数据加载

运行：python tests/test_core.py
"""
import os
import shutil
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dataset_loader, metrics  # noqa: E402


def test_iou():
    a = [0, 0, 10, 10]
    b = [0, 0, 10, 10]
    assert abs(metrics.compute_iou(a, b) - 1.0) < 1e-9
    assert abs(metrics.compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) - 0.0) < 1e-9
    # 半重叠：面积 100，交 50 -> IoU = 50/(100+100-50)=0.3333
    assert abs(metrics.compute_iou([0, 0, 10, 10], [0, 5, 10, 15]) - 50 / 150) < 1e-6
    print("[PASS] test_iou")


def test_ap_perfect():
    # 2 张图，各 1 个 GT，预测完全重合且置信度高
    all_gt = [[[0, 0, 10, 10]], [[0, 0, 10, 10]]]
    all_pred = [[[0, 0, 10, 10, 0.9]], [[0, 0, 10, 10, 0.8]]]
    res = metrics.compute_ap50(all_gt, all_pred, iou_thr=0.5)
    assert res["ap_voc"] == 1.0, res["ap_voc"]
    assert res["ap_coco"] == 1.0, res["ap_coco"]
    assert res["num_gt"] == 2
    print("[PASS] test_ap_perfect ap_voc=%.4f ap_coco=%.4f" % (res["ap_voc"], res["ap_coco"]))


def test_ap_none():
    # 无检测时：VOC 11 点仅首点 (0,1) 对 t=0 贡献 1/11；COCO 101 点贡献 1/101（与 pycocotools 标准行为一致）
    res = metrics.compute_ap50([[[0, 0, 10, 10]]], [])
    assert abs(res["ap_voc"] - 1.0 / 11.0) < 1e-9, res["ap_voc"]
    assert abs(res["ap_coco"] - 1.0 / 101.0) < 1e-9, res["ap_coco"]
    print("[PASS] test_ap_none")


def test_ap_multi():
    # 2 GT + 2 TP + 2 FP，验证 11 点与 101 点 AP 手算值
    # 排序: TP(0.9) R=0.5 P=1.0 | FP(0.8) R=0.5 P=0.5 | TP(0.7) R=1.0 P=2/3 | FP(0.6) R=1.0 P=0.5
    # 包络后 11 点: t=0,0.1..0.5 -> 1.0（5个）; t=0.6..1.0 -> 2/3（5个）
    # ap_voc = (1 + 5*1 + 5*2/3)/11 = 28/33 ≈ 0.84848
    all_gt = [[[0, 0, 10, 10], [20, 20, 30, 30]]]
    all_pred = [[
        [0, 0, 10, 10, 0.9],   # TP
        [50, 50, 60, 60, 0.8], # FP
        [20, 20, 30, 30, 0.7], # TP
        [70, 70, 80, 80, 0.6], # FP
    ]]
    res = metrics.compute_ap50(all_gt, all_pred, iou_thr=0.5)
    expected_voc = 28.0 / 33.0
    assert abs(res["ap_voc"] - expected_voc) < 1e-6, f'{res["ap_voc"]} != {expected_voc}'
    # 101 点: t=0(1.0) + 0.01..0.49(49个=1.0) + 0.50(右侧端点 2/3)
    #        + 0.51..0.99(49个=2/3, 区间端点包络) + 1.00(=0.5)
    expected_coco = (1.0 + 49 + 2.0 / 3 + 49 * (2.0 / 3) + 0.5) / 101.0
    assert abs(res["ap_coco"] - expected_coco) < 1e-4, f'{res["ap_coco"]} != {expected_coco}'
    print("[PASS] test_ap_multi ap_voc=%.4f (期望 %.4f) ap_coco=%.4f (期望 %.4f)"
          % (res["ap_voc"], expected_voc, res["ap_coco"], expected_coco))


def test_synthetic_custom():
    tmp_dir = tempfile.mkdtemp(prefix="yolo_test_")
    try:
        os.makedirs(os.path.join(tmp_dir, "images"))
        os.makedirs(os.path.join(tmp_dir, "labels"))
        Image.new("RGB", (200, 100), (255, 0, 0)).save(os.path.join(tmp_dir, "images", "a.jpg"))
        # person(0) 中心(100,50) 尺寸(40,60) -> [80,20,120,80]; 另一类(1) 应过滤
        with open(os.path.join(tmp_dir, "labels", "a.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.6\n1 0.9 0.1 0.1 0.1\n")
        ds = dataset_loader.load_custom_yolo(tmp_dir)
        assert ds.num_images == 1 and ds.num_person == 1
        bx = ds.samples[0][1][0]
        assert abs(bx[0] - 80) < 1e-6 and abs(bx[3] - 80) < 1e-6, bx
        print("[PASS] test_synthetic_custom", bx)
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_iou()
    test_ap_perfect()
    test_ap_none()
    test_ap_multi()
    test_synthetic_custom()
    print("\n全部核心模块自测通过 ✔")
