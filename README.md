# 🎯 目标检测系统

基于 YOLO 系列模型的目标检测精度测评与演示系统，提供 **人员检测** 与 **景点检测** 两类任务：

- **人员检测**：RT-DETR / YOLOv5 / YOLOv8 / YOLO11 共 8 个主流检测模型，在 VOC、USC、Penn-Fudan 数据集上测评 Person AP@0.5（合格判据 **>90%**）；支持上传任意图片进行行人识别。
- **景点检测**：3 个 NYC 地标检测模型，在 NYC Landmarks 数据集上测评 AP@0.5。

系统自带 Python 运行时（`runtime/`），**解压即用，无需安装任何环境**。

---

## 1. 快速开始（Windows）

1. 将项目解压到**纯英文路径**（如 `D:\object_detection`）；
2. **双击 `启动系统.bat`**——首次运行会自动在桌面创建「目标检测系统」图标并启动服务；
3. 等待约十秒，浏览器自动打开系统页面（未自动打开则手动访问 `http://localhost:8502`）；
4. 以后直接**双击桌面的「目标检测系统」图标**即可启动，无需再打开项目文件夹；
5. 局域网内其他机器访问 `http://<本机IP>:8502`。

> 提示：若移动了项目文件夹导致桌面图标失效，双击文件夹里的 `启动系统.bat` 一次，会自动重建图标。

关闭服务：在弹出的命令行窗口按 `Ctrl+C` 或直接关闭窗口。

> 启动脚本会自动完成 CUDA 环境引导（有 NVIDIA 显卡时自动启用 GPU，无显卡安全回退 CPU），无需任何手动配置。

---

## 2. 使用流程

### 2.1 数据集精度测评

1. 左侧栏「🎯 检测任务」选择 **人员检测** 或 **景点检测**；
2. 「⚙️ 测评配置」中选择数据集、待测模型（单选）、推理设备（auto 自动选 GPU）；
3. **未开始测评时**，主区「📊 综合测试结果」展示所选模型在该数据集上的全量测评综合指标：醒目标注当前模型与数据集，AP@0.5 / Recall / Precision 数字卡片；下方为同数据集全部模型的指标对比明细表（模型名完整、按 AP@0.5 从高到低排序、达标绿色加粗），供选型参考；
4. 「测评图片数上限」：`0` = 全量测评（正式达标判定口径）；`>100` 时从全量随机抽样（固定种子、可复现）。**下限 100 张**——样本过小 AP 统计波动过大，系统会自动纠正；
5. 点击「① 加载数据集」→「② 开始测评」，可随时「⏹ 取消测评」；开始测评后页面切换为测评视图（综合指标参考区自动隐藏）；
6. 测评完成后查看核心指标（**AP@0.5 / Recall / Precision / 速度**）与「📸 检测结果可视化」：
   - **搜索 + 下拉 + 翻页**：搜索框输入文件名关键词快速过滤；下拉框直接跳转指定图片；「◀ 上一张 / 下一张 ▶」按钮连续浏览；右侧小字显示「共 X 张」（过滤时额外标注「匹配 N」）；
   - **原图 / 结果左右对照**：左侧为无框原始图像，右侧为检测结果图——红框（真实标注）+ 绿框（模型预测，附置信度），两图并排直接对比；
7. 「📄 报告导出」下载 HTML 测评报告或 CSV 结果表。

> 所有功能在一屏内完成，无需滚动页面；切换任务、数据集、模型或重新测评时自动清空旧结果。

### 2.2 自定义图片识别（仅人员检测）

1. 检测任务选「🚶 人员检测」，数据集类型选「自定义上传」；
2. 上传一张或多张图片（jpg/png/bmp 等，**仅存于浏览器会话内存，不写入磁盘**）；
3. 点击「② 开始识别上传图片」；
4. 结果页左侧原图、右侧检测结果对比展示，多图可左右切换。

> 景点任务不提供自定义上传：景点检测模型面向 10 类纽约地标，任意上传图片不在其检测范围内。

---

## 3. 测评口径

| 项目 | 口径 |
|---|---|
| 核心指标 | **Person AP@0.5**（COCO 101 点插值，>90% 判定达标，绿色标注）；同时计算 VOC 11 点插值 |
| 匹配规则 | 预测按置信度降序，与真实框贪心匹配，**IoU ≥ 0.5** 记为 TP |
| 置信度阈值 | 固定 **0.25**（ultralytics 标准部署口径，前端不暴露） |
| NMS IoU | 固定 0.45 |
| 推理分辨率 | 默认 640 |
| 随机抽样 | 固定种子 42，同参数结果可复现 |

---

## 4. 数据集与模型

### 数据集（已随项目提供）

| 任务 | 数据集 | 图片数 | 实例数 | 目录 |
|---|---|---|---|---|
| 人员 | VOC2007 test | 1,996 | 4,990 | `data/voc/` |
| 人员 | VOC2012 val（官方划分） | 2,145 | 4,928 | `data/voc/` |
| 人员 | USC Pedestrian (Set A+B+C) | 359 | 816 | `data/usc/` |
| 人员 | Penn-Fudan Pedestrian | 170 | 423 | `data/PennFudanPed/` |
| 景点 | NYC Landmarks（10 类纽约地标） | 944 | 1,229 | `data/archive/nyc_landmarks/` |

### 模型（权重已预置在 `models/`）

| 模型 | 权重文件 | 任务 |
|---|---|---|
| RT-DETR-l | rtdetr-l.pt | 人员 |
| YOLOv5l / YOLOv5x | yolov5lu.pt / yolov5xu.pt | 人员 |
| YOLOv8m / YOLOv8x | yolov8m.pt / yolov8x.pt | 人员 |
| YOLO11m / YOLO11l / YOLO11x | yolo11m/l/x.pt | 人员 |
| YOLO11n-景点 / YOLOv8n-景点 / YOLOv5n-景点 | landmarks_*.pt（NYC 地标检测） | 景点 |

---

## 5. 目录结构

```
object_detection/
├── 启动系统.bat            # ★ 一键启动（首次运行自动创建桌面图标，解压即用）
├── streamlit_app.py        # Streamlit Web 界面（主程序）
├── config.py               # 全局配置：模型清单、数据集、测评参数、CUDA 引导
├── assets/                 # 应用图标等资源
├── core/
│   ├── metrics.py          #   IoU 匹配、PR 曲线、AP@0.5（VOC 11 点 / COCO 101 点）
│   ├── dataset_loader.py   #   多数据集加载与随机抽样
│   ├── evaluator.py        #   YOLO 推理评估器
│   ├── benchmark_data.py   #   全量测评综合指标（界面固定展示用）
│   ├── visualize.py        #   检测框绘制（红=GT，绿=预测+置信度）
│   └── reporter.py         #   HTML/CSV 报告生成
├── models/                 # 模型权重（已随项目提供，缺失时 ultralytics 自动下载）
├── data/                   # 测评数据集（见第 4 节）
├── reports/                # 测评报告输出目录
├── tests/                  # 核心模块自测（python tests/test_core.py）
├── runtime/                # 自包含 Python 运行时（勿增删包）
└── requirements.txt        # 自行搭建环境时的依赖清单
```

---

## 6. 常见问题

**Q：启动后页面打不开？**
确认命令行窗口无报错；本机防火墙需放行 8502 端口；浏览器访问 `http://localhost:8502`。

**Q：为什么没有用上 GPU？**
侧边栏「推理设备」下方会显示真实检测结果。auto 模式下有可用 NVIDIA 显卡自动启用 GPU；本机无 N 卡时安全回退 CPU，不影响功能。

---

## 7. 自行搭建环境（不使用 runtime 时）

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
python -m streamlit run streamlit_app.py --server.port 8502 --server.fileWatcherType none
```

> ⚠️ 启动参数必须带 `--server.fileWatcherType none`，否则运行中编辑文件可能损坏模块导入。
