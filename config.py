# -*- coding: utf-8 -*-
"""
全局配置：模型清单、数据集目录、测评参数。

注意：本模块被 streamlit_app 首行导入。为实现"有 N 卡的机器一启动就能
自动识别 GPU、无需用户手动改任何设置"，所有 CUDA 环境启动引导
（DLL 搜索路径、CUDA_VISIBLE_DEVICES 等）必须放在文件最顶部、
且在 import torch 之前完成。
"""
import os
import functools


def _bootstrap_cuda_env() -> None:
    """在任何 import torch 之前做的一次性 CUDA 环境启动引导。

    解决"机器有 NVIDIA 显卡，但 torch.cuda.is_available() 仍为 False"
    的三大常见场景，使 GPU 开箱即用、无需用户修改任何配置：

      1. Windows 笔记本 Optimus 双显卡（AMD/Intel 核显 + NVIDIA 独显）：
         Python.exe 默认跑在核显上，NVML/nvidia-smi 能看到卡但 torch
         初始化时默认枚举不到。解法：提前设置 CUDA_VISIBLE_DEVICES=0
         强制 CUDA 运行时只看 NVIDIA 设备（若本机无 N 卡此变量被忽略，
         不会影响 CPU-only 机器）。
      2. CUDA 运行时 DLL 搜索失败：torch\lib 里自带了 cudart/cublas/cudnn 等
         30+ 个 DLL，但部分精简 Windows 系统里 site-packages 的 DLL 搜索
         不生效，需手动 os.add_dll_directory 并加到 PATH 头。
      3. 旧版 torch 会在导入时提前读 CUDA_VISIBLE_DEVICES，确保此函数
         在 import torch 之前调用即可。
    """
    # 1) CUDA_VISIBLE_DEVICES：强制只枚举第 0 块 NVIDIA 卡，
    #    绕过 Optimus 路由。对无 N 卡机器无副作用（CUDA 会忽略该变量）。
    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 2) torch\lib DLL 搜索：同时处理两种常见路径结构
    #    (a) 正常安装：<prefix>/Lib/site-packages/torch/lib
    #    (b) 开发/脚本：<python_exe_dir>/Lib/site-packages/torch/lib
    import sys as _sys
    candidates = []
    if getattr(_sys, "executable", None):
        candidates.append(
            os.path.join(os.path.dirname(_sys.executable),
                         "Lib", "site-packages", "torch", "lib")
        )
    candidates.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Lib", "site-packages", "torch", "lib"))
    # 如果 config.py 就在 site-packages 里（pip 安装态），再加一条
    try:
        import site as _site
        for sp in getattr(_site, "getsitepackages", lambda: [])():
            candidates.append(os.path.join(sp, "torch", "lib"))
    except Exception:  # noqa: BLE001
        pass

    for dll_dir in dict.fromkeys(candidates):  # 去重保序
        if not dll_dir or not os.path.isdir(dll_dir):
            continue
        # Win10+ add_dll_directory：比 PATH 更可靠，不会被 PATH 污染影响
        try:
            _os_add = getattr(os, "add_dll_directory", None)
            if callable(_os_add):
                _os_add(dll_dir)
        except (OSError, FileNotFoundError):
            pass
        # 同时塞到 PATH 头，兼容一些旧版本 torch 子进程调用
        old_path = os.environ.get("PATH", "")
        if dll_dir.lower() not in (p.lower() for p in old_path.split(os.pathsep)):
            os.environ["PATH"] = dll_dir + os.pathsep + old_path


# 启动时立即生效（必须在任何 import torch 之前）
_bootstrap_cuda_env()


# ---------------- 路径 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ---------------- 模型清单 ----------------
# key: 展示名称; value: 权重文件（相对项目根；不存在时由 ultralytics 自动下载）
MODEL_CONFIGS = {
    "RT-DETR-l": {"file": "rtdetr-l.pt", "desc": "DETR 架构，精度高", "task": "person"},
    "YOLOv5l": {"file": "yolov5lu.pt", "desc": "经典 v5 精度高", "task": "person"},
    "YOLOv5x": {"file": "yolov5xu.pt", "desc": "经典 v5 最高精度", "task": "person"},
    "YOLOv8m": {"file": "yolov8m.pt", "desc": "v8 精度均衡", "task": "person"},
    "YOLOv8x": {"file": "yolov8x.pt", "desc": "v8 系列最高精度", "task": "person"},
    "YOLO11m": {"file": "yolo11m.pt", "desc": "新一代均衡", "task": "person"},
    "YOLO11l": {"file": "yolo11l.pt", "desc": "新一代精度高", "task": "person"},
    "YOLO11x": {"file": "yolo11x.pt", "desc": "新一代最高精度", "task": "person"},
    "YOLO11n-景点": {"file": "landmarks_yolo11n.pt", "desc": "NYC 景点 10 类地标专项检测", "task": "landmark"},
    "YOLOv8n-景点": {"file": "landmarks_yolov8n.pt", "desc": "NYC 景点 10 类地标专项检测", "task": "landmark"},
    "YOLOv5n-景点": {"file": "landmarks_yolov5nu.pt", "desc": "NYC 景点 10 类地标专项检测", "task": "landmark"},
}
DEFAULT_MODELS = [
    "RT-DETR-l", "YOLOv5l", "YOLOv5x", "YOLOv8m", "YOLOv8x",
    "YOLO11m", "YOLO11l", "YOLO11x",
    "YOLO11n-景点", "YOLOv8n-景点", "YOLOv5n-景点",
]
# 前端首次进入时默认选中的单个模型
DEFAULT_MODEL = "YOLOv8x"

# COCO 中 person 类别在模型输出中的 class id = 0
PERSON_CLASS_ID = 0

# NYC 景点数据集标注类别（YOLO txt 格式，目录 data/archive/nyc_landmarks/train/）
# 10 个类别 ID 由文件名 + 标注统计推断得出（数据集本身无 data.yaml）
LANDMARK_CLASSES = {
    0: "empire_state_building",    # 帝国大厦
    1: "1_world_trade_center",     # 世贸中心一号楼
    2: "432_park_ave",             # 公园大道432号
    3: "united_nations_building",   # 联合国总部大厦
    4: "flatiron",                 # 熨斗大厦
    5: "brooklyn_bridge",          # 布鲁克林大桥
    6: "chrysler_building",        # 克莱斯勒大厦
    7: "metlife_building",         # 大都会人寿大厦
    8: "statue_of_liberty",        # 自由女神像
    9: "30_hudson_yards",          # 哈德逊广场
}
LANDMARK_CLASS_ZH = {
    0: "帝国大厦", 1: "世贸中心一号楼", 2: "公园大道432号", 3: "联合国总部大厦",
    4: "熨斗大厦", 5: "布鲁克林大桥", 6: "克莱斯勒大厦", 7: "大都会人寿大厦",
    8: "自由女神像", 9: "哈德逊广场",
}

# ---------------- 数据集配置 ----------------
# kind -> (名称, 数据根目录, 说明)；前端开放的测评数据集 + 自定义上传
DATASET_DIRS = {
    "voc2007c": {"name": "VOC2007", "dir": os.path.join(DATA_DIR, "voc"), "version": "", "task": "person"},
    "voc2012": {"name": "VOC2012", "dir": os.path.join(DATA_DIR, "voc"), "version": "", "task": "person"},
    "pennfudan": {"name": "Penn-Fudan", "dir": os.path.join(DATA_DIR, "PennFudanPed"), "version": "", "task": "person"},
    "usc": {"name": "USC", "dir": os.path.join(DATA_DIR, "usc"), "version": "", "task": "person"},
    "landmarks": {"name": "NYC 景点", "dir": os.path.join(DATA_DIR, "archive", "nyc_landmarks"), "version": "", "task": "landmark"},
    "custom": {"name": "自定义上传", "dir": os.path.join(DATA_DIR, "custom_upload"), "version": "", "task": "custom"},
}

# ---------------- 测评参数 ----------------
# conf=0.25 部署口径（ultralytics 标准默认），Precision 贴近真实部署效果。
# 前端不暴露 conf 修改接口，统一用此值保证测评口径一致。
DEFAULT_CONF = 0.25      # 置信度阈值（固定，不在前端暴露）
DEFAULT_IOU = 0.45       # NMS IoU
MATCH_IOU = 0.5          # 匹配 IoU 阈值（测评表规定）
MAX_SUBSET = 5000        # 子集数量上限
BATCH_SIZE = 8           # 推理批次

# ---------------- 推理分辨率 ----------------
# 默认 640；如个别数据集有特殊需要（宽幅小目标），可在 DATASET_IMGSZ 中按 kind 指定，
# 属于后端统一测评口径的一部分，前端不暴露。
DEFAULT_IMGSZ = 640
DATASET_IMGSZ = {}


def _cuda_available() -> bool:
    """检测当前环境是否真的有可用 CUDA GPU（torch 装的是 CPU 版或无显卡都返回 False）。"""
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=1)
def _has_nvidia_gpu() -> bool:
    """检测本机是否存在物理 NVIDIA 显卡（与 torch 是否带 CUDA 无关，仅供提示用）。
    Windows 用 PowerShell 查询，失败时返回 False（不影响推理）。"""
    import sys
    if sys.platform != "win32":
        return False
    try:
        import subprocess
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=10,
        )
        return "nvidia" in (r.stdout or "").lower()
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=1)
def device_status() -> tuple:
    """返回 (是否可用GPU, 说明文字) 供界面显示真实设备状态。
    多分支精准诊断，避免"有 N 卡但提示不对"的情况让用户一头雾水。
    结果做 lru_cache，避免 Streamlit 每次刷新都重复检测。
    """
    import sys
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return False, f"CPU（torch 导入失败：{type(e).__name__}）"

    # ---- 分支 1：torch 能正常使用 CUDA ----
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = ""
        return True, f"GPU（CUDA{', ' + name if name else ''}）"

    # ---- 分支 2：torch 报告 CUDA 不可用，深挖原因 ----
    has_nvidia = _has_nvidia_gpu()
    torch_is_cuda_build = getattr(torch.version, "cuda", None) is not None

    # 2a：torch 是 CPU 版（常见于未通过项目 runtime 环境启动，误用了系统其他 Python）
    if not torch_is_cuda_build:
        exe = sys.executable or ""
        if has_nvidia:
            hint = "检测到 NVIDIA GPU，但你运行的 Python 装的是 CPU 版 torch"
            if "runtime" not in exe.lower() and "venv" not in exe.lower():
                hint += "；请通过项目自带「启动系统.bat」启动（使用 runtime\python.exe）"
            return False, f"CPU（{hint}）"
        return False, "CPU（torch 为 CPU 版，无法使用 GPU）"

    # 2b：torch 确实是 CUDA 构建（+cu12x），且检测到有 NVIDIA 硬件
    #     → 这才是"本该能用上 GPU 却没成功"的异常场景，需要展开诊断
    if has_nvidia:
        init_err = ""
        try:
            torch.cuda.init()
        except Exception as e:  # noqa: BLE001
            init_err = str(e).replace("\n", " ").strip()
        low = init_err.lower()
        if "no nvidia driver" in low:
            # 驱动不可访问：典型场景：笔记本被 Windows 图形设置强制跑核显，
            # 或 NVML 驱动服务没起来。给具体可操作的建议，且含中文不含英文报错。
            return False, (
                "CPU（检测到 NVIDIA GPU，但 CUDA 驱动不可用。请依次尝试："
                "① 关闭所有 Python/模拟器窗口，通过本项目「启动系统.bat」重启；"
                "② 打开「设置→显示→图形设置→浏览」选中 runtime\\python.exe，"
                "选项设为「高性能」后重试；"
                "③ NVIDIA 显卡驱动损坏时请到官网下载最新驱动重装）"
            )
        if "cuda driver version" in low and "cuda runtime version" in low:
            return False, (
                "CPU（检测到 NVIDIA GPU，但显卡驱动过旧，低于 torch 使用的 CUDA "
                f"{torch.version.cuda} 版本。请到 NVIDIA 官网更新显卡驱动到最新版）"
            )
        if init_err:
            # 其它不常见错误：抓重点 + 给通用建议
            return False, (
                "CPU（检测到 NVIDIA GPU，但 CUDA 初始化失败："
                f"{init_err[:100]}。请通过「启动系统.bat」启动，"
                "确认 NVIDIA 驱动正常，并在「设置→图形设置」将 runtime\\python.exe 设为高性能）"
            )
        # init() 未抛错但 is_available 仍为 False（极罕见）
        return False, "CPU（检测到 NVIDIA GPU，但 CUDA 环境异常，请重启或重装显卡驱动）"

    # 2c：torch 是 CUDA 构建，但本机没有 NVIDIA 独显（核显/其他品牌显卡、或服务器无 N 卡）
    #     → 属正常情况，一行简洁说明即可，不输出 torch.init 原始错误
    return False, "CPU（本机未检测到 NVIDIA GPU）"


def resolve_device(device: str, strict: bool = False) -> str:
    """
    将界面设备选项（auto/cpu/cuda:0）归一化为推理引擎可用的设备字符串。
    兼容任意电脑环境（CPU 版 torch / 无 GPU / 有 GPU 均可）：
    - auto    -> 有 GPU 用 "0"，否则 "cpu"
    - cuda:0  -> GPU 可用则 "0"（ultralytics 格式）
                strict=False 时不可用安全回退 "cpu"
                strict=True  时不可用抛出 RuntimeError，上层可据此弹提示阻止继续
    - cpu     -> "cpu"
    确保无论用户选哪个、机器是否有 GPU，默认不会抛 'Invalid CUDA device' 错误；
    strict=True 用于需要显式拒绝用户选择 GPU 时。
    """
    if device in (None, "auto"):
        return "0" if _cuda_available() else "cpu"
    s = str(device)
    if s.startswith("cuda"):
        if _cuda_available():
            return "0"  # ultralytics 格式：单卡用 "0"
        # CUDA 不可用
        if strict:
            _ok, info = device_status()
            reason = info if info else "未检测到可用 GPU"
            raise RuntimeError(f"{reason}")
        return "cpu"
    return s if s != "cpu" else "cpu"

# ---------------- 其他 ----------------
APP_TITLE = "目标检测系统"
