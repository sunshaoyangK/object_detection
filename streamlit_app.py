# -*- coding: utf-8 -*-
"""
人员目标检测精度测评系统（Person AP@0.5）— Streamlit Web 界面

启动方式：
    streamlit run streamlit_app.py --server.port 8502

界面设计（前后端职责分离）：
    - 数据集根目录、置信度阈值、NMS IoU、匹配 IoU 等后端参数
      全部由 config.py 统一管理，前端不暴露任何路径或阈值调节接口；
    - 前端仅提供「选数据集 / 选模型 / 启动测评 / 查看结果」四类操作；
    - 结果区为固定高度面板：指标在一个框内滚动、图片在一个框内滚动，
      无需整页滚动即可查看完整内容。
"""
import datetime
import os
import random
import threading
import time
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

import config
from config import (
    APP_TITLE, DATASET_DIRS, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_MODEL,
    DEFAULT_MODELS, MATCH_IOU, MAX_SUBSET, MODEL_CONFIGS, PERSON_CLASS_ID,
)
from core.dataset_loader import load_dataset
from core.evaluator import Evaluator
from core import reporter, benchmark_data
from core.visualize import draw_detections

# ---------------------------------------------------------------------------
# 测评任务对象统一存放在 st.session_state["_task"]。
# 注意：Streamlit 每次 rerun 都会重新执行脚本、重置模块级变量，
# 因此不能用模块全局变量保存任务对象，否则进度与取消都会失效。
# ---------------------------------------------------------------------------


class EvalTask:
    def __init__(self, model_names, dataset, conf, nms_iou, device, imgsz=None):
        self.model_names = model_names
        self.dataset = dataset
        self.conf = conf
        self.nms_iou = nms_iou
        self.device = device
        self.imgsz = imgsz or config.DEFAULT_IMGSZ
        self.state = "running"  # running / done / error
        self.error = None
        self.progress = 0.0
        self.current_model = ""
        self.current_done = 0
        self.current_total = 0
        self.done_models = 0
        self.total_models = len(model_names)
        self.results = {}
        self.visualizations = {}   # {model: {path: [[x1,y1,x2,y2,conf], ...]}}
        self.cancel = False

    def _progress_cb(self, done, total, model_name):
        self.current_model = model_name
        self.current_done = done
        self.current_total = total
        self.progress = min(
            (self.done_models + done / max(total, 1)) / max(self.total_models, 1),
            1.0,
        )

    def run(self):
        try:
            # 保存全部已推理图片的预测框，供可视化区按文件名逐张检索
            sample_paths = [p for p, _ in self.dataset.samples]
            for i, name in enumerate(self.model_names):
                if self.cancel:
                    break
                self.done_models = i
                self.current_model = f"{name}（加载中…）"
                # 景点检测模型（task=landmark）保留全部类别预测，GT 聚合口径
                keep_all = MODEL_CONFIGS.get(name, {}).get("task") == "landmark"
                ev = Evaluator(name, self.device, keep_all_classes=keep_all)
                res = ev.evaluate(
                    self.dataset,
                    conf=self.conf,
                    nms_iou=self.nms_iou,
                    progress_cb=self._progress_cb,
                    cancel_event=lambda: self.cancel,
                    sample_paths=sample_paths,
                    imgsz=self.imgsz,
                )
                self.results[name] = res
                self.visualizations[name] = res.get("sample_preds", {})
            self.done_models = len(self.model_names)
            self.progress = 1.0
            # 用户取消时单独标记：部分结果直接丢弃，不作为测评成果展示
            self.state = "cancelled" if self.cancel else "done"
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            self.state = "error"


# ---------------------------------------------------------------------------
# 页面配置与全局样式
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")

CSS = """
<style>
:root{
  --brand:#2563eb; --ok:#16a34a; --bad:#dc2626;
  --line:#e5e9f0; --txt:#0f172a; --sub:#6b7280;
}
.block-container{padding-top:.9rem; padding-bottom:1.2rem; max-width:1500px;}
.main [data-testid="stVerticalBlock"]{gap:.4rem;}
[data-testid="stVerticalBlockBorderWrapper"]{padding-top:.5rem !important; padding-bottom:.5rem !important;}
.bm-cards{display:flex;gap:10px;margin:2px 0;}
.bm-card{flex:1;background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:6px 12px;text-align:center;}
.bm-label{font-size:.72rem;color:var(--sub);}
.bm-val{font-size:1.25rem;font-weight:800;color:var(--brand);line-height:1.35;}
.app-title{font-size:1.4rem; font-weight:800; color:var(--txt); line-height:1.25;}
.app-sub{color:var(--sub); font-size:.83rem; margin-top:4px;}
.sb-title{font-size:1.8rem; font-weight:800; color:var(--txt); text-align:center;
           background:#e5e7eb; border-radius:10px; padding:8px 16px; margin-bottom:4px;}
.sec-title{display:flex; align-items:center; font-size:1.05rem; font-weight:800;
           color:var(--txt); margin:6px 0 10px;}
.sec-bar{width:4px; height:16px; border-radius:2px; background:var(--brand);
         margin-right:9px; display:inline-block;}
.hint{color:var(--sub); font-size:.85rem;}
.ds-line{background:#f8fafc; border:1px solid var(--line); border-radius:10px;
         padding:8px 14px; color:#334155; font-size:.9rem;}
.ds-line b{color:#0f172a;}
.model-card{border:1px solid var(--line); border-radius:8px; padding:6px 10px;
            margin-bottom:6px; background:#fff; box-shadow:0 1px 2px rgba(16,24,40,.04);}
.model-head{display:flex; align-items:center; gap:8px; margin-bottom:4px;}
.model-name{font-weight:700; font-size:.9rem; color:var(--txt);}
.model-stats{display:flex; flex-wrap:wrap; gap:6px;}
.ms{flex:1 1 100px; background:#f8fafc; border:1px solid var(--line);
    border-radius:6px; padding:5px 6px; text-align:center;}
.ms-num{font-size:.85rem; font-weight:700; color:var(--txt);}
.ms-num.ok{color:var(--ok);}
.ms-label{font-size:.65rem; color:var(--sub); margin-top:1px;}
[data-testid="stSidebar"]{background:#f7f9fc; border-right:1px solid var(--line); scrollbar-width:none;}
[data-testid="stSidebar"]::-webkit-scrollbar{width:0; height:0; display:none;}
[data-testid="stSidebar"] h1{font-size:1.0rem;}
[data-testid="stSidebar"] h2{margin:.1rem 0 .25rem;}
[data-testid="stSidebar"] .block-container{padding-top:.2rem; padding-bottom:.6rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.95rem;}
[data-testid="stSidebar"] hr{margin:.5rem 0;}
/* 主操作按钮（② 开始测评等 primary 按钮）固定蓝色底色 */
.stButton > button[kind="primary"]{
  background-color:#2563eb !important;
  border-color:#2563eb !important;
  color:#ffffff !important;
}
.stButton > button[kind="primary"]:hover{
  background-color:#1d4ed8 !important;
  border-color:#1d4ed8 !important;
}
.stButton > button[kind="primary"]:disabled{
  background-color:#93b4f5 !important;
  border-color:#93b4f5 !important;
  color:#ffffff !important;
}
img{border-radius:8px;}
[data-testid="stExpander"] details summary{padding:.35rem 1rem; min-height:0;}
[data-testid="stExpander"]{margin:0;}
.stRadio > div{gap:.4rem;}
.stRadio label{margin:0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def section(title: str):
    """渲染分区标题（左侧蓝色竖条 + 加粗标题）。"""
    st.markdown(f'<div class="sec-title"><span class="sec-bar"></span>{title}</div>',
                unsafe_allow_html=True)


def model_result_card(name: str, r: dict) -> str:
    """单个模型的核心指标卡片（纯 HTML，不包含交互组件）。达标数值绿色标注。"""
    ap_voc, ap_coco = float(r["ap_voc"]), float(r["ap_coco"])
    recall = float(r["recall"][-1])
    precision = float(r["precision"][-1])
    cells = [
        (f"{ap_coco * 100:.2f}%", "AP@0.5", "ok" if ap_coco > 0.9 else ""),
        (f"{recall * 100:.2f}%", "Recall", ""),
        (f"{precision * 100:.2f}%", "Precision", ""),
        (f"{r['avg_time_ms']:.0f} ms/张", f"速度（{r['fps']:.1f} FPS）", ""),
    ]
    stats = "".join(
        f'<div class="ms"><div class="ms-num {cls}">{num}</div>'
        f'<div class="ms-label">{lab}</div></div>'
        for num, lab, cls in cells
    )
    head = f'<div class="model-head"><span class="model-name">{name}</span></div>'
    return f'<div class="model-card">{head}<div class="model-stats">{stats}</div></div>'


def style_results_table(df: pd.DataFrame):
    """结果对比表：AP 列按百分比显示，>90% 绿色标注。"""
    if df.empty:
        return df
    sty = df.style.format({
        "AP@0.5": lambda v: f"{v * 100:.2f}%",
        "平均耗时(ms)": lambda v: f"{v:.0f}",
        "FPS": lambda v: f"{v:.1f}",
        "总耗时(s)": lambda v: f"{v:.0f}",
    })
    sty = sty.map(
        lambda v: "color:#16a34a;font-weight:600;" if v > 0.9 else "color:#6b7280;",
        subset=["AP@0.5"],
    )
    return sty


# ---------------------------------------------------------------------------
# 侧边栏：标题 + 测评配置与操作 + 使用说明（仅功能选项，不暴露路径/阈值）
# ---------------------------------------------------------------------------
with st.sidebar:
    section("🎯 检测任务")
    task_label = st.radio(
        "检测任务", ["🚶 人员检测", "🏛️ 景点检测"],
        index=0, label_visibility="collapsed",
    )
    task_key = "person" if "人员" in task_label else "landmark"

    section("⚙️ 测评配置")
    # 自定义上传仅人员任务提供（景点任务使用固定的 NYC 地标测评数据集）
    ds_items = [k for k, v in DATASET_DIRS.items()
                if v.get("task") == task_key
                or (v.get("task") == "custom" and task_key == "person")]
    kind = st.selectbox(
        "数据集类型",
        ds_items,
        format_func=lambda k: (
            f"{DATASET_DIRS[k]['name']}（{DATASET_DIRS[k]['version']}）"
            if DATASET_DIRS[k]['version'] else DATASET_DIRS[k]['name']
        ),
    )
    is_custom = kind == "custom"
    # 任务或数据集切换后清空旧结果，避免串数据（含上传的内存图片与识别结果）
    _cur_ctx = f"{task_key}:{kind}"
    if st.session_state.get("_cur_ctx") != _cur_ctx:
        for _k in ("dataset", "dataset_summary", "last_results",
                   "last_visualizations", "last_meta", "_upload_results",
                   "_upload_imgs", "_up_img_idx",
                   "_vis_sel"):
            st.session_state.pop(_k, None)
        st.session_state["_cur_ctx"] = _cur_ctx
    max_images = 0
    if not is_custom:
        max_images = st.number_input(
            "测评图片数上限（0=全部，最小 100 张）",
            min_value=0, max_value=MAX_SUBSET, value=0, step=100,
            help="上限>0 时从全量数据集中随机抽样（固定种子、可复现）。"
                 "低于 100 张时 AP 统计波动过大，已自动限制为 100；"
                 "正式达标判定请设为 0 跑全量。",
        )
        # 小样本 AP 波动过大（10~50 张时随机抽样仍频繁跌破 90%），强制下限 100 张
        if 0 < max_images < 100:
            st.caption("⚠️ 样本过小结果不可靠，已自动调整为 100 张；正式判定请设为 0 跑全量。")
            max_images = 100

    if task_key == "person":
        person_models = [m for m in DEFAULT_MODELS
                         if MODEL_CONFIGS[m].get("task") == "person"]
        _model_sel = st.selectbox(
            "待测模型",
            person_models,
            index=person_models.index(DEFAULT_MODEL),
            format_func=lambda m: m,
        )
        model_names = [_model_sel]
    else:
        landmark_models = [m for m in DEFAULT_MODELS
                           if MODEL_CONFIGS[m].get("task") == "landmark"]
        _lm_sel = st.selectbox(
            "待测模型",
            landmark_models,
            index=landmark_models.index("YOLO11n-景点"),
            format_func=lambda m: m,
        )
        model_names = [_lm_sel]

    # 模型切换后清空旧测评/识别结果，避免上一个模型的结果残留界面
    _model_ctx = f"{task_key}:{model_names[0] if model_names else ''}"
    if st.session_state.get("_model_ctx") != _model_ctx:
        for _k in ("last_results", "last_visualizations", "last_meta",
                   "_vis_sel", "_upload_results", "_up_img_idx",
                   "_task", "_task_alive"):
            st.session_state.pop(_k, None)
        st.session_state["_model_ctx"] = _model_ctx

    device = st.selectbox(
        "推理设备", ["auto", "cpu", "cuda:0"], index=0,
        help="auto 自动选择 GPU（可用时）",
    )
    # 显示真实解析到的设备及原因，每台电脑环境不同结果也不同（动态检测）。
    # 选 cuda:0 但本机无可用 GPU 时也会回退 CPU，提示里讲清原因，不报错。
    _ok, _info = config.device_status()
    _resolved = config.resolve_device(device)
    icon = "⚡" if (_ok and _resolved != "cpu") else "🔁"
    st.caption(f"{icon} 当前使用：**{_info}**")

    section("🚀 操作")
    load_btn = None
    start_btn = None
    stop_btn = None
    recog_btn = None
    if is_custom:
        recog_btn = st.button("② 开始识别上传图片", type="primary", use_container_width=True)
    else:
        load_btn = st.button("① 加载数据集", use_container_width=True)
        # 按钮状态直接取任务对象的真实状态：完成/取消/出错当帧立即复原，
        # 避免「开始测评」迟迟不解禁、取消按钮残留的问题
        _t = st.session_state.get("_task")
        _running = _t is not None and _t.state == "running"
        start_btn = st.button("② 开始测评", type="primary", use_container_width=True,
                              disabled=_running)
        if _running:
            stop_btn = st.button("⏹ 取消测评", use_container_width=True)

# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
# 顶部标题卡片（填补主区域上方空白）
st.markdown(
    f'<div class="sb-title" style="text-align:center;">🎯 {APP_TITLE}</div>',
    unsafe_allow_html=True,
)
st.markdown("")  # 一点间距

if is_custom:
    # ================= 自定义图片识别（仅人员检测任务提供） =================
    with st.container(border=True):
        section("🖼️ 上传图片")
        st.caption("上传任意图片，使用侧边栏所选行人模型检测图中行人。")
        uploaded = st.file_uploader(
            "上传图片（可多选：jpg/png/bmp/ppm）",
            type=["jpg", "jpeg", "png", "bmp", "ppm"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    # 上传图片仅保存在内存（session_state），不写入本地磁盘
    upload_imgs = st.session_state.get("_upload_imgs", {})

    if uploaded:
        imgs = {}
        for f in uploaded:
            try:
                imgs[f.name] = Image.open(BytesIO(f.getvalue())).convert("RGB")
            except Exception:
                continue  # 跳过无法解码的文件
        st.session_state["_upload_imgs"] = imgs
        upload_imgs = imgs
    elif upload_imgs:
        # 上传区被清空 -> 清除旧图片与识别结果
        st.session_state.pop("_upload_imgs", None)
        st.session_state.pop("_upload_results", None)
        st.session_state.pop("_up_img_idx", None)
        upload_imgs = {}

    if upload_imgs:
        st.info(f"已就绪 {len(upload_imgs)} 张图片，点击侧边栏「② 开始识别上传图片」即可。")

    if recog_btn:
        if not upload_imgs:
            st.warning("请先上传图片。")
        elif str(device).startswith("cuda"):
            # 硬性校验：用户选了 cuda:0 但本机无可用 GPU，直接弹窗并阻止，不做静默回退
            try:
                config.resolve_device(device, strict=True)
            except RuntimeError as _e:
                st.error(
                    f"⚠️ 未检测到可用 GPU 或 CUDA 版本不匹配（{_e}），请切换到「CPU」或「auto」再试。",
                    icon="🚫",
                )
                recog_btn = False
        if recog_btn:
            # 用侧边栏所选行人模型检测行人
            res = {}
            with st.spinner("行人识别中，请稍候…"):
                for mname in model_names:
                    ev = Evaluator(mname, device)
                    res[mname] = ev.predict_images(upload_imgs, conf=DEFAULT_CONF,
                                                   nms_iou=DEFAULT_IOU)
            st.session_state["_upload_results"] = res

    up_res = st.session_state.get("_upload_results")
    if up_res:
        with st.container(border=True):
            section("📸 识别结果")
            # 单模型展示（模型均为单选），取第一个
            mname, preds_by_name = next(iter(up_res.items()))
            n_box = sum(len(v) for v in preds_by_name.values())
            st.markdown(f"**{mname}** — 共检测到 {n_box} 个人")

            names = list(preds_by_name.keys())
            total = len(names)
            idx = st.session_state.get("_up_img_idx", 0)
            if idx >= total:
                idx = 0
                st.session_state["_up_img_idx"] = 0

            # 左右切换按钮 + 中间计数（多张图时逐张对比查看）
            prev_col, cnt_col, next_col = st.columns([1, 2, 1])
            with prev_col:
                if st.button("◀ 上一张", use_container_width=True,
                             disabled=total <= 1, key="_up_prev"):
                    st.session_state["_up_img_idx"] = (idx - 1) % total
                    st.rerun()
            with cnt_col:
                st.markdown(
                    f'<div style="text-align:center;font-weight:600;color:#0f172a;'
                    f'padding:4px 0;">第 {idx + 1} / {total} 张</div>',
                    unsafe_allow_html=True,
                )
            with next_col:
                if st.button("下一张 ▶", use_container_width=True,
                             disabled=total <= 1, key="_up_next"):
                    st.session_state["_up_img_idx"] = (idx + 1) % total
                    st.rerun()

            name = names[idx]
            preds = preds_by_name[name]
            disp = preds
            # 限高：不滚动页面即可看全貌
            st.markdown(
                '<style>div[data-testid="stImage"] img{max-height:38vh;object-fit:'
                'contain;margin:auto;}</style>',
                unsafe_allow_html=True,
            )
            # 左原图 | 右结果图 对比
            orig = upload_imgs.get(name)
            left_col, right_col = st.columns(2)
            with left_col:
                if orig is not None:
                    st.image(orig, caption=f"原图：{name}", use_container_width=True)
            with right_col:
                if orig is not None:
                    result_img = draw_detections(orig, [], disp)
                    st.image(
                        result_img,
                        caption=f"结果：检测到 {len(preds)} 人",
                        use_container_width=True,
                    )

else:
    # ================= 数据集精度测评 =================

    # ---- 综合测试结果（选模型参考）----
    # 未开始测评时固定展示所选模型在该数据集上的全量指标；开始测评/测评完成后隐藏，
    # 让首屏专注展示本次测评的进度、结果指标与可视化，避免信息堆叠
    _eval_active = (st.session_state.get("_task") is not None
                    or bool(st.session_state.get("last_results")))
    bm_block = benchmark_data.get_dataset_benchmark(task_key, kind)
    if bm_block is not None and not _eval_active:
        with st.container(border=True):
            _sel_model = model_names[0] if model_names else None
            _bm = bm_block["models"].get(_sel_model) if _sel_model else None
            if _bm is not None:
                ap_bm, rec_bm, prec_bm = _bm
                # 醒目标注：当前所选模型 / 数据集 / 全量口径
                st.markdown(
                    f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;'
                    f'padding:6px 12px;font-size:.88rem;color:#1e3a8a;margin-bottom:6px;">'
                    f'🎯 <b>当前模型：{_sel_model}</b> ｜ 数据集：<b>{bm_block["name"]}</b>'
                    f' ｜ 全量测评口径（置信度 0.25 · IoU≥0.5）</div>',
                    unsafe_allow_html=True,
                )
                section("📊 综合测试结果（当前模型全量指标）")
                st.markdown(
                    f'<div class="bm-cards">'
                    f'<div class="bm-card"><div class="bm-label">AP@0.5</div>'
                    f'<div class="bm-val" style="color:var(--ok)">{ap_bm:.2f}%</div></div>'
                    f'<div class="bm-card"><div class="bm-label">Recall</div>'
                    f'<div class="bm-val">{rec_bm:.2f}%</div></div>'
                    f'<div class="bm-card"><div class="bm-label">Precision</div>'
                    f'<div class="bm-val">{prec_bm:.2f}%</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            # 各模型指标对比表（按 AP@0.5 从高到低，≥90% 绿色加粗）
            bm_rows = [
                {"模型": m, "AP@0.5(%)": v[0], "Recall(%)": v[1], "Precision(%)": v[2]}
                for m, v in bm_block["models"].items()
            ]
            bm_df = pd.DataFrame(bm_rows).sort_values("AP@0.5(%)", ascending=False)
            st.markdown(
                f'<div style="font-weight:700;font-size:.92rem;color:#0f172a;margin:8px 0 2px;">'
                f'📈 {bm_block["name"]} 各模型指标对比（按 AP@0.5 从高到低）</div>',
                unsafe_allow_html=True,
            )
            bm_style = bm_df[["模型", "AP@0.5(%)", "Recall(%)", "Precision(%)"]].style.format({
                "AP@0.5(%)": "{:.2f}", "Recall(%)": "{:.2f}", "Precision(%)": "{:.2f}",
            }).map(
                lambda v: "color:#16a34a;font-weight:700;" if v >= 90 else "",
                subset=["AP@0.5(%)"],
            )
            st.dataframe(bm_style, use_container_width=True, hide_index=True,
                         height=42 + 37 * len(bm_df))

    # 加载数据集（数据集根目录由 config 统一指定，前端不可见）
    if load_btn:
        # 重新加载数据集时清掉旧任务 + 旧测评结果 + 旧可视化状态，避免残留误导
        for _k in ("_task", "last_results", "last_visualizations",
                   "_vis_sel"):
            st.session_state.pop(_k, None)
        try:
            with st.spinner(f"正在加载数据集 {DATASET_DIRS[kind]['name']}，首次需解析全部标注，请稍候…"):
                ds = load_dataset(kind, DATASET_DIRS[kind]["dir"],
                                  max_images=max_images or None)
            st.session_state["dataset"] = ds
            st.session_state["dataset_summary"] = ds.summary()
            # 立即刷新：回到未测评布局（综合指标参考区重新显示），旧测评视图干净移除
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("dataset", None)
            st.session_state.pop("dataset_summary", None)
            st.error(f"数据集加载失败：{e}")

    # 开始测评（测评参数 conf / nms_iou 均由 config 固定）
    _running_task = st.session_state.get("_task")
    if start_btn:
        ds = st.session_state.get("dataset")
        if ds is None:
            st.warning("请先点击侧边栏「① 加载数据集」。")
        elif _running_task is not None and _running_task.state == "running":
            # 单例防护：测评进行中忽略重复启动请求，避免多进程争抢资源
            st.info("⏳ 测评正在进行中，请等待完成或先取消。")
        elif not model_names:
            st.warning("请至少选择一个待测模型。")
        elif str(device).startswith("cuda"):
            # 硬性校验：用户选了 cuda:0 但本机无可用 GPU，直接弹窗并阻止，不做静默回退
            try:
                config.resolve_device(device, strict=True)
            except RuntimeError as _e:
                st.error(
                    f"⚠️ 未检测到可用 GPU 或 CUDA 版本不匹配（{_e}），请切换到「CPU」或「auto」再试。",
                    icon="🚫",
                )
                start_btn = False
        if start_btn:
            # 重新测评前清空上一轮结果：测评进行中界面下方保持空白，不残留旧数据
            for _k in ("last_results", "last_visualizations", "last_meta",
                       "_vis_sel"):
                st.session_state.pop(_k, None)
            imgsz = config.DATASET_IMGSZ.get(kind, config.DEFAULT_IMGSZ)
            st.session_state["_task"] = EvalTask(model_names, ds,
                                                 DEFAULT_CONF, DEFAULT_IOU, device,
                                                 imgsz=imgsz)
            st.session_state["_task_alive"] = True
            threading.Thread(target=st.session_state["_task"].run, daemon=True).start()
            # 立即刷新：隐藏综合指标参考区，切换为测评进度/结果视图
            st.rerun()

    # 取消
    if stop_btn:
        t = st.session_state.get("_task")
        if t is not None and t.state == "running":
            t.cancel = True
            st.info("正在取消测评，请稍候…")

    # ---- 测评数据概况（紧凑一行：数据集名称 / 图片数 / 实例数） ----
    if "dataset_summary" in st.session_state:
        s = st.session_state["dataset_summary"]
        ds_cur = st.session_state.get("dataset")
        cls_lower = {c.lower() for c in (ds_cur.classes or ["person"])} if ds_cur else {"person"}
        inst_label = "Person 实例" if "person" in cls_lower else "目标实例"
        st.markdown(
            f'<div class="ds-line">📊 测评数据：<b>{s["name"]}</b>'
            f' ｜ 图片数 <b>{s["num_images"]:,}</b>'
            f' ｜ {inst_label} <b>{s["num_person"]:,}</b></div>',
            unsafe_allow_html=True,
        )

    # ---- 一次性提示（如取消测评后的通知，刷新回未测评布局时展示一次） ----
    if st.session_state.get("_notice"):
        st.info(st.session_state.pop("_notice"))

    # ---- 任务状态与进度 ----
    task = st.session_state.get("_task")
    if task is not None:
        if task.state == "running":
            with st.container(border=True):
                section("⏳ 测评进行中")
                st.markdown(
                    f"当前模型：**{task.current_model}**"
                    f"（模型 {task.done_models + 1}/{task.total_models}）"
                )
                if task.current_total > 0:
                    st.markdown(f"图片处理：**{task.current_done}/{task.current_total}** 张")
                st.progress(float(task.progress), text=f"总体进度 {task.progress * 100:.1f}%")
            time.sleep(0.6)
            st.rerun()
        elif task.state == "error":
            st.session_state["_task_alive"] = False
            with st.container(border=True):
                section("⚠️ 测评出错")
                st.error(task.error)
        elif task.state == "cancelled":
            # 取消：丢弃部分结果，回到未测评布局（综合指标参考区恢复显示）
            st.session_state["_task_alive"] = False
            st.session_state["_notice"] = "⏹ 测评已取消，结果未保留。"
            st.session_state.pop("_task", None)
            st.rerun()
        elif task.state == "done":
            st.session_state["_task_alive"] = False
            # 仅当完成的任务与当前所选模型一致时才落盘展示（测评中途切换模型则丢弃旧任务结果）；
            # 落盘后立即移除任务对象，避免后续 rerun 反复把旧结果写回界面
            if model_names and task.model_names and task.model_names[0] == model_names[0]:
                st.session_state["last_results"] = task.results
                st.session_state["last_visualizations"] = task.visualizations
                st.session_state["last_meta"] = {
                    "conf": task.conf,
                    "nms_iou": task.nms_iou,
                    "device": task.device,
                    "imgsz": task.imgsz,
                    "match_iou": MATCH_IOU,
                    "person_class": PERSON_CLASS_ID,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            st.session_state.pop("_task", None)

    # ---- 测评结果：上方紧凑指标表格 + 下方单图可视化，一屏内完成 ----
    results = st.session_state.get("last_results")
    if results:
        vis = st.session_state.get("last_visualizations")

        # 指标区：只保留核心指标卡（紧凑一行）
        section("📈 测评结果指标")
        for name, r in results.items():
            st.markdown(model_result_card(name, r), unsafe_allow_html=True)

        # 可视化区：逐张检索（搜索过滤 + 下拉跳转 + 翻页浏览 + 原图/结果左右对照）
        section("📸 检测结果可视化")
        if vis:
            ds = st.session_state.get("dataset")
            gt_by_path = dict(ds.samples) if ds is not None else {}

            # 单模型测评，结果取第一个模型
            vis_model = list(vis.keys())[0]
            preds_by_path = vis.get(vis_model, {})
            paths = list(preds_by_path.keys())
            total = len(paths)

            # 关键词搜索：按文件名过滤下拉候选范围
            kw = st.text_input(
                "🔍 图片文件名搜索",
                placeholder="输入文件名关键词（如 008455、hiking），留空=全部",
                key="_vis_kw",
                label_visibility="collapsed",
            )
            filtered_paths = [
                p for p in paths
                if not kw or kw.lower() in os.path.basename(p).lower()
            ]

            # 选中项边界保护：_vis_sel 可能来自 rerun 残留、或不在过滤结果中
            sel = st.session_state.get("_vis_sel")
            if sel not in filtered_paths:
                sel = filtered_paths[0] if filtered_paths else None
                if sel is not None:
                    st.session_state["_vis_sel"] = sel
                    st.rerun()

            # 翻页意图解析：button 只写 _vis_delta（+1/-1），不在回调里写 _vis_sel
            # （Streamlit 禁止 button 回调修改同周期 selectbox 的 key，会报 cannot be modified）
            # 翻页范围跟随搜索过滤结果：搜索后仅在匹配图片之间循环浏览
            delta = st.session_state.pop("_vis_delta", None)
            if delta is not None and filtered_paths:
                cur = filtered_paths.index(sel) if sel in filtered_paths else 0
                st.session_state["_vis_sel"] = filtered_paths[
                    (cur + delta) % len(filtered_paths)
                ]
                st.rerun()

            # 控件行：下拉框 | 上一张 | 下一张 | 计数
            col_sel, col_prev, col_next, col_cnt = st.columns([4.2, 1.0, 1.0, 1.2])
            with col_sel:
                st.selectbox(
                    "图片索引", options=filtered_paths,
                    index=filtered_paths.index(sel) if sel in filtered_paths else 0,
                    format_func=os.path.basename, label_visibility="collapsed",
                    key="_vis_sel",
                )
            with col_prev:
                if st.button("◀ 上一张", use_container_width=True,
                             disabled=len(filtered_paths) <= 1):
                    st.session_state["_vis_delta"] = -1
                    st.rerun()
            with col_next:
                if st.button("下一张 ▶", use_container_width=True,
                             disabled=len(filtered_paths) <= 1):
                    st.session_state["_vis_delta"] = 1
                    st.rerun()
            with col_cnt:
                # 若正在过滤，额外提示匹配数
                extra = f"（匹配 {len(filtered_paths)}）" if kw and len(filtered_paths) != total else ""
                st.markdown(
                    f'<div style="text-align:right;font-weight:500;'
                    f'color:#475569;padding-top:9px;font-size:0.9rem;">'
                    f'共 {total} 张{extra}</div>',
                    unsafe_allow_html=True,
                )

            # 原图 / 检测结果 左右对照（红框=真实标注，绿框=模型预测，并排直接对比）
            if paths and sel:
                gt = gt_by_path.get(sel, [])
                pred = preds_by_path.get(sel, [])
                st.markdown(
                    '<style>div[data-testid="stImage"] img{max-height:28vh;object-fit:'
                    'contain;margin:auto;}</style>',
                    unsafe_allow_html=True,
                )
                col_orig, col_pred = st.columns(2)
                with col_orig:
                    st.image(
                        Image.open(sel).convert("RGB"),
                        caption=f"原图｜{os.path.basename(sel)}",
                        use_container_width=True,
                    )
                with col_pred:
                    img = draw_detections(sel, gt, pred)
                    st.image(
                        img,
                        caption=f"检测结果（真 {len(gt)} / 预 {len(pred)}）",
                        use_container_width=True,
                    )
        else:
            st.info("完成一轮测评后，这里可逐张查看全量图片的检测结果「红框=真实 / 绿框=预测」。")

        # ---- 报告导出（紧凑一行，与可视化同屏可见） ----
        st.caption("📄 报告导出")
        meta = st.session_state.get("last_meta", {})
        ds_sum = st.session_state.get("dataset_summary", {})
        html = reporter.generate_html_report(results, ds_sum, meta)
        table = reporter.build_results_table(results)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ec1, ec2, _ = st.columns([0.3, 0.3, 0.4])
        with ec1:
            st.download_button(
                "⬇️ 下载 HTML 测评报告",
                data=html.encode("utf-8"),
                file_name=f"person_ap_report_{ts}.html",
                mime="text/html",
                use_container_width=True,
            )
        with ec2:
            csv = table.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ 下载 CSV 结果表",
                data=csv,
                file_name=f"person_ap_report_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
