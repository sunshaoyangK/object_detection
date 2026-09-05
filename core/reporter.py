# -*- coding: utf-8 -*-
"""
报告生成模块：模型对比表格、HTML/CSV 测评报告导出。
"""
import datetime

import pandas as pd


def build_results_table(results: dict) -> pd.DataFrame:
    """将多个模型的结果汇总成对比表格。"""
    rows = []
    for name, r in results.items():
        rows.append({
            "模型": name,
            "AP@0.5": round(r["ap_coco"], 4),
            "匹配数/预测数": f'{r["matched"]}/{r["num_pred"]}',
            "真实框数": r["num_gt"],
            "图片数": r["num_images"],
            "平均耗时(ms)": round(r["avg_time_ms"], 1),
            "FPS": round(r["fps"], 1),
            "总耗时(s)": round(r["total_time"], 1),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 按 AP 降序
    df = df.sort_values("AP@0.5", ascending=False).reset_index(drop=True)
    return df


def generate_html_report(results: dict, dataset: dict, meta: dict) -> str:
    """
    生成测评报告 HTML 文本。
    meta: {conf, nms_iou, device, person_class, match_iou, timestamp, report_desc}
    """
    df = build_results_table(results)
    table_html = df.to_html(index=False) if not df.empty else "<p>无结果</p>"

    dt = meta.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    rows_html = ""
    for name, r in results.items():
        rows_html += f"""
        <tr>
            <td>{name}</td>
            <td>{r['ap_coco']:.4f}</td>
            <td>{r['matched']}/{r['num_pred']}</td>
            <td>{r['num_gt']}</td>
            <td>{r['avg_time_ms']:.1f}</td>
            <td>{r['fps']:.1f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>人员目标检测精度测评报告</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 30px; color: #222; }}
h1 {{ color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 8px; }}
h2 {{ color: #2e74b5; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
th {{ background: #1f4e79; color: #fff; }}
tr:nth-child(even) {{ background: #f4f7fb; }}
.info {{ background: #eef4fb; padding: 14px; border-radius: 6px; line-height: 1.9; }}
.pass {{ color: #1e8e3e; font-weight: bold; }}
.fail {{ color: #d93025; font-weight: bold; }}
</style>
</head>
<body>
<h1>人员目标检测精度测评报告</h1>
<p>生成时间：{dt}</p>

<h2>一、测评目的</h2>
<p>验证人员目标检测模型的检测精度。</p>

<h2>二、测评数据</h2>
<div class="info">
<strong>数据集：</strong>{dataset.get('name', '-')}（版本：{dataset.get('version', '-')}）<br>
<strong>测评范围：</strong>{dataset.get('scope', '-')}<br>
<strong>图片数：</strong>{dataset.get('num_images', '-')}　<strong>Person 实例数：</strong>{dataset.get('num_person', '-')}
</div>

<h2>三、测评配置</h2>
<div class="info">
<strong>检测类别：</strong>Person（人员）<br>
<strong>匹配条件：</strong>预测框与真实框 IoU &ge; {meta.get('match_iou', 0.5)}<br>
<strong>置信度阈值：</strong>{meta.get('conf', '-')}　<strong>NMS IoU：</strong>{meta.get('nms_iou', '-')}<br>
<strong>推理设备：</strong>{meta.get('device', '-')}　<strong>模型 Person 类别号：</strong>{meta.get('person_class', 0)}
</div>

<h2>四、评价指标</h2>
<p>Person AP@0.5（Precision-Recall 曲线下面积）。</p>

<h2>五、测评结果</h2>
<table>
<tr><th>模型</th><th>AP@0.5</th><th>匹配/预测</th><th>真实框</th><th>平均耗时(ms)</th><th>FPS</th></tr>
{rows_html}
</table>

<h2>六、合格判定</h2>
<p>合格判据：<strong>Person AP@0.5 &gt; 90%</strong></p>
<table>
<tr><th>模型</th><th>AP@0.5</th><th>结论</th></tr>
"""
    for name, r in results.items():
        ap = float(r["ap_coco"])
        passed = ap > 0.9
        cls = "pass" if passed else "fail"
        verdict = "合格（>90%）" if passed else "不合格（<=90%）"
        html += (f'<tr><td>{name}</td><td>{ap:.4f}</td>'
                 f'<td class="{cls}">{verdict}</td></tr>')
    html += """
</table>
<p><em>注：合格判据为 AP@0.5 大于 90%。</em></p>
</body>
</html>
"""
    return html
