"""Portable HTML and static research figures. No remote scripts or fonts."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def plot_overview(summary: pd.DataFrame, ledgers: dict, directory: Path, synthetic: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#475569",
            "text.color": "#142b40",
            "axes.edgecolor": "#cbd5e1",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
        }
    )
    colors = ["#147d92", "#ce7948", "#6778a7", "#3a9273", "#ba5970"]
    full = summary.loc[(summary["period"] == "full") & (summary["sector"] == "all_commodities")]
    height = max(4.7, 1.6 + len(full) * 0.23)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, height), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.set_facecolor("#f8fafb")
    # Show a predefined subset, never choose curves by the best realized return.
    preferred = ["t_mom", "Val_oneyear", "vol_std", "C_mainsub", "warrant_count"]
    plotted = 0
    for factor in preferred:
        match = full.loc[full["factor"] == factor]
        if match.empty:
            continue
        row = match.iloc[0]
        ledger = ledgers.get((row["family"], row["sector"], factor))
        if ledger is None or ledger.empty:
            continue
        axes[0].plot(
            ledger.index,
            (1 + ledger["net_return"]).cumprod(),
            label=factor,
            color=colors[plotted % len(colors)],
            linewidth=1.6,
        )
        plotted += 1
    axes[0].axhline(1, color="#cbd5e1", linewidth=0.8, linestyle="--")
    axes[0].set_title("Net equity | fixed factor examples", loc="left", fontsize=12, pad=16)
    axes[0].set_ylabel("Equity (initial = 1)")
    if plotted:
        axes[0].legend(frameon=False, fontsize=9)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No complete executable example\nSee data quality report",
            ha="center",
            transform=axes[0].transAxes,
        )
    axes[0].tick_params(axis="x", rotation=20)
    valid = full.dropna(subset=["rank_ic"])
    if len(valid):
        positions = np.arange(len(valid))
        axes[1].barh(
            positions,
            valid["rank_ic"],
            color=["#147d92" if x >= 0 else "#ce7948" for x in valid["rank_ic"]],
            height=0.66,
        )
        axes[1].set_yticks(positions, valid["factor"], fontsize=8)
        axes[1].invert_yaxis()
    axes[1].axvline(0, color="#94a3b8", linewidth=0.8)
    axes[1].set_title("Rank IC | all commodities", loc="left", fontsize=12, pad=16)
    axes[1].set_xlabel("Mean daily Spearman correlation")
    fig.suptitle("COMMODITY FACTOR LAB", x=0.06, y=1.02, ha="left", fontsize=15, weight="bold")
    fig.text(
        0.06,
        -0.02,
        "SYNTHETIC DATA / SOFTWARE DEMONSTRATION"
        if synthetic
        else "HISTORICAL SNAPSHOTS / EXPLORATORY RESEARCH",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(pad=1.8)
    fig.savefig(
        directory / "overview.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def write_report(
    directory: Path, summary: pd.DataFrame, quality: pd.DataFrame, synthetic: bool
) -> None:
    title = "离线演示 · 合成数据" if synthetic else "历史快照 · 探索性研究"
    usable = quality.loc[quality["status"] == "ok"]
    invalid = int(quality.get("invalid_date_rows", pd.Series(dtype=float)).sum())
    repaired = int(quality.get("date_header_repaired", pd.Series(dtype=bool)).fillna(False).sum())
    metrics = [
        ("输入文件", len(quality)),
        ("可读且有样本", len(usable)),
        ("排除的异常日期", invalid),
        ("修复的日期列名", repaired),
    ]
    cards = "".join(
        f'<div class="metric"><strong>{value:,}</strong><span>{label}</span></div>'
        for label, value in metrics
    )
    display = summary.copy()
    percent = {
        "annual_return",
        "gross_annual_return",
        "annual_volatility",
        "max_drawdown",
        "total_return",
        "ic_positive_rate",
        "mean_gross_exposure",
    }
    for col in display.columns:
        if col in percent:
            display[col] = display[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        elif pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    columns = [
        "family",
        "sector",
        "factor",
        "period",
        "status",
        "days",
        "ic_days",
        "rank_ic",
        "ic_ir",
        "gross_sharpe",
        "sharpe",
        "annual_return",
        "max_drawdown",
        "annual_turnover",
    ]
    table = display[columns].to_html(
        index=False, classes="results", border=0, table_id="results", escape=True
    )
    problem_mask = quality["status"] != "ok"
    for column in ("invalid_date_rows", "invalid_ohlc_rows", "calendar_gap_rows"):
        if column in quality:
            problem_mask |= quality[column].fillna(0).gt(0)
    if "date_header_repaired" in quality:
        problem_mask |= quality["date_header_repaired"].fillna(False).astype(bool)
    problems = quality.loc[problem_mask]
    problem_columns = [
        x
        for x in (
            "family",
            "file",
            "status",
            "invalid_date_rows",
            "invalid_ohlc_rows",
            "calendar_gap_rows",
            "date_header_repaired",
            "error",
        )
        if x in quality
    ]
    problem_table = (
        problems[problem_columns].fillna("").to_html(index=False, border=0, escape=True)
        if len(problems)
        else "<p>没有文件被整体排除。行级问题请查看完整数据质量清单。</p>"
    )
    notice = (
        "这是一份固定随机种子生成的合成数据演示。曲线用于验证软件流程，不代表任何真实期货市场的收益。"
        if synthetic
        else "历史数据的覆盖期、发布时点和连续合约拼接方法尚未统一验证。这里的结果属于探索性诊断，不能沿用论文旧指标作为新框架的业绩。"
    )
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Commodity Factor Lab · Research report</title>
<style>
:root{--ink:#142b40;--muted:#63758a;--line:#dce4ec;--teal:#147d92}
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:15px/1.65 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1400px;margin:0 auto;padding:56px 40px}header{display:flex;justify-content:space-between;gap:30px;align-items:start}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.18em;color:var(--teal)}h1{font-size:40px;line-height:1.15;letter-spacing:-.03em;margin:14px 0}h2{font-size:23px;margin:0 0 16px}p{color:var(--muted);max-width:920px}.badge{border:1px solid #9cc7ce;border-radius:20px;padding:7px 14px;white-space:nowrap;color:var(--teal);font-size:12px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}.metric{padding:24px;background:white;border:1px solid var(--line);border-radius:12px}.metric strong{display:block;font-size:32px;line-height:1.2}.metric span{display:block;color:var(--muted);font-size:13px;margin-top:8px}
section{background:white;border:1px solid var(--line);border-radius:14px;padding:30px;margin:22px 0}.notice{border-left:3px solid #ce7948;padding:12px 20px;background:#fff9f4;color:#715442}.chart{display:block;width:100%;height:auto;border-radius:10px}.scroll{overflow:auto;max-height:620px}table{border-collapse:collapse;font-size:12px;width:100%;text-align:left;white-space:nowrap}thead{position:sticky;top:0;background:#edf3f6}th,td{padding:11px 13px;border-bottom:1px solid #e8edf2}th{text-align:left;font-weight:600}tbody tr:hover{background:#f0f8f8}input,select{border:1px solid var(--line);border-radius:7px;padding:10px 12px;margin:0 10px 16px 0;color:var(--ink);font:inherit;background:white}input{width:290px}a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}.links{display:flex;gap:22px;flex-wrap:wrap}.foot{font-size:12px;color:var(--muted);margin-top:25px}
@media(max-width:700px){main{padding:28px 16px}header{display:block}h1{font-size:32px}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:18px}.badge{display:inline-block;margin-top:10px}input{max-width:100%}}
</style></head><body><main>
<header><div><div class="eyebrow">COMMODITY FACTOR LAB / RESEARCH NOTE</div><h1>从因子信号到可复核的研究</h1><p>数据质量、截面预测能力、组合成本与时间外检验，在同一份报告中检查。</p></div><span class="badge">TITLE</span></header>
<div class="metrics">CARDS</div><p class="notice">NOTICE</p>
<section><h2>研究概览</h2><img class="chart" src="overview.png" alt="Fixed factor examples and cross-sectional Rank IC"><p>组合总名义敞口为 1，多空各 0.5。收盘信号在发布滞后后，于下一交易日开盘执行，持有至再下一开盘；手续费与滑点按实际调仓名义金额单边计费。图中展示预先指定的因子示例。</p></section>
<section><h2>检验结果</h2><p>full 是探索性全样本，train / test 按配置时间划分；因子方向预先固定为 +1，除非配置明确指定。short_history 少于最短样本门槛；missing_held_price 表示持仓缺失价格，回测不输出虚构收益。ICIR 为日 IC 均值除以日 IC 标准差，未年化。</p>
<input id="search" aria-label="搜索因子或板块" placeholder="搜索因子、板块或状态…"><select id="period" aria-label="选择样本区间"><option value="full">full · 全样本</option><option value="train">train · 训练期</option><option value="test">test · 时间外</option><option value="">全部区间</option></select><span id="count"></span><div class="scroll">TABLE</div></section>
<section><h2>输入质量与排除记录</h2><p>日期列名修复、异常 OHLC、交易日缺口、因子覆盖率和排除原因全部记录在 quality.csv。交易日缺口按同类数据中该品种首尾日期之间缺少的日期统计。原始文件保持不变。</p><div class="scroll">PROBLEMS</div></section>
<section><h2>复核本次运行</h2><div class="links"><a href="summary.csv">指标明细 CSV</a><a href="quality.csv">数据质量 CSV</a><a href="run.json">参数与输入哈希</a></div><p>daily/ 内保存逐日毛净收益、成本和交易日期，weights/ 保存各品种持仓权重，ic/ 保存逐日截面相关性；文件与图表均可离线查看。gross_sharpe 为毛收益夏普，sharpe 为净收益夏普。分层均值属于信号诊断，不视为已计成本的可交易组合。最终开盘仅估值，未假设强制清仓。</p></section>
<div class="foot">Commodity Factor Lab · Reproducible research workspace · 0.1.0</div>
</main><script>
const rows=[...document.querySelectorAll('#results tbody tr')];const search=document.querySelector('#search');const period=document.querySelector('#period');
function filter(){let count=0;for(const row of rows){const visible=row.textContent.toLowerCase().includes(search.value.toLowerCase())&&(!period.value||row.cells[3].textContent===period.value);row.hidden=!visible;if(visible)count++}document.querySelector('#count').textContent=count+' 条结果'}
search.addEventListener('input',filter);period.addEventListener('change',filter);filter();
</script></body></html>"""
    for key, value in {
        "TITLE": escape(title),
        "CARDS": cards,
        "NOTICE": escape(notice),
        "TABLE": table,
        "PROBLEMS": problem_table,
    }.items():
        page = page.replace(key, value)
    (directory / "index.html").write_text(page, encoding="utf-8")
