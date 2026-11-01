"""A compact, auditable retrospective-study report with evidence gates visible."""

from html import escape
from pathlib import Path

import pandas as pd

from .protocol import SIGNALS


def _table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    frame = frame.reindex(columns=columns) if columns else frame.copy()
    display = frame.copy()
    for col in display.select_dtypes(include="number"):
        display[col] = display[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "—")
    return (
        '<div class="scroll">'
        + display.fillna("—").to_html(index=False, border=0, escape=True)
        + "</div>"
    )


def write_study_report(
    directory: Path,
    summary: pd.DataFrame,
    costs: pd.DataFrame,
    folds: pd.DataFrame,
    ledgers: dict,
    run: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
    fig.set_facecolor("#f6f8fa")
    colors = ["#147d92", "#ce7948", "#6778a7", "#142b40"]
    for number, name in enumerate([*SIGNALS, "equal_weight_long"]):
        frame = ledgers.get(("all_commodities", name))
        if frame is not None and frame["gross_exposure"].gt(0).any():
            axes[0].plot(
                frame.index, (1 + frame["net_return"]).cumprod(), label=name, color=colors[number]
            )
    axes[0].set_title("Common-sample retrospective net equity", loc="left")
    axes[0].set_ylabel("Nominal exposure proxy; initial capital = 1")
    if axes[0].lines:
        axes[0].legend(frameon=False, fontsize=8)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No valid full-universe ledger\nSee missing-price errors in summary.csv",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            color="#a15c36",
        )
    global_costs = costs.loc[(costs["sector"] == "all_commodities") & costs["signal"].isin(SIGNALS)]
    for number, name in enumerate(SIGNALS):
        rows = global_costs.loc[global_costs["signal"] == name]
        if "annual_return" in rows and rows["annual_return"].notna().any():
            axes[1].plot(
                rows["cost_bps"],
                rows["annual_return"] * 100,
                "o-",
                label=name,
                color=colors[number],
            )
    axes[1].set_title("Cost stress test; all declared scenarios", loc="left")
    axes[1].set_xlabel("One-way cost (bps)")
    axes[1].set_ylabel("Annualised net return (%)")
    axes[1].axhline(0, color="#94a3b8", lw=0.8)
    if not global_costs.get("annual_return", pd.Series(dtype=float)).notna().any():
        axes[1].text(
            0.5,
            0.6,
            "Cost comparison unavailable\nInput quality gate failed",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            color="#a15c36",
        )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)
    fig.text(
        0.04,
        -0.02,
        "SYNTHETIC DEMO"
        if run["synthetic"]
        else "UNVERIFIED CONTINUOUS-PRICE PROXY / NO CONFIRMED ALPHA",
        fontsize=9,
    )
    fig.tight_layout(pad=2)
    fig.savefig(
        directory / "overview.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    columns = [
        "sector",
        "signal",
        "status",
        "days",
        "invested_days",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "mean_hac_p",
        "mean_by_q",
        "rank_ic",
        "ic_by_q",
        "benchmark_intercept",
        "benchmark_intercept_p",
        "claim_eligible",
    ]
    candidate_count = int(summary["positive_mean_candidate"].sum())
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Commodity Factor Lab · 研究重建</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f7;color:#142b40;font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1400px;margin:auto;padding:45px 30px}}h1{{font-size:36px;margin:12px 0}}h2{{font-size:23px}}.eyebrow{{color:#147d92;font-weight:700;font-size:12px;letter-spacing:2px}}p{{color:#566b7c}}section{{background:white;border:1px solid #dce4ec;border-radius:12px;padding:26px;margin:22px 0}}.notice{{border-left:4px solid #ce7948;background:#fff8f1;padding:18px 22px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:28px 0}}.card{{background:white;border:1px solid #dce4ec;border-radius:10px;padding:20px}}strong.number{{display:block;font-size:32px}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}}th,td{{padding:10px 12px;border-bottom:1px solid #e6edf1;text-align:left}}th{{background:#edf3f6;position:sticky;top:0}}.scroll{{overflow:auto;max-height:530px}}img{{width:100%;height:auto}}a{{color:#147d92}}.links{{display:flex;gap:22px;flex-wrap:wrap}}@media(max-width:700px){{main{{padding:25px 15px}}.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:16px}}h1{{font-size:29px}}}}</style></head><body><main>
<div class="eyebrow">COMMODITY FACTOR LAB / RESEARCH RESET / v{escape(run["version"])}</div><h1>让研究结果接受检验</h1><p>统一行情、共同样本、固定规则、滚动时间评估与成本约束。</p>
<div class="cards"><div class="card"><strong class="number">3</strong>预先固定的候选信号</div><div class="card"><strong class="number">{run["planned_primary_hypotheses"]}</strong>完整申报的主检验</div><div class="card"><strong class="number">{len(folds)}</strong>历史滚动评估区间</div><div class="card"><strong class="number">0</strong>已确认的投资发现</div></div>
<p class="notice">{"合成数据仅验证软件。" if run["synthetic"] else "当前输入是历史连续价格快照，换月、可用时点及官方交易日历尚未核验。"}历史数据已被研究过，本次滚动评估不能追溯成为真正未见的时间外验证。统计候选与可投资结论分开记录；所有 claim_eligible 均为 false。</p>
<section><h2>共同样本与成本</h2><img src="overview.png" alt="共同样本净值与交易成本压力测试"><p>三类信号均从同一行情重新计算，共享仅依赖当时历史的品种资格。满 {run["protocol"]["history_sessions"]} 个连续有效日线且过去 20 日成交量中位数达到门槛后参与。每 {run["protocol"]["rebalance_sessions"]} 个交易日调仓，期间持仓按价格漂移；收盘信号在后续第 {1 + run["protocol"]["publication_lag_sessions"]} 个交易日开盘执行，初始开仓与最终清仓均计成本。等权做多和中期动量为明确列出的基准，不将零净敞口称为市场 Beta 中性。</p></section>
<section><h2>主检验结果</h2><p>主成本固定为 {run["protocol"]["primary_cost_bps"]} bps；未按表现挑选成本。收益均值和 {run["protocol"]["rebalance_sessions"]} 日 Rank IC 分别使用 HAC 近似检验、完整主检验族 BY 校正。正收益且通过均值校正的统计候选为 {candidate_count} 个，不能据此绕过输入质量门槛。annual_return 为小数；benchmark_intercept 是有限基准回归截距的年化诊断，不是已证实的定价 Alpha。</p>{_table(summary, columns)}</section>
<section><h2>滚动区间与标签隔离</h2><p>规则、因子方向及权重公式始终固定，不声称训练了模型。每个区间之前保留历史校准窗口并剔除标签会进入测试期的训练信号；首个区间可能没有足够预热后的训练 IC。各测试段连接成一个持续运行的组合，区间边界不重置资金。</p>{_table(folds, ["fold", "train_start", "train_last_signal", "train_last_label_end", "test_entry_start", "test_entry_end", "purged_signal_sessions"])}</section>
<section><h2>所有成本情景</h2>{_table(costs, ["sector", "signal", "cost_bps", "status", "annual_return", "sharpe", "max_drawdown"])}</section>
<section><h2>复核与尚未通过的条件</h2><div class="links"><a href="summary.csv">主检验 CSV</a><a href="source_comparison.csv">跨来源行情冲突</a><a href="paired_intervals.csv">配对分块区间</a><a href="fold_results.csv">各测试段结果</a><a href="universe.csv">逐日品种资格</a><a href="run.json">协议、输入和源码哈希</a></div><p>尚需：真实合约与换月核验、可用时点、官方日历、保证金及涨跌停成交约束，以及事先冻结后新增的数据。没有恢复论文旧业绩，也没有把信号平滑称为 TWAP 或将等权称为风险平价。</p></section>
</main></body></html>"""
    (directory / "index.html").write_text(page, encoding="utf-8")
