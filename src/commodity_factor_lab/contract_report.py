"""Contract research evidence dashboard, deliberately separate from trading performance."""

from html import escape
from pathlib import Path

import pandas as pd


def table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for col in formatted.select_dtypes(include="number"):
        formatted[col] = formatted[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "—")
    for col in formatted.select_dtypes(include="bool"):
        formatted[col] = formatted[col].map({True: "是", False: "否"})
    return (
        '<div class="scroll">'
        + formatted.fillna("—")
        .rename(
            columns={
                "signal": "因子",
                "controls": "对照变量",
                "days": "有效日期",
                "mean_partial_rank_ic": "平均增量秩相关",
                "mean_rank_ic": "平均原始秩相关",
                "hac_p": "近似 p 值",
                "hac_t": "HAC t 值",
                "by_q": "BY q 值",
                "claim_eligible": "可作投资结论",
                "positive_predictive_candidate": "探索候选",
                "code": "品种",
                "input": "输入",
                "source_kind": "来源类型",
                "input_archives": "年包数",
                "planned_requests": "计划查询",
                "attempted_requests": "已尝试",
                "successful_requests": "成功",
                "failed_requests": "失败",
                "unattempted_requests": "未尝试",
                "point_in_time_certified": "时点已认证",
                "inference_status": "推断状态",
                "curve_days": "曲线日期数",
                "momentum_days": "动量日期数",
                "basis_momentum_days": "基差动量日期数",
                "inventory_days": "季节仓单日期数",
                "rows": "记录数",
                "sessions": "交易日期数",
                "contracts": "合约数",
                "start": "开始",
                "end": "结束",
                "invalid_price_rows": "缺失或异常价格行",
                "inconsistent_ohlc_rows": "高低关系异常行",
                "zero_volume_rows": "零成交行",
                "status": "状态",
                "final_equity": "期末权益",
                "fees": "累计手续费",
                "trade_legs": "成交腿数",
                "rejected_product_orders": "拒绝品种订单数",
                "max_cash_reconciliation_error": "最大现金对账误差",
                "terminal_unliquidated": "终端未清仓",
            }
        )
        .to_html(index=False, border=0, escape=True)
        + "</div>"
    )


def write_contract_report(
    output: Path,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    daily: pd.DataFrame,
    quality: pd.DataFrame,
    labels: pd.DataFrame,
    run: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    colors = ["#147d92", "#ce7948", "#6778a7"]
    coverage.set_index("code")[["curve_days", "basis_momentum_days", "inventory_days"]].plot.bar(
        ax=axes[0], color=colors, width=0.8
    )
    axes[0].set_title("Coverage by product; unavailable is not zero alpha", loc="left", fontsize=10)
    axes[0].set_ylabel("Observed sessions")
    for i, (signal, group) in enumerate(daily.groupby("signal", sort=False)):
        series = group.set_index("date").partial_rank_ic.rolling(63, min_periods=40).mean()
        if series.notna().any():
            axes[1].plot(series.index, series, color=colors[i], label=signal)
    axes[1].set_title(
        "63-session mean partial rank IC; not a profit curve", loc="left", fontsize=10
    )
    axes[1].axhline(0, color="#94a3b8", lw=0.7)
    axes[1].axvline(
        pd.Timestamp(run["protocol"]["study"]["evaluation_start"]),
        color="#94a3b8",
        lw=0.8,
        linestyle="--",
    )
    if axes[1].lines and len(axes[1].lines) > 1:
        axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=2)
    fig.savefig(output / "overview.png", dpi=150, facecolor="#f5f8fa")
    plt.close(fig)
    counts = int(quality.rows.sum())
    candidates = int(summary.positive_predictive_candidate.sum())
    config = run["protocol"]["study"]
    source_label = "合成逐合约行情记录" if run.get("synthetic") else "官方逐合约行情记录"
    synthetic_note = "所有输入均为合成数据，仅验证软件。" if run.get("synthetic") else ""
    limitations = "".join(f"<li>{escape(item)}</li>" for item in run["limitations"])
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Commodity Factor Lab · 合约与库存研究</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f8;color:#173149;font:15px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}}main{{max-width:1360px;margin:auto;padding:42px 28px}}.eyebrow{{font-size:12px;letter-spacing:2px;color:#147d92;font-weight:700}}h1{{font-size:36px;margin:12px 0}}h2{{font-size:23px}}p{{color:#596f80}}section,.card{{background:#fff;border:1px solid #dce5ec;border-radius:12px;padding:25px}}section{{margin:24px 0}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}}.number{{display:block;font-size:32px;font-weight:700}}.notice{{background:#fff7ed;border-left:4px solid #ce7948;padding:18px 22px}}img{{width:100%;height:auto}}.scroll{{overflow:auto;max-height:480px}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px;width:100%}}td,th{{padding:11px;border-bottom:1px solid #e5edf2;text-align:left}}th{{background:#edf4f7}}a{{color:#147d92}}.links{{display:flex;gap:20px;flex-wrap:wrap}}@media(max-width:700px){{main{{padding:24px 15px}}h1{{font-size:29px}}.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:17px}}}}</style></head><body><main>
<div class="eyebrow">COMMODITY FACTOR LAB / CONTRACT &amp; WARRANT RESEARCH / v{escape(run["version"])}</div>
<h1>让因子回到真实合约与经济含义</h1><p>研究问题：期限结构与注册仓单，在价格动量之外还有多少信息？</p>
<div class="cards"><div class="card"><span class="number">{len(coverage)}</span>预先固定的试验品种</div><div class="card"><span class="number">{counts:,}</span>{source_label}</div><div class="card"><span class="number">3</span>完整登记的主检验</div><div class="card"><span class="number">0</span>已确认的投资发现</div></div>
<p class="notice">{synthetic_note}这是历史信息的预测诊断。预测标签采用同一合约的未来结算价变化，未假设可以在结算价成交；本报告没有历史投资业绩、Sharpe 或收益净值。真实研究所用官方年包仍是后来导出的历史版本，所有 claim_eligible 均为 false。</p>
<section><h2>从指标列表转向可检验假设</h2><p>H1：carry 相对 126 日动量的增量关系。H2：基差动量相对动量与 carry 的增量关系。H3：品种自身季节仓单分位相对动量与 carry 的增量关系。仓单代表注册仓单，不代表全部社会库存。</p><p>固定评估起点 {escape(config["evaluation_start"])}，预测期限 {config["holding_sessions"]} 个交易日，日度重叠标签用 {config["hac_lags"]} 日 HAC 近似推断；三个主检验共同做 BY 校正。历史已被查看，这次协议冻结不能追溯为真正事前登记。</p></section>
<section><h2>数据覆盖与预测诊断</h2><img src="overview.png" alt="品种数据覆盖和部分秩相关的滚动诊断"><p>曲线使用当时已出现的合约参数；历史收益链每次都在同一合约上计算，不把合约更换的价格差当成动量。仓单仅对品种自身过去季节进行比较；缺失和短历史保持不可用。</p>{table(coverage)}</section>
<section><h2>完整主检验</h2><p>候选与标签先秩变换，再分别剔除同日控制变量的线性关系，计算部分秩相关。正方向且本轮 BY q≤0.05 的预测候选为 {candidates} 个。它们仍受小截面、历史修订及时点假设约束，不能直接解释为可投资 Alpha。30 日到期缓冲草稿已因无法覆盖标签期限而修订为 60 日，旧协议与运行仍保留；当前三项校正不能覆盖此前的协议探索。</p>{table(summary)}</section>
<section><h2>逐年与留一品种检查</h2><p>只检查描述性稳定性，不按最优年份或品种选择模型，不增加事后显著性宣称。完整记录见 CSV。</p><div class="links"><a href="yearly_diagnostics.csv">逐年诊断</a><a href="leave_one_product_out.csv">留一品种诊断</a><a href="factor_definitions.json">因子定义与角色</a></div></section>
<section><h2>行情质量与标签失效</h2><p>原始缺价不填补。跨到期、超出未来区间或缺少同合约首尾价格的标签单独计数；因子资格不查看未来标签是否完整。</p>{table(quality)}{table(labels)}</section>
<section><h2>采集完成度</h2><p>读取成功、解析成功与历史可用时点认证是三个不同条件。失败、尚未查询和后来修订的记录均保留，不把缺口默认为零仓单。</p>{table(pd.DataFrame(run["source_collection"]))}<a href="source_collection.csv">完整采集统计</a></section>
<section><h2>下一道证据门槛</h2><p>逐日现金结算与保证金组件已建立，真实投资验证还需要经核验的夜盘开盘日历、历史手续费和保证金、合约乘数及方向成交条件。合成交易账用于验证软件，不能替代这些真实输入。</p><ul>{limitations}</ul><div class="links"><a href="summary.csv">主检验</a><a href="curves.csv">因子与合约身份</a><a href="labels.csv">同合约预测标签</a><a href="daily_ic.csv">逐日诊断</a><a href="run.json">输入、协议与源码哈希</a></div></section>
</main></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
