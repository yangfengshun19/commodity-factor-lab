"""Offline replay of the bounded physical-spot source assay; never reads return labels."""

import hashlib
import json
import tomllib
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from . import __version__
from .contract_report import table
from .spot import INSTRUMENTS, match_spot_diagnostics, parse_spot_page


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spot_table(frame: pd.DataFrame) -> str:
    labels = {
        "symbol": "现货合约",
        "code": "品种",
        "kind": "交割类型",
        "planned_dates": "固定日期",
        "parsed_dates": "已解析日期",
        "valid_traded_quotes": "有成交的有效报价",
        "same_date_contract_diagnostics": "同日期价格比较",
        "strict_factor_observations": "严格因子观测",
        "clock_certified": "历史发布已认证",
        "date": "日期",
        "quote_status": "报价状态",
        "close": "现货收盘",
        "volume_as_published": "公布成交量",
        "front_contract": "原选期货合约",
        "reference_settle": "期货结算",
        "diagnostic_log_quote_ratio": "对数价格比（仅诊断）",
        "parse_status": "解析状态",
        "table_rows": "表内合约数",
        "date_identity": "日期依据",
        "publisher_clock": "历史发布时间",
    }
    values = {
        "physical_cash_t0": "实盘 T+0",
        "physical_spot_t2": "实物 T+2",
        "quote_ok": "有成交且价格合格",
        "no_trade": "零成交，不计算",
        "symbol_absent": "表内无此合约",
        "empty_daily_table": "当日空表",
        "ok": "通过",
        "article_heading": "正文标题日期",
        "explicit_table_rows": "逐行明确日期",
        "empty_table_no_date_claim": "空表，无报价日期",
        "not_certified": "未认证",
    }
    display = frame.copy()
    for column in display.select_dtypes(include="object"):
        display[column] = display[column].map(lambda value: values.get(value, value))
    return table(display.rename(columns=labels))


def run_spot_readiness(config: Path, sources: Path) -> Path:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {name: (root / value).resolve() for name, value in plan["paths"].items()}
    study = plan["study"]
    if (
        study["kind"] != "data_readiness_only"
        or study["prediction_tests_added"] != 0
        or study["predictive_statistics_computed"]
        or study["spot_basis_claim_eligible"]
        or not study["require_positive_volume"]
        or not study["require_publisher_clock"]
        or study["allow_assumed_publication_lag"]
        or not study["require_brand_location_tax_mapping"]
        or study["min_inference_assets_unchanged"] < 8
        or study["spot_price_field"] != "close"
        or study["gold_tax_effective_date"] != "2025-11-01"
    ):
        raise ValueError(
            "The source assay cannot activate factor inference or relax matching gates"
        )
    instruments = {item["symbol"]: item for item in plan["instruments"]}
    if len(plan["instruments"]) != 3 or instruments.keys() != INSTRUMENTS.keys():
        raise ValueError("Physical instrument identities differ from the frozen mapping")
    for symbol, (code, kind, unit) in INSTRUMENTS.items():
        item = instruments[symbol]
        if (
            (item["code"], item["kind"], item["quote_unit"]) != (code, kind, unit)
            or item["quote_multiplier"] != 1.0
            or item["brand_location_tax_history_verified"]
        ):
            raise ValueError(
                "Do not confuse quote units with trading lots or certify unknown history"
            )
    for name, expected in plan["input_hashes"].items():
        if sha256(paths[name]) != expected:
            raise ValueError(f"Frozen assay input changed: {name}")
    manifest_path = paths["raw"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(config) or manifest["sources_sha256"] != sha256(
        sources
    ):
        raise ValueError(
            "Source capture and offline assay must use the same frozen protocol/catalog"
        )
    if pd.Timestamp(plan["registered_utc"]) >= pd.Timestamp(manifest["created_utc"]):
        raise ValueError("Protocol must predate local source sampling")
    expected_dates = plan["probe"]["dates"]
    requests = manifest["planned_requests"]
    if [r["date"] for r in requests if "date" in r] != expected_dates:
        raise ValueError("Capture does not match the fixed date schedule")
    planned = {item["id"]: item for item in requests}
    records = manifest["records"]
    if len({item["id"] for item in records}) != len(records):
        raise ValueError("Duplicate source receipt identity")
    quotes, page_audits = [], []
    for item in records:
        if item["id"] not in planned or any(item[k] != v for k, v in planned[item["id"]].items()):
            raise ValueError("Receipt source identity differs from its frozen request")
        if "file" in item:
            path = paths["raw"] / item["file"]
            if not path.resolve().is_relative_to(paths["raw"]) or sha256(path) != item["sha256"]:
                raise ValueError("Raw source hash/path does not match its receipt")
        if item["mode"] == "document":
            continue
        audit = {"date": item["date"], "url": item["url"], "sha256": item.get("sha256", "")}
        if item["status"] != "http_ok":
            audit.update(parse_status="source_unavailable", error=item.get("error", "HTTP failure"))
        else:
            try:
                parsed, detail = parse_spot_page(path.read_bytes(), item["date"], item["mode"])
                parsed["source_url"] = item["url"]
                parsed["source_sha256"] = item["sha256"]
                quotes.extend(parsed.to_dict("records"))
                audit.update(**detail, parse_status="ok", error="")
            except (ValueError, UnicodeError) as error:
                audit.update(parse_status="rejected", error=str(error))
        page_audits.append(audit)
    processed = {item["date"] for item in page_audits}
    page_audits.extend(
        {"date": date, "parse_status": "not_attempted", "error": "capture stopped before this date"}
        for date in expected_dates
        if date not in processed
    )
    if not quotes:
        raise ValueError("No structurally valid source tables; see retained capture manifest")
    # Build once: pandas concat of all-NaT blocks hits NumPy's generic-unit deprecation.
    parsed = pd.DataFrame(quotes)
    curves = pd.read_csv(paths["curves"], parse_dates=["date"])
    diagnostics = match_spot_diagnostics(parsed, curves)
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    holidays = diagnostics.date.eq(plan["probe"]["holiday_negative_control"])
    holiday_audit = [
        a for a in page_audits if a["date"] == plan["probe"]["holiday_negative_control"]
    ]
    if diagnostics.loc[holidays, "quote_status"].eq("quote_ok").any() or any(
        a.get("table_rows", 0) > 0 for a in holiday_audit
    ):
        raise ValueError("Holiday negative control unexpectedly produced a diagnostic")
    diagnostics["observed_futures_session"] = diagnostics.date.isin(calendar.date)
    if (diagnostics.diagnostic_available & ~diagnostics.observed_futures_session).any():
        raise ValueError("Diagnostic is outside the frozen futures session calendar")
    coverage = []
    for symbol, group in diagnostics.groupby("symbol", sort=False):
        available = group.loc[group.diagnostic_available]
        coverage.append(
            {
                "symbol": symbol,
                "code": group.code.iloc[0],
                "kind": group.kind.iloc[0],
                "planned_dates": len(expected_dates),
                "parsed_dates": len(group),
                "valid_traded_quotes": int(group.quote_status.eq("quote_ok").sum()),
                "same_date_contract_diagnostics": len(available),
                "strict_factor_observations": 0,
                "clock_certified": False,
            }
        )
    coverage = pd.DataFrame(coverage)
    valid = diagnostics.loc[diagnostics.diagnostic_available]
    assets_per_date = valid.groupby("date").code.nunique()
    summary = {
        "kind": "physical_spot_source_readiness_only",
        "version": __version__,
        "protocol_registered_utc": plan["registered_utc"],
        "planned_dates": len(expected_dates),
        "http_ok_responses": sum(item["status"] == "http_ok" for item in records),
        "captured_responses": len(records),
        "parsed_date_tables": sum(row["parse_status"] == "ok" for row in page_audits),
        "valid_traded_quotes": int(diagnostics.quote_status.eq("quote_ok").sum()),
        "same_date_contract_diagnostics": len(valid),
        "distinct_commodities": int(valid.code.nunique()),
        "maximum_commodities_per_date": int(assets_per_date.max()) if len(assets_per_date) else 0,
        "minimum_assets_for_inference_unchanged": study["min_inference_assets_unchanged"],
        "strict_factor_observations": 0,
        "prediction_tests_added": 0,
        "predictive_statistics_computed": False,
        "registered_exploration_family_size_unchanged": len(
            pd.read_csv(paths["prior_exploration"])
        ),
        "historical_first_release_certified": False,
        "spot_basis_claim_eligible": False,
        "decision": "do_not_bulk_collect_or_activate_cross_sectional_spot_basis_yet",
        "inputs": {
            name: {"path": str(paths[name]), "sha256": sha256(paths[name])}
            for name in plan["input_hashes"]
        },
        "protocol_sha256": sha256(config),
        "sources_sha256": sha256(sources),
        "source_manifest_sha256": sha256(manifest_path),
        "code_sha256": {
            file.name: sha256(file)
            for file in (Path(__file__), Path(__file__).with_name("spot.py"))
        },
    }
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    diagnostics.to_csv(output / "quote_diagnostics.csv", index=False)
    pd.DataFrame(page_audits).to_csv(output / "source_audit.csv", index=False)
    coverage.to_csv(output / "coverage.csv", index=False)
    documents = [item for item in records if item["mode"] == "document"]
    pd.DataFrame(documents).drop(columns=["headers"], errors="ignore").to_csv(
        output / "source_documents.csv", index=False
    )
    source_links = " · ".join(
        f'<a href="{escape(item["url"], quote=True)}">{escape(item["id"])}</a>'
        for item in documents
    )
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>真实现货数据准入 · v{__version__}</title>
<style>body{{font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#193348;margin:auto;max-width:1180px;padding:36px 24px}}h1{{font-size:36px}}h2{{margin-top:38px}}.card{{background:white;border:1px solid #dbe5eb;border-radius:16px;padding:22px;margin:22px 0;overflow:auto}}.tag{{color:#835519;background:#fff0d1;padding:7px 14px;border-radius:20px}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:9px 12px;text-align:left;border-bottom:1px solid #dde7ee}}th{{background:#eaf2f7}}a{{color:#126b83}}code{{background:#eaf2f7}}.small{{font-size:13px;color:#526a7b}}</style>
<p>Commodity Factor Lab · 数据来源准入 · v{__version__}</p><h1>价格取得了，因子尚未获准</h1>
<p><span class="tag">仅检查数据 · 未计算 IC、p 值或投资收益</span></p>
<div class="card"><b>{summary["planned_dates"]} 个固定日期 · {summary["same_date_contract_diagnostics"]} 条同日期合约报价比较 · 最多 {summary["maximum_commodities_per_date"]} 个品种</b>
<p>两种黄金报价属于同一个品种。即使三份报价齐全，也只有黄金和白银两个品种，达不到既定截面至少八品种的门槛。本轮新增预测检验为零，已登记探索族仍为 {summary["registered_exploration_family_size_unchanged"]} 项。</p></div>
<h2>报价覆盖</h2><div class="card">{spot_table(coverage)}</div>
<p>样本含春节休市负对照；缺行、零成交、空价格均保留缺失，不向前填充，不以延期合约替代。小样本用于决定能否扩采，不能外推全期覆盖率。</p>
<h2>仍未通过的准入条件</h2><div class="card"><ul>
<li>网页中的行情日期及交易结束时间不能证明首次发布。当前下载时间、HTTP 修改时间均未倒填为历史可用时间。</li>
<li>黄金 Au99.95/Au99.99 为现货实盘 T+0；白银 Ag99.99 为实物 T+2。Au(T+D)、Ag(T+D)、mAu(T+D) 和国际板 iAu99.99 不替代本次映射。</li>
<li>价格按元/克或元/千克比较；交易手数、交割重量不能再乘入报价。牌号相近不等于同品牌、同库、同税后可交割价值，历史完整映射尚未建立。</li>
<li>现货收盘与期货结算使用不同时间及定价窗口。下表只是跨市场报价诊断，不是同步套利价差、便利收益或可执行盈亏。</li>
<li>2025 年第11号黄金税收公告自11月1日起实施，以实际交割出库时间适用。样本仅标注日期区间，不能按报价日期直接判定税负，也未按一个固定税率调整历史价格。</li></ul></div>
<h2>逐笔价格比较</h2><p><code>log(现货收盘价 / 原冻结选约结算价)</code>，仅价格和成交量检查通过才列数；严格匹配因子仍全缺失。</p>
<div class="card">{spot_table(diagnostics[["date", "symbol", "quote_status", "close", "volume_as_published", "front_contract", "reference_settle", "diagnostic_log_quote_ratio", "claim_eligible"]])}</div>
<h2>源响应与日期身份</h2><div class="card">{spot_table(pd.DataFrame(page_audits)[["date", "parse_status", "table_rows", "date_identity", "publisher_clock"]])}</div>
<p>保留原 HTML、响应头、采样起止、哈希和冻结配置。官方静态表已做日期身份验证；旧页面正文日期与新页面每行日期分别校验，查询错日或重复行会拒绝。规格文件是当前所见版本，不能认证完整历史规则。</p>
<p>官方规格与制度页面：{source_links}。<a href="source_documents.csv">本机捕获清单与哈希</a>。期货黄金和白银报价单位另见<a href="https://www.shfe.com.cn/reports/tradedata/">上期所行情说明</a>。</p>
<h2>下一准入节点</h2><p>暂不批量拉取全年现货。优先找到有历史发布证据、地区/牌号/税口径清楚的工业品现货来源，再固定扩样计划。若只能取得两种贵金属，须另立低维时间序列研究协议，不能临时降低当前截面门槛。</p>
<p class="small"><a href="quote_diagnostics.csv">逐条报价与缺失</a> · <a href="coverage.csv">覆盖表</a> · <a href="source_audit.csv">源检查</a> · <a href="run.json">输入、源码与决定</a></p>
<p class="small">冻结时间 {escape(plan["registered_utc"])}；本机采样日期 2026-10-29。原始行情不随公开源码包分发。</p></html>"""
    (output / "index.html").write_text(report, encoding="utf-8")
    return output
