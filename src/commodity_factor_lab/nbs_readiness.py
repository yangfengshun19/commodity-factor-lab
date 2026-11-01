"""Offline source, specification and timing assay for industrial reference prices."""

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from . import __version__
from .contract_report import table
from .nbs import align_release_events, map_industrial_references, parse_nbs_release
from .nbs_ingest import digest, nbs_plan


def display(frame: pd.DataFrame) -> str:
    labels = {
        "period": "统计期",
        "period_end": "统计期结束",
        "source_declared_published_at": "正文发布时刻（UTC）",
        "first_signal_date": "首个可对齐信号日",
        "previous_signal_date": "前一个信号日",
        "related_code": "相关期货品种",
        "product": "公布产品及规格",
        "reference_price": "本期参考价",
        "mapping_status": "参考映射状态",
        "reference_mapping_usable": "可作相关工业品参考",
        "in_existing_universe": "在原研究池中",
        "periods": "抽查期数",
        "records": "记录数",
        "products": "价格产品数",
        "monitoring_scope": "调查范围",
        "id": "来源",
        "status": "状态",
        "first_signal_cutoff": "首个信号截止（UTC）",
    }
    values = {
        "reference_only": "相关参考价",
        "specification_break": "规格变动，分段保存",
        "different_product_form": "产品形态不同，排除映射",
        "ambiguous_pulp_grade": "纸浆口径不同，排除映射",
        "source_clock_event_alignment_only": "按公告时钟对齐",
        "outside_frozen_calendar": "发布晚于原行情区间",
        "nationwide_31_provinces": "全国31省",
        "24_provinces": "24省",
        "http_ok": "已捕获",
        "ok": "通过",
    }
    result = frame.copy()
    for column in result.select_dtypes(include="object"):
        result[column] = result[column].map(lambda value: values.get(value, value))
    return table(result.rename(columns=labels))


def run_nbs_readiness(config: Path) -> Path:
    plan, paths = nbs_plan(config)
    receipt = paths["raw"] / "manifest.json"
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        manifest["protocol_sha256"] != digest(config)
        or manifest["planned_sources"] != plan["sources"]
    ):
        raise ValueError("NBS capture does not match the frozen source assay")
    if pd.Timestamp(plan["registered_utc"]) >= pd.Timestamp(manifest["created_utc"]):
        raise ValueError("Local capture must follow the fixed source sampling plan")
    planned = {item["id"]: item for item in plan["sources"]}
    if len({item["id"] for item in manifest["records"]}) != len(manifest["records"]):
        raise ValueError("Duplicate NBS capture receipt")
    prices, specifications, audits, documents = [], [], [], []
    for item in manifest["records"]:
        if item["id"] not in planned or any(item[k] != v for k, v in planned[item["id"]].items()):
            raise ValueError("Captured NBS source identity differs from its plan")
        path = paths["raw"] / item.get("file", "missing.html")
        if not path.resolve().is_relative_to(paths["raw"]):
            raise ValueError("Source response path escaped its capture directory")
        if item["status"] != "http_ok" or digest(path) != item["sha256"]:
            raise ValueError(
                "A planned NBS source is unavailable or has changed; inspect its receipt"
            )
        if item["kind"] == "methodology":
            documents.append(
                {
                    "id": item["id"],
                    "url": item["url"],
                    "sha256": item["sha256"],
                    "status": item["status"],
                }
            )
            continue
        frame, specs, audit = parse_nbs_release(path.read_bytes(), item["period"])
        frame["source_url"], frame["source_sha256"] = item["url"], item["sha256"]
        prices.extend(frame.to_dict("records"))
        specifications.extend(specs.to_dict("records"))
        match = re.search(r"/t(20\d{6})_", item["url"])
        url_date = pd.to_datetime(match[1], format="%Y%m%d") if match else pd.NaT
        published_date = (
            audit["source_declared_published_at"]
            .tz_convert("Asia/Shanghai")
            .tz_localize(None)
            .normalize()
        )
        audits.append(
            {
                **audit,
                "url": item["url"],
                "sha256": item["sha256"],
                "url_path_date": url_date,
                "url_date_differs_from_published_date": bool(url_date != published_date)
                if not pd.isna(url_date)
                else None,
            }
        )
    if len(manifest["records"]) != len(plan["sources"]):
        raise ValueError("Complete the fixed bounded assay before publishing its coverage")
    prices = pd.DataFrame(prices)
    excluded = set(plan["negative_controls"]["unpublished_periods"])
    if not excluded.isdisjoint(prices.period):
        raise ValueError("A cancelled period must remain absent; no later-change inversion")
    curves = pd.read_csv(paths["curves"], parse_dates=["date"])
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    mapped = map_industrial_references(prices, set(curves.code.unique()))
    events = align_release_events(mapped, calendar)
    in_calendar = events.first_signal_date.notna()
    if any(
        pd.to_datetime(events.loc[in_calendar, "first_signal_cutoff"], utc=True)
        < pd.to_datetime(events.loc[in_calendar, "source_declared_published_at"], utc=True)
    ):
        raise ValueError("A release reached a signal before its declared publication clock")
    previous = events.previous_signal_cutoff.notna()
    if any(
        pd.to_datetime(events.loc[previous, "previous_signal_cutoff"], utc=True)
        >= pd.to_datetime(events.loc[previous, "source_declared_published_at"], utc=True)
    ):
        raise ValueError("The selected event is not the first observed eligible cutoff")
    related = mapped.loc[mapped.related_code.ne("")].copy()
    usable = related.loc[related.reference_mapping_usable]
    coverage = (
        related.groupby(["related_code", "mapping_status", "in_existing_universe"], sort=False)
        .agg(periods=("period", "nunique"), records=("reference_price", "count"))
        .reset_index()
    )
    input_events = usable.merge(
        events[["period", "first_signal_date", "first_signal_cutoff", "status"]],
        on="period",
        validate="many_to_one",
    )
    # This is an event assay. Never fill the unsampled periods into a daily predictor panel.
    input_events["sample_is_full_history"] = False
    input_events["strict_spot_basis_eligible"] = False
    summary = {
        "version": __version__,
        "kind": plan["study"]["kind"],
        "captured_responses": len(manifest["records"]),
        "price_releases": len(audits),
        "price_records": len(prices),
        "specification_records": len(specifications),
        "existing_universe_reference_codes": sorted(usable.related_code.unique()),
        "existing_universe_reference_records": len(usable),
        "reference_records_with_an_in_calendar_event": int(
            input_events.first_signal_date.notna().sum()
        ),
        "source_clock_events_in_calendar": int(in_calendar.sum()),
        "url_date_mismatch_releases": sum(
            a["url_date_differs_from_published_date"] is True for a in audits
        ),
        "unpublished_negative_controls_absent": sorted(excluded),
        "reconstructed_missing_periods": 0,
        "daily_fill_performed": False,
        "prediction_tests_added": 0,
        "predictive_statistics_computed": False,
        "registered_exploration_family_size_unchanged": len(
            pd.read_csv(paths["prior_exploration"])
        ),
        "minimum_assets_unchanged": plan["study"]["minimum_assets_unchanged"],
        "historical_first_vintage_certified": False,
        "strict_spot_basis_claim_eligible": False,
        "decision": "source_viable_for_separate_period_reference_study_after_full_coverage_and_protocol",
        "inputs": {
            key: {"path": str(paths[key]), "sha256": digest(paths[key])}
            for key in plan["input_hashes"]
        },
        "protocol_sha256": digest(config),
        "raw_manifest_sha256": digest(receipt),
        "code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("nbs.py", "nbs_ingest.py", "nbs_readiness.py", "spot.py")
        },
    }
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in {
        "published_reference_prices": mapped,
        "product_specifications": pd.DataFrame(specifications),
        "release_events": events,
        "reference_input_events": input_events,
        "coverage": coverage,
        "source_audit": pd.DataFrame(audits),
        "methodology_sources": pd.DataFrame(documents),
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    links = " · ".join(
        f'<a href="{escape(item["url"], quote=True)}">{escape(item["id"])}</a>'
        for item in plan["sources"]
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>工业品参考数据接入 · v{__version__}</title>
<style>body{{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#193348;max-width:1180px;margin:36px auto;padding:24px}}h1{{font-size:36px}}section{{background:white;border:1px solid #dbe5eb;border-radius:16px;margin:24px 0;padding:24px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #dce5ec;text-align:left}}th{{background:#eaf2f7}}a{{color:#126b83}}.tag{{color:#26705e}}.small{{font-size:13px;color:#546e80}}</style>
<p>Commodity Factor Lab · v{__version__} · 数据接入</p><h1>工业品参考数据接入已跑通</h1><p class="tag">{len(audits)}份固定旬报 · {len(prices)}条价格记录 · 覆盖原研究池{usable.related_code.nunique()}个品种</p>
<section><h2>现在具体多了什么数据</h2><p>已读取国家统计局公开的生产资料价格旬报及产品规格。原池可作相关参考的品种为铜、铝、锌、螺纹钢、热卷、天然橡胶、纸浆。保留本期价格、统计期间、正文发布时间、调查范围与规格分段；不是每天都实际成交的现货价。{len(prices)}条是不同商品/期次的记录，不是独立市场样本数。</p><p>这条官方来源值得继续扩采并另立低频参考信息研究。当前八份是固定小样本，尚未构成完整历史；本轮没有计算预测效果或投资收益，既有{summary["registered_exploration_family_size_unchanged"]}项探索族不变。</p></section>
<section><h2>发布时间防错已经落到程序</h2><p>统计期结束 → 正文发布 → 首个信号截止。按原冻结期货日历对齐；网页地址日期与发布时间不同，不能拿地址日期替代。下表发布时间为UTC，加8小时为北京时间。</p>{display(events[["period", "period_end", "source_declared_published_at", "first_signal_date", "previous_signal_date", "status"]])}<p>例如2024年1月下旬的数据，正文到2月4日才发布，最早对齐2月5日的期货信号。2025年12月下旬到2026年1月4日才发布，完全不进入截至2025年底的原样本。</p></section>
<section><h2>两个未发布期次保持空缺</h2><p>官方说明2024年2月中旬因春节取消发布；2025年发布日程也取消1月下旬。后续报告即使给出相对这些期次的涨跌，也不能倒算后回填到未发布时点。本程序未生成这两期记录，没有用稀疏样本填出一条完整日频因子。</p></section>
<section><h2>规格和调查范围分别保存</h2>{display(coverage)}<p>2020年样本为24省口径，后期为全国31省，序列身份保留这一差异。旧螺纹直径范围、热轧薄板与板卷、漂白化学浆与进口针叶浆分开；铅有公开报价但不在原十品种池中，不为了凑数量加入。原八品种截面门槛不变。</p></section>
<section><h2>如何使用这批数据</h2><p>可作为带公告时点和规格标签的工业品价格参考，用于后续单独设计的供需或价格信息研究。全国经营企业的综合报价包含流通费用、利润和税费，不能当成某一交割库的可执行报价；本轮也没有把旬价与期货结算相除称为严格现货基差。</p><p>正文发布时间是当前页面明确声明的发布时钟，优于猜测URL日期；当前下载仍不能认证所有历史初发版本。完整扩采时还须核对期次清单、制度变更、实际发布日程与修订记录。商业来源SMM的历史接口需要授权；本轮没有开户、购买或尝试获取锁定报价。</p></section>
<section><h2>复核文件</h2><p><a href="published_reference_prices.csv">全部公布价格与映射状态</a> · <a href="product_specifications.csv">规格说明</a> · <a href="release_events.csv">公告与信号对齐</a> · <a href="source_audit.csv">原页面核查</a> · <a href="run.json">冻结输入与源码哈希</a></p><p class="small">官方原文：{links}。原响应保存在本机，公开源码包不带原始行情。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
