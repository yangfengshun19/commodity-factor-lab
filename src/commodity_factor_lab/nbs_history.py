"""Normalize the fixed official price archive, account for every period, and profile breaks."""

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from . import __version__
from .contract_report import table
from .nbs import align_release_events, map_industrial_references, parse_nbs_release
from .nbs_archive import freeze_history_catalog, history_plan
from .nbs_ingest import digest


def series_identity_changes(mapped: pd.DataFrame) -> pd.DataFrame:
    """Metadata changes between observed publications; never assert an exact legal effective day."""
    rows = []
    related = mapped.loc[mapped.related_code.ne("")].sort_values(["related_code", "period_start"])
    if related.duplicated(["period", "related_code"]).any():
        raise ValueError("Multiple reference products mapped to the same code/period need review")
    for code, group in related.groupby("related_code", sort=False):
        previous = None
        for row in group.itertuples():
            if previous is not None and row.comparability_key != previous.comparability_key:
                rows.append(
                    {
                        "code": code,
                        "previous_observed_period": previous.period,
                        "first_observed_new_identity": row.period,
                        "previous_product": previous.product,
                        "current_product": row.product,
                        "previous_scope": previous.monitoring_scope,
                        "current_scope": row.monitoring_scope,
                        "product_label_changed": row.product != previous.product,
                        "scope_changed": row.monitoring_scope != previous.monitoring_scope,
                        "specification_text_changed": row.specification_description
                        != previous.specification_description,
                        "previous_key": previous.comparability_key,
                        "current_key": row.comparability_key,
                        "exact_effective_date_certified": False,
                    }
                )
            previous = row
    return pd.DataFrame(
        rows,
        columns=[
            "code",
            "previous_observed_period",
            "first_observed_new_identity",
            "previous_product",
            "current_product",
            "previous_scope",
            "current_scope",
            "product_label_changed",
            "scope_changed",
            "specification_text_changed",
            "previous_key",
            "current_key",
            "exact_effective_date_certified",
        ],
    )


def history_table(frame: pd.DataFrame) -> str:
    return table(
        frame.rename(
            columns={
                "year": "年份",
                "expected_periods": "日历期次",
                "published": "已解析公告",
                "cancelled": "官方取消",
                "parse_rejected": "解析未通过",
                "prices": "价格记录",
                "specifications": "规格记录",
                "code": "品种",
                "periods": "有公布价格的期次",
                "reference_usable": "相关参考价期次",
                "identity_segments": "元数据身份数",
                "first": "首期",
                "last": "末期",
                "previous_observed_period": "前次观测期",
                "first_observed_new_identity": "首次观测新身份期",
                "previous_product": "原产品规格",
                "current_product": "新产品规格",
                "scope_changed": "调查范围变化",
                "specification_text_changed": "规格说明文本变化",
                "period": "统计期",
                "error": "问题",
                "status": "状态",
            }
        )
    )


def run_nbs_history(config: Path, rules: Path) -> Path:
    plan, paths = history_plan(config)
    catalog = freeze_history_catalog(config, rules)
    for name, expected in catalog["metadata_inputs_sha256"].items():
        if digest(paths["raw"] / name) != expected:
            raise ValueError("Frozen discovery evidence changed: " + name)
    discovery = json.loads((paths["raw"] / "discovery/manifest.json").read_text(encoding="utf-8"))
    documents = {item["id"]: item for item in discovery["records"]}
    for cancellation in catalog["cancellations"]:
        source = documents[cancellation["schedule_id"]]
        if digest(paths["raw"] / "discovery" / source["file"]) != cancellation["source_sha256"]:
            raise ValueError("Official cancellation evidence changed")
    source_directory = paths["raw"] / "releases"
    manifest_path = source_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != digest(config) or manifest["catalog_sha256"] != digest(
        paths["raw"] / "catalog.json"
    ):
        raise ValueError("Article capture does not match the frozen history catalog")
    records = {item["id"]: item for item in manifest["records"]}
    if len(records) != len(manifest["records"]) or set(records) != {
        item["period"] for item in catalog["releases"]
    }:
        raise ValueError("Complete the frozen source capture before normalizing its full coverage")
    prices, specifications, audits, rejected = [], [], [], []
    for index, source in enumerate(catalog["releases"]):
        item = records[source["period"]]
        path = (source_directory / item["file"]).resolve()
        if (
            not path.is_relative_to(source_directory)
            or item["status"] != "http_ok"
            or item["url"] != source["url"]
            or digest(path) != item["sha256"]
        ):
            raise ValueError("Article source identity or captured hash changed")
        try:
            frame, spec, audit = parse_nbs_release(path.read_bytes(), source["period"])
            frame["source_url"], frame["source_sha256"] = source["url"], item["sha256"]
            prices.extend(frame.to_dict("records"))
            specifications.extend(spec.to_dict("records"))
            audits.append({**audit, "source_url": source["url"], "source_sha256": item["sha256"]})
        except (ValueError, UnicodeError) as error:
            rejected.append(
                {
                    "period": source["period"],
                    "error": str(error),
                    "source_url": source["url"],
                    "source_sha256": item["sha256"],
                }
            )
        if index % 25 == 0 or index + 1 == len(catalog["releases"]):
            print(
                f"Normalized source articles {index + 1}/{len(catalog['releases'])}; rejected {len(rejected)}",
                flush=True,
            )
    if not prices:
        raise ValueError("No source prices passed parsing; retain and review the captured archive")
    prices = pd.DataFrame(prices)
    curves = pd.read_csv(paths["curves"], parse_dates=["date"])
    calendar_frame = pd.read_csv(paths["calendar"], parse_dates=["date"])
    mapped = map_industrial_references(prices, set(curves.code.unique()))
    events = align_release_events(mapped, calendar_frame)
    changes = series_identity_changes(mapped)
    successful = {item["period"] for item in audits}
    failed = {item["period"]: item["error"] for item in rejected}
    ledger = pd.DataFrame(catalog["period_ledger"])
    ledger["status"] = ledger.apply(
        lambda row: (
            "published_and_parsed"
            if row.period in successful
            else "published_parse_rejected"
            if row.period in failed
            else row.status
        ),
        axis=1,
    )
    ledger["error"] = ledger.period.map(failed).fillna("")
    yearly = []
    for year in range(2020, 2026):
        group = ledger.loc[ledger.period.str.startswith(str(year))]
        parsed = int(group.status.eq("published_and_parsed").sum())
        yearly.append(
            {
                "year": year,
                "expected_periods": len(group),
                "published": parsed,
                "cancelled": int(group.status.eq("officially_cancelled").sum()),
                "parse_rejected": int(group.status.eq("published_parse_rejected").sum()),
                "prices": int(prices.period.str.startswith(str(year)).sum()),
                "specifications": sum(
                    item["period"].startswith(str(year)) for item in specifications
                ),
            }
        )
    yearly = pd.DataFrame(yearly)
    related = mapped.loc[mapped.related_code.ne("")]
    reference_coverage = []
    for code in sorted(set(curves.code.unique())):
        group = related.loc[related.related_code.eq(code)]
        reference_coverage.append(
            {
                "code": code,
                "periods": group.period.nunique(),
                "reference_usable": int(group.reference_mapping_usable.sum()),
                "identity_segments": group.comparability_key.nunique(),
                "first": group.period.iloc[0] if len(group) else "",
                "last": group.period.iloc[-1] if len(group) else "",
            }
        )
    reference_coverage = pd.DataFrame(reference_coverage)
    input_events = mapped.loc[mapped.reference_mapping_usable].merge(
        events[["period", "first_signal_date", "first_signal_cutoff", "status"]],
        on="period",
        validate="many_to_one",
    )
    input_events["source_is_complete_normalized_history"] = not rejected
    # Keep one observation per release/product. No artificial daily repetition, interpolation or returns.
    complete = (
        not rejected
        and len(records) == len(catalog["releases"])
        and len(ledger) == 216
        and ledger.status.isin(["published_and_parsed", "officially_cancelled"]).all()
    )
    summary = {
        "version": __version__,
        "kind": "complete_official_period_reference_history",
        "expected_calendar_periods": len(ledger),
        "captured_articles": len(records),
        "officially_cancelled_periods": len(catalog["cancellations"]),
        "parsed_articles": len(audits),
        "parse_rejected_articles": len(rejected),
        "price_records": len(prices),
        "specification_records": len(specifications),
        "complete_normalized_history": bool(complete),
        "release_http_requests": manifest["requests"],
        "seed_responses_reused": sum(item.get("reused_seed", False) for item in records.values()),
        "raw_release_bytes": sum(item.get("bytes", 0) for item in records.values()),
        "reference_usable_records": int(mapped.reference_mapping_usable.sum()),
        "reference_codes_in_existing_universe": sorted(
            mapped.loc[mapped.reference_mapping_usable, "related_code"].unique()
        ),
        "source_clock_events_in_original_calendar": int(events.first_signal_date.notna().sum()),
        "metadata_identity_changes": len(changes),
        "daily_forward_fill_performed": False,
        "missing_period_reconstruction_performed": False,
        "prediction_tests_added": 0,
        "predictive_statistics_computed": False,
        "registered_exploration_family_size_unchanged": len(
            pd.read_csv(paths["prior_exploration"])
        ),
        "minimum_assets_unchanged": plan["history"]["minimum_assets_unchanged"],
        "historical_first_vintage_certified": False,
        "strict_spot_basis_claim_eligible": False,
        "protocol_sha256": digest(config),
        "rules_sha256": digest(rules),
        "catalog_sha256": digest(paths["raw"] / "catalog.json"),
        "release_manifest_sha256": digest(manifest_path),
        "inputs": {
            name: {"path": str(paths[name]), "sha256": digest(paths[name])}
            for name in plan["input_hashes"]
        },
        "code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ["nbs.py", "nbs_archive.py", "nbs_search.py", "nbs_history.py", "spot.py"]
        },
    }
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in {
        "published_reference_prices": mapped,
        "product_specifications": pd.DataFrame(specifications),
        "release_events": events,
        "reference_input_events": input_events,
        "period_ledger": ledger,
        "yearly_coverage": yearly,
        "reference_coverage": reference_coverage,
        "identity_changes": changes,
        "source_audit": pd.DataFrame(audits),
        "parse_rejections": pd.DataFrame(
            rejected, columns=["period", "error", "source_url", "source_sha256"]
        ),
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    title = "六年工业品参考价格历史已建成" if complete else "历史原文已采集，部分解析仍待修复"
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NBS完整历史 · v{__version__}</title><style>body{{font:16px/1.8 system-ui,'Microsoft YaHei',sans-serif;max-width:1200px;margin:35px auto;padding:24px;background:#f4f7fa;color:#193348}}h1{{font-size:36px}}section{{background:white;border:1px solid #dbe5eb;border-radius:16px;padding:24px;margin:24px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #dce5ec;text-align:left}}th{{background:#eaf2f7}}a{{color:#126b83}}.small{{font-size:13px;color:#536b7b}}</style><p>Commodity Factor Lab · v{__version__} · 2020—2025</p><h1>{title}</h1>
<section><h2>从8期抽查扩大到六年历史</h2><p><b>216个日历期次，{len(records)}份官方原文，{len(catalog["cancellations"])}期有官方取消证据。</b>已解析{len(audits)}期、{len(prices):,}条价格及{len(specifications):,}条规格记录。来源通过官方目录与官网公开检索定位，检索摘要不作为价格。原文、响应头、抓取时刻及哈希保留。</p>{history_table(yearly)}</section>
<section><h2>原研究池的参考覆盖</h2>{history_table(reference_coverage)}<p>七个工业品有相关参考，铅保留在原始公布数据中但不加入原研究池。黄金、白银、沥青没有本源映射，不能补零或借用其他商品。全国旬度参考价包含流通费用、利润和税费，不能直接当成交割库可执行报价。</p></section>
<section><h2>规格变化不被抹平</h2><p>共记录{len(changes)}次相邻公告间元数据身份变化，可能来自产品、规格说明文本或调查范围。它们是待核对的变化窗口，不是已经认证的制度生效日；文本差异也不必然代表经济规格变化。</p>{history_table(changes[["code", "previous_observed_period", "first_observed_new_identity", "previous_product", "current_product", "scope_changed", "specification_text_changed"]])}</section>
<section><h2>信息时点与缺期</h2><p>正文发布时间独立于统计期、URL日期和下载日期，按原冻结期货信号截止对齐。{summary["source_clock_events_in_original_calendar"]}次公告能落在原日历中，2025年12月下旬在2026年才公布，不填回2025年。取消期保留空缺，后续涨跌不倒算回填；本报告不生成重复的日频价格。</p><p>正文时钟是当前页面声明，本次历史下载仍不能认证全部初发版本。年度日程用来核对取消与调整，不用计划时间替代实际正文时钟。</p></section>
<section><h2>本轮完成与下一步</h2><p>已完成固定范围的来源历史建设；新增预测检验为零，既有{summary["registered_exploration_family_size_unchanged"]}项探索族不变。接下来另立已发布工业品价格信息的研究协议，确定频率、基准与检验后再计算效果。原八品种门槛保持，七个参考品种不直接启动旧截面实验。</p><p><a href="period_ledger.csv">全部216期状态</a> · <a href="published_reference_prices.csv">全部价格与映射</a> · <a href="product_specifications.csv">产品规格</a> · <a href="release_events.csv">公告时点对齐</a> · <a href="identity_changes.csv">元数据变化</a> · <a href="parse_rejections.csv">解析问题</a> · <a href="run.json">范围与源码哈希</a></p><p class="small">原行情不随公开源码包分发；记录数与程序检查次数都不是独立统计样本数。冻结方案 {escape(config.name)}；原始材料和以前的研究结果保留。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
