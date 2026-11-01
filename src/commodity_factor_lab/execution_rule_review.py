"""Trace dated margin floors and auction timing; scenario agreement is not vintage certification."""

import json
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from . import __version__
from .contract_report import table
from .execution_readiness import parse_notice, parse_parameter_snapshot, source_text
from .execution_sources import capture_public_sources
from .nbs_ingest import digest

IDS = {
    "rule_release_20240823",
    "cu_rules_202409",
    "rb_rules_202409",
    "risk_rules_202409",
    "trading_rules_202409",
    "cu_rules_later_negative",
    "monthly_index",
}


def review_plan(config: Path) -> tuple[dict, dict]:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {name: (root / value).resolve() for name, value in plan["paths"].items()}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Rule-review paths must stay within its workspace")
    study, policy = plan["study"], plan["collection"]
    if (
        study["kind"] != "historical_margin_rule_components_and_session_review"
        or study["contracts"] != ["cu2503", "cu2504", "rb2505"]
        or study["dates"] != ["2025-01-24", "2025-01-27", "2025-02-05", "2025-02-06"]
        or study["ordinary_reference"]
        != "observed_20250124_trading_snapshot_not_certified_notice_baseline"
        or study["recovery_branches"]
        != ["released_at_reopening_settlement", "not_released_at_reopening_settlement"]
        or any(
            study[name]
            for name in (
                "complete_notice_inventory_certified",
                "historical_first_vintage_certified",
                "actual_opening_clock_certified",
            )
        )
        or study["prediction_tests_added"] != 0
        or study["registered_exploration_family_size_unchanged"] != 15
        or policy["maximum_new_requests"] != 7
        or policy["minimum_interval_seconds"] < 1
        or not 1 <= policy["maximum_response_bytes"] <= 3_000_000
        or policy["stop_after_consecutive_failures"] != 3
    ):
        raise ValueError(
            "Keep the fixed rule-only scope, two scenarios and qualification boundaries"
        )
    if len(plan["sources"]) != 7 or {item["id"] for item in plan["sources"]} != IDS:
        raise ValueError("Keep all registered rule sources including the later-version control")
    for item in plan["sources"]:
        url = urlsplit(item["url"])
        if (
            url.scheme != "https"
            or url.hostname not in {"www.shfe.com.cn", "www.shfe.cn"}
            or url.port is not None
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError("Rule capture requires explicit official public URLs")
    for name, expected in plan["input_hashes"].items():
        if digest(paths[name]) != expected:
            raise ValueError(f"Frozen rule-review input changed: {name}")
    if paths["output"].is_relative_to(paths["raw"]):
        raise ValueError("Keep review output outside raw sources")
    links = json.loads(paths["discovered_links"].read_text(encoding="utf-8"))
    indexed = {item["url"] for item in links["links"]}
    for item in plan["sources"]:
        if "/historicalversion/" in item["url"] and item["url"] not in indexed:
            raise ValueError("Rule article was not found in the saved official directory")
    return plan, paths


def capture_rule_review(config: Path) -> dict:
    plan, paths = review_plan(config)
    return capture_public_sources(config, plan, paths)


def _unique(pattern: str, text: str) -> str:
    values = set(re.findall(pattern, text))
    if len(values) != 1:
        raise ValueError("Missing or conflicting active rule text")
    return values.pop()


def rule_identity(content: bytes) -> dict:
    text = source_text(content)
    publication = _unique(r"发布日期[:：](\d{4}-\d{2}-\d{2})", text)
    dates = re.findall(r"自(\d{4})年(\d+)月(\d+)日起(?:开始)?实施", text)
    dates = {f"{int(y):04}-{int(m):02}-{int(d):02}" for y, m, d in dates}
    if len(dates) != 1:
        raise ValueError("Rule needs an unambiguous declared effective date")
    return {
        "source_declared_date": publication,
        "declared_effective_date": dates.pop(),
        "historical_first_vintage_certified": False,
    }


def product_rule(content: bytes, code: str) -> dict:
    if code not in {"cu", "rb"}:
        raise ValueError("Only the registered CU and RB rule components are in scope")
    text = source_text(content)
    identity = rule_identity(content)
    identity.update(
        code=code,
        multiplier_tonnes=int(_unique(r"交易单位为(\d+)吨/手", text)),
        listed_floor=int(_unique(r"合约挂牌之日起(\d+)%", text)) / 100,
        month_before_floor=int(_unique(r"交割月前第一月的第一个交易日起(\d+)%", text)) / 100,
        delivery_month_floor=int(_unique(r"交割月份第一个交易日起(\d+)%", text)) / 100,
        last_two_sessions_floor=int(_unique(r"最后交易日前二个交易日起(\d+)%", text)) / 100,
    )
    return identity


def date_applicable(rule: dict, date) -> bool:
    day = pd.Timestamp(date)
    return day >= max(
        pd.Timestamp(rule["source_declared_date"]), pd.Timestamp(rule["declared_effective_date"])
    )


def known_margin_floor(components: dict[str, float], complete: bool = False) -> dict:
    """Combine already-normalized absolute floors. Unknown notices cannot become an exact rate."""
    if not components or type(complete) is not bool:
        raise ValueError("Explicit floor components and completeness are required")
    values = np.array(list(components.values()), dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or (values > 1).any():
        raise ValueError("Absolute margin floors must be finite decimal fractions in (0,1]")
    floor = float(values.max())
    return {
        "known_component_floor": floor,
        "binding_components": ";".join(
            name
            for name, value in components.items()
            if np.isclose(value, floor, atol=1e-14, rtol=1e-12)
        ),
        "fully_resolved_rate": floor if complete else None,
        "status": "complete_under_declared_inputs"
        if complete
        else "incomplete_components_lower_bound_only",
    }


def month_before_start(contract: str, calendar: pd.DataFrame) -> pd.Timestamp:
    matched = re.fullmatch(r"(?:cu|rb)(\d{2})(\d{2})", contract)
    if matched is None or not 1 <= int(matched[2]) <= 12:
        raise ValueError("Contract must carry a valid registered product and delivery month")
    month = pd.Timestamp(year=2000 + int(matched[1]), month=int(matched[2]), day=1) - pd.DateOffset(
        months=1
    )
    dates = pd.to_datetime(calendar.date)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Reference calendar must be chronological and unique")
    subset = dates.loc[dates.dt.to_period("M").eq(month.to_period("M"))]
    if subset.empty:
        raise ValueError("No complete reference month available for the maturity boundary")
    return subset.min()


def scheduled_auction(holiday_content: bytes, trading_content: bytes) -> dict:
    holiday, trading = source_text(holiday_content), source_text(trading_content)
    identity = rule_identity(trading_content)
    interval = re.search(r"2025年2月5日.{0,30}?(08:55)-(09:00)", holiday)
    duration = re.search(
        r"集合竞价前(\d+)分钟为合约买、卖指令申报时间,后(\d+)分钟为集合竞价撮合", trading
    )
    fallback = "集合竞价未产生成交价格的,以当日竞价交易的第一笔成交价为开盘价" in trading
    if interval is None or duration is None or not fallback:
        raise ValueError(
            "Missing dated auction interval, submission/matching phases or open-price fallback"
        )
    if not date_applicable(identity, "2025-02-05"):
        raise ValueError("Auction rule was not declared effective by the reviewed date")
    start = pd.Timestamp("2025-02-05T08:55:00+08:00")
    end = pd.Timestamp("2025-02-05T09:00:00+08:00")
    submission, matching = map(int, duration.groups())
    if end - start != pd.Timedelta(submission + matching, unit="m"):
        raise ValueError("Auction phase durations disagree with the dated holiday interval")
    return {
        "trading_date": "2025-02-05",
        "scheduled_auction_start": start.isoformat(),
        "scheduled_submission_cutoff_at": (start + pd.Timedelta(submission, unit="m")).isoformat(),
        "scheduled_matching_end": end.isoformat(),
        "submission_minutes": submission,
        "matching_minutes": matching,
        "daily_open_can_be_later_first_trade": fallback,
        "actual_open_at": None,
        "actual_execution_clock_certified": False,
    }


def run_rule_review(config: Path) -> Path:
    plan, paths = review_plan(config)
    manifest_path = paths["raw"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != digest(config) or len(manifest["records"]) != 7:
        raise ValueError("Captured rule set differs from the fixed scope")
    expected = {item["id"]: item for item in plan["sources"]}
    bodies, versions = {}, []
    if {item["id"] for item in manifest["records"]} != set(expected):
        raise ValueError("Missing or duplicate rule captures")
    for item in manifest["records"]:
        if (
            any(item.get(name) != value for name, value in expected[item["id"]].items())
            or item["status"] != "http_ok"
        ):
            raise ValueError("Rule source metadata or access state changed")
        if Path(item["file"]).name != item["file"]:
            raise ValueError("Raw rule file must be a direct child")
        path = paths["raw"] / item["file"]
        if digest(path) != item["sha256"]:
            raise ValueError("Raw rule body differs from its capture receipt")
        bodies[item["id"]] = path.read_bytes()
        record = {
            "id": item["id"],
            "url": item["url"],
            "sha256": item["sha256"],
            "historical_first_vintage_certified": False,
        }
        if item["kind"] == "rule":
            record.update(rule_identity(path.read_bytes()))
        versions.append(record)
    rules = {
        "cu": product_rule(bodies["cu_rules_202409"], "cu"),
        "rb": product_rule(bodies["rb_rules_202409"], "rb"),
        "later_cu_negative": product_rule(bodies["cu_rules_later_negative"], "cu"),
    }
    risk = source_text(bodies["risk_rules_202409"])
    if "交易保证金按照规定交易保证金比例中的最高值收取" not in risk:
        raise ValueError("Historical maximum-of-applicable-ratios evidence missing")
    release = source_text(bodies["rule_release_20240823"])
    if "自2024年10月23日实施" not in release:
        raise ValueError("Missing parent announcement effective-date evidence")
    old_manifest = json.loads(paths["seed_manifest"].read_text(encoding="utf-8"))
    old_records = {item["id"]: item for item in old_manifest["records"]}
    inputs = []
    for date in plan["study"]["dates"]:
        for kind in ("trading", "settlement"):
            key = f"{kind}_{date.replace('-', '')}"
            item = old_records[key]
            path = paths["seed"] / item["file"]
            if digest(path) != item["sha256"]:
                raise ValueError("Seed parameter snapshot changed")
            rows, _ = parse_parameter_snapshot(
                path.read_bytes(), kind, date.replace("-", ""), plan["study"]["contracts"]
            )
            inputs.extend(
                {**row, "source_id": key, "source_sha256": item["sha256"]} for row in rows
            )
    parameters = pd.DataFrame(inputs)
    base = (
        parameters.loc[parameters.date.eq("2025-01-24") & parameters.kind.eq("trading")]
        .set_index("contract")
        .spec_long_margin_rate.to_dict()
    )
    holiday_item = old_records["cny2025"]
    holiday_body = (paths["seed"] / holiday_item["file"]).read_bytes()
    if digest(paths["seed"] / holiday_item["file"]) != holiday_item["sha256"]:
        raise ValueError("Holiday source changed")
    holiday = parse_notice(holiday_body, "cny2025")
    auction = scheduled_auction(holiday_body, bodies["trading_rules_202409"])
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    cases, components = [], []
    for item in parameters.itertuples():
        code = re.match(r"[a-z]+", item.contract)[0]
        rule = rules[code]
        if not date_applicable(rule, item.date) or not date_applicable(
            rule_identity(bodies["risk_rules_202409"]), item.date
        ):
            raise ValueError("An earlier-effective product and risk rule is required")
        boundary = month_before_start(item.contract, calendar)
        for branch in plan["study"]["recovery_branches"]:
            floors = {
                "product_listed_floor": rule["listed_floor"],
                "observed_preholiday_reference": base[item.contract],
            }
            if item.date >= boundary:
                floors["month_before_delivery_floor"] = rule["month_before_floor"]
            holiday_active = item.date > pd.Timestamp("2025-01-24") or (
                item.date == pd.Timestamp("2025-01-24") and item.kind == "settlement"
            )
            after_reopen_close = item.date > pd.Timestamp("2025-02-05") or (
                item.date == pd.Timestamp("2025-02-05") and item.kind == "settlement"
            )
            if branch == "released_at_reopening_settlement" and after_reopen_close:
                holiday_active = False
            if holiday_active:
                floors["holiday_spec_floor"] = holiday["rates"][code]["spec_margin"]
            result = known_margin_floor(floors, complete=False)
            row = {
                "date": item.date,
                "phase": item.kind,
                "contract": item.contract,
                "branch": branch,
                "baseline_is_observed_reference_not_notice": True,
                "maturity_boundary": boundary,
                "holiday_active_in_scenario": holiday_active,
                "snapshot_spec_long_margin_rate": item.spec_long_margin_rate,
                "snapshot_spec_short_margin_rate": item.spec_short_margin_rate,
                "reference_calibration_row": item.date == pd.Timestamp("2025-01-24")
                and item.kind == "trading",
                **result,
                "real_execution_eligible": False,
            }
            row["matches_observed_long_rate"] = bool(
                np.isclose(
                    result["known_component_floor"],
                    item.spec_long_margin_rate,
                    atol=1e-14,
                    rtol=1e-12,
                )
            )
            cases.append(row)
            components.extend(
                {
                    "date": item.date,
                    "phase": item.kind,
                    "contract": item.contract,
                    "branch": branch,
                    "component": name,
                    "absolute_floor": value,
                }
                for name, value in floors.items()
            )
    # The unreleased branch deliberately retains the holiday floor through the review end;
    # it is an illustrative alternative, not an exhaustive model of later recovery paths.
    scenarios = pd.DataFrame(cases)
    exclusions = pd.DataFrame(
        [
            {
                "source_id": "cu_rules_later_negative",
                "review_date": date,
                "date_applicable": date_applicable(rules["later_cu_negative"], date),
                "reason": "later_declared_effective_version_not_usable_for_earlier_case",
            }
            for date in plan["study"]["dates"]
        ]
    )
    released = scenarios.loc[scenarios.branch.eq("released_at_reopening_settlement")]
    summary = {
        "version": __version__,
        "kind": plan["study"]["kind"],
        "new_rule_http_requests": manifest["new_requests"],
        "historical_directory_pages_read": 3,
        "new_rule_sources": len(versions),
        "reused_seed_sources": 9,
        "parameter_rows": len(parameters),
        "scenario_rows": len(scenarios),
        "released_scenario_numeric_agreements": int(released.matches_observed_long_rate.sum()),
        "released_scenario_rows": len(released),
        "observed_reference_calibration_rows_in_each_scenario": int(
            released.reference_calibration_row.sum()
        ),
        "later_version_dates_excluded": int((~exclusions.date_applicable).sum()),
        "scheduled_submission_cutoff_at": auction["scheduled_submission_cutoff_at"],
        "fully_resolved_margin_rates": int(scenarios.fully_resolved_rate.notna().sum()),
        "historical_first_vintage_certified": False,
        "complete_notice_inventory_certified": False,
        "actual_opening_clock_certified": False,
        "real_execution_eligible": False,
        "prediction_tests_added": 0,
        "registered_exploration_family_size_unchanged": len(
            pd.read_csv(paths["prior_exploration"])
        ),
        "real_investment_backtests": 0,
        "protocol_sha256": digest(config),
        "capture_manifest_sha256": digest(manifest_path),
        "inputs": {
            name: {"path": str(paths[name]), "sha256": digest(paths[name])}
            for name in plan["input_hashes"]
        },
        "code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("execution_rule_review.py", "execution_sources.py", "execution.py")
        },
    }
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in {
        "source_versions": pd.DataFrame(versions),
        "seed_parameters": parameters,
        "margin_scenarios": scenarios,
        "margin_components": pd.DataFrame(components),
        "version_exclusions": exclusions,
        "scheduled_auction": pd.DataFrame([auction]),
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    display = released[
        [
            "date",
            "phase",
            "contract",
            "known_component_floor",
            "snapshot_spec_long_margin_rate",
            "binding_components",
        ]
    ]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>旧规则与集合竞价 · v{__version__}</title><style>body{{max-width:1200px;margin:35px auto;padding:24px;font:16px/1.85 system-ui,'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#193348}}section{{background:white;border:1px solid #dbe5eb;border-radius:16px;padding:24px;margin:24px 0}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:9px;border-bottom:1px solid #dbe5eb}}.scroll{{overflow:auto}}a{{color:#126b83}}</style><p>Commodity Factor Lab · v{__version__}</p><h1>铜2503为何没有恢复到9%？</h1><section><h2>找到当时日期适用的规则组件</h2><p>2024年规则版本自10月23日实施，铜交割前一月的最低比例为10%；风险办法规定多个适用比例取最高值。原日历中2025年2月首个交易日为2月5日，所以cu2503进入10%阶段，cu2504尚未进入。在释放春节附加比例的假设分支下，前者max(5%,9%,10%)=10%，后者max(5%,9%)=9%，均与保存快照一致。</p><p>9%是所选节前交易快照的观察参照，不是本轮已认证的普通月份通知。24个规则/参数比较中含3个参照行，不能当成24个独立确认；本轮保留两个恢复情景，不用事后观察值决定当时条件已满足。未释放分支在本轮区间持续保留春节比例，仅作对照，并未穷举后续所有恢复路径。</p><div class="scroll">{table(display)}</div></section><section><h2>相同数字不代表版本适用</h2><p>另取2025年8月铜规则作负对照，其10%条款数值相同，但声明生效日在案例之后，四个日期均排除。当前页面版本仍不能单独认证历史初发；已知组件的最大值只记为情景下限，精确已解析保证金比例保持缺失，不直接传给真实交易账。</p>{table(exclusions)}</section><section><h2>开盘价与申报截止</h2><p>2024年交易管理办法规定集合竞价前4分钟申报、后1分钟撮合；未产生竞价成交价时，日线开盘价可取后续第一笔成交。结合2025年春节通知的08:55—09:00复市集合竞价，可推导计划申报截止08:59。它不是已认证实际成交时刻，也不能用09:00之前任意时刻的决策保证获得开盘价。</p>{table(pd.DataFrame([auction]))}</section><section><h2>范围与后续</h2><p>这是历史规则解释、组件情景和时序校验，不新增预测检验或真实投资收益。七份新规则/目录原文、九份旧来源复用，全部响应及哈希保留。十五项探索保持。完整通知覆盖、历史版本、实际时段、经纪收费及成交资格仍分别待证。</p><p><a href="margin_scenarios.csv">两个恢复情景</a> · <a href="margin_components.csv">逐项绝对比例</a> · <a href="source_versions.csv">来源版本</a> · <a href="rules.json">产品规则</a> · <a href="scheduled_auction.csv">计划竞价时序</a> · <a href="version_exclusions.csv">新版负对照</a> · <a href="run.json">输入与代码指纹</a></p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
