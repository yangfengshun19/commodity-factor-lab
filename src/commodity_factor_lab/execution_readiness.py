"""Source-supported rule components, never an automatic historical execution certificate."""

import json
import re
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .contract_report import table
from .execution_sources import execution_plan
from .nbs_ingest import digest

GATES = (
    "complete_notice_inventory",
    "historical_first_vintage",
    "broker_fee_schedule",
    "account_margin_schedule",
    "dated_session_clock",
    "directional_fill_evidence",
    "contract_multiplier_expiry",
)


class TextReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def source_text(content: bytes) -> str:
    reader = TextReader()
    reader.feed(content.decode("utf-8-sig"))
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", "".join(reader.parts)))


def _match(pattern, text):
    matched = re.search(pattern, text)
    if matched is None:
        raise ValueError("Official notice no longer matches the registered rule extraction")
    return matched


def parse_notice(content: bytes, identity: str) -> dict:
    text = source_text(content)
    publication = _match(r"发布日期[:：](\d{4}-\d{2}-\d{2})", text).group(1)
    if identity in {"rb2410_april", "rb_offcycle_may"}:
        effect = _match(r"自(\d{4})年(\d+)月(\d+)日交易\(即(\d+)月(\d+)日晚夜盘\)起", text)
        year, month, day, night_month, night_day = map(int, effect.groups())
        record = {
            "id": identity,
            "source_declared_date": publication,
            "effective_trading_date": f"{year:04}-{month:02}-{day:02}",
            "effective_night_natural_date": f"{year - (night_month > month):04}-{night_month:02}-{night_day:02}",
            "declared_clock_precision": "date_only",
            "first_vintage_certified": False,
        }
        if identity == "rb2410_april":
            _match(
                r"RB2410合约交易手续费调整为成交金额的万分之一,日内平今仓交易手续费调整为成交金额的万分之一",
                text.replace("，", ","),
            )
            if publication != "2024-03-28" or record["effective_trading_date"] != "2024-04-01":
                raise ValueError("Unexpected dated single-contract notice")
            record.update(
                contract="rb2410",
                general_fee_rate=float(Decimal("1") / 10000),
                close_today_fee_rate=float(Decimal("1") / 10000),
            )
        else:
            clause = _match(
                r"螺纹钢期货除1月、5月、10月合约以外其他月合约的交易手续费调整为成交金额的万分之([^,，。]+)",
                text,
            )
            amount = {"零点二": Decimal("0.2")}.get(clause.group(1))
            if (
                amount is None
                or publication != "2024-05-29"
                or record["effective_trading_date"] != "2024-05-31"
            ):
                raise ValueError("Unexpected off-cycle fee notice")
            _match(r"日内平今仓交易手续费调整为成交金额的万分之零点二", text)
            _match(
                r"进入交割月前第二月的第一个交易日起,交易手续费恢复到成交金额的万分之一",
                text.replace("，", ","),
            )
            record.update(
                excluded_delivery_months=[1, 5, 10],
                reduced_fee_rate=float(amount / 10000),
                restored_fee_rate=float(Decimal("1") / 10000),
                restore_months_before_delivery=2,
            )
        return record
    if identity != "cny2025" or publication != "2025-01-22":
        raise ValueError("Unregistered notice identity or date")
    _match(r"自2025年1月24日.*?收盘结算时起", text)
    rates = {}
    for code, names in (("cu", "铜、铝、锌、铅"), ("rb", "螺纹钢、热轧卷板、不锈钢")):
        values = _match(
            re.escape(names)
            + r"期货合约的涨跌停板幅度调整为(\d+)%,套保交易保证金比例调整为(\d+)%,投机交易保证金比例调整为(\d+)%",
            text.replace("，", ","),
        )
        rates[code] = {
            "price_limit": int(values[1]) / 100,
            "hedge_margin": int(values[2]) / 100,
            "spec_margin": int(values[3]) / 100,
        }
    for fragment in (
        "2025年1月27日",
        "晚上不进行夜盘交易",
        "2025年2月5日",
        "08:55-09:00",
        "当晚恢复夜盘交易",
        "自第一个未出现单边市的交易日收盘结算时",
        "恢复至原有水平",
    ):
        if fragment not in text:
            raise ValueError(
                "Holiday date, night-session or conditional restoration evidence missing"
            )
    return {
        "id": identity,
        "source_declared_date": publication,
        "effective_trading_date": "2025-01-24",
        "effective_phase": "settlement",
        "rates": rates,
        "restoration_not_before": "2025-02-05",
        "restoration_conditional": True,
        "declared_clock_precision": "date_only",
        "first_vintage_certified": False,
    }


def fee_candidate(date: str, contract: str, notices: dict, calendar: pd.DataFrame) -> dict:
    matched = re.fullmatch(r"rb(\d{2})(\d{2})", contract)
    if matched is None or not 1 <= int(matched[2]) <= 12:
        raise ValueError("Fee assay requires a registered RB futures contract")
    stamp = pd.Timestamp(date)
    record = {
        "date": stamp,
        "contract": contract,
        "notice_expected_rate": np.nan,
        "reason": "no_registered_rule_yet",
        "maturity_reset_date": pd.NaT,
        "historical_execution_eligible": False,
    }
    april, may = notices["rb2410_april"], notices["rb_offcycle_may"]
    if contract == april["contract"] and stamp >= pd.Timestamp(april["effective_trading_date"]):
        record.update(
            notice_expected_rate=april["general_fee_rate"],
            reason="dated_single_contract_rule",
            source_id=april["id"],
        )
    year, month = 2000 + int(matched[1]), int(matched[2])
    if (
        stamp >= pd.Timestamp(may["effective_trading_date"])
        and month not in may["excluded_delivery_months"]
    ):
        boundary = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(
            months=may["restore_months_before_delivery"]
        )
        days = pd.to_datetime(calendar.date)
        in_month = days.loc[days.dt.to_period("M").eq(boundary.to_period("M"))]
        if in_month.empty or not days.is_unique or not days.is_monotonic_increasing:
            raise ValueError("Complete chronological reference calendar month is required")
        reset = in_month.min()
        record.update(
            notice_expected_rate=may["restored_fee_rate"]
            if stamp >= reset
            else may["reduced_fee_rate"],
            reason="maturity_window_restored" if stamp >= reset else "offcycle_reduced",
            maturity_reset_date=reset,
            source_id=may["id"],
        )
    return record


def parse_parameter_snapshot(
    content: bytes, kind: str, expected_date: str, contracts: list[str]
) -> tuple[list, dict]:
    data = json.loads(content.decode("utf-8-sig"))
    if str(data["report_date"]).strip() != expected_date:
        raise ValueError("Parameter body report date differs from the requested date")
    update = pd.Timestamp(
        datetime.strptime(data["update_date"], "%Y%m%d %H:%M:%S"), tz="Asia/Shanghai"
    )
    field = {"trading": "ContractDailyTradeArgument", "settlement": "Settlement"}[kind]
    selected, seen = [], set()
    for raw in data[field]:
        contract = raw["INSTRUMENTID"].strip().lower()
        if contract in seen:
            raise ValueError("Duplicate contract in parameter snapshot")
        seen.add(contract)
        if contract not in contracts:
            continue
        if str(raw["TRADINGDAY"]).strip() != expected_date:
            raise ValueError("Parameter row date differs from its report")
        row = {
            "date": pd.Timestamp(expected_date),
            "contract": contract,
            "kind": kind,
            "source_update_at": update,
            "spec_long_margin_rate": float(raw["SPEC_LONGMARGINRATIO"]),
            "spec_short_margin_rate": float(raw["SPEC_SHORTMARGINRATIO"]),
            "raw_trade_fee_unit": raw.get("TRADEFEEUNIT", ""),
            "raw_trade_fee_ratio": float(raw["TRADEFEERATION"])
            if "TRADEFEERATION" in raw
            else np.nan,
            "snapshot_is_first_vintage": False,
        }
        if not all(
            np.isfinite(row[name]) and 0 < row[name] <= 1
            for name in ("spec_long_margin_rate", "spec_short_margin_rate")
        ):
            raise ValueError("Invalid margin field")
        selected.append(row)
    if set(contracts) - {row["contract"] for row in selected}:
        raise ValueError("A registered contract is missing from the parameter snapshot")
    return selected, {
        "date": expected_date,
        "kind": kind,
        "source_update_at": update.isoformat(),
        "source_update_after_report_date": update.date() > pd.Timestamp(expected_date).date(),
        "historical_first_vintage_certified": False,
    }


def admission(evidence: dict) -> tuple[bool, str]:
    if set(evidence) != set(GATES) or any(
        type(value) not in (bool, np.bool_) for value in evidence.values()
    ):
        raise ValueError("Every execution evidence gate needs an explicit boolean")
    missing = [name for name in GATES if not evidence[name]]
    return not missing, ";".join(missing)


def run_execution_readiness(config: Path) -> Path:
    plan, paths = execution_plan(config)
    manifest_path = paths["raw"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != digest(config):
        raise ValueError("Execution captures use another protocol")
    captures = {item["id"]: item for item in manifest["records"]}
    expected = {item["id"]: item for item in plan["sources"]}
    if len(captures) != len(manifest["records"]) or set(captures) != set(expected):
        raise ValueError("Capture source identities differ from the registered assay")
    for identity, item in captures.items():
        if any(item.get(name) != value for name, value in expected[identity].items()):
            raise ValueError("Captured source metadata differs from the frozen plan")
        if Path(item.get("file", "")).name != item.get("file"):
            raise ValueError("Capture file must be a direct raw-directory file")
        if "seed_sha256" in item and item.get("sha256") != item["seed_sha256"]:
            raise ValueError("Reused response is not the frozen seed")
    if len(captures) != len(plan["sources"]) or any(
        item["status"] not in {"http_ok", "seed_reused"} for item in captures.values()
    ):
        raise ValueError(
            "Source capture is incomplete; preserve failures before drawing readiness conclusions"
        )
    notices, snapshots, sources = {}, [], []
    for item in captures.values():
        path = paths["raw"] / item["file"]
        if digest(path) != item["sha256"]:
            raise ValueError("Execution raw source differs from its receipt")
        source = {
            "id": item["id"],
            "url": item["url"],
            "sha256": item["sha256"],
            "kind": item["kind"],
            "first_vintage_certified": False,
        }
        if item["kind"] == "notice":
            notice = parse_notice(path.read_bytes(), item["id"])
            notices[item["id"]] = notice
            source["source_declared_date"] = notice["source_declared_date"]
        elif item["kind"] in {"trading", "settlement"}:
            contracts = (
                []
                if "seed_file" in item
                else (
                    plan["study"]["margin_contracts"]
                    if item["date"].startswith("2025")
                    else plan["study"]["fee_contracts"]
                )
            )
            rows, audit = parse_parameter_snapshot(
                path.read_bytes(), item["kind"], item["date"], contracts
            )
            snapshots.extend(
                {**row, "source_id": item["id"], "source_sha256": item["sha256"]} for row in rows
            )
            source.update(audit)
        else:
            if "21:00" not in source_text(path.read_bytes()):
                raise ValueError("Current-hours negative control no longer contains its clock")
            source["historical_clock_certified_from_current_page"] = False
        sources.append(source)
    snapshots = pd.DataFrame(snapshots)
    sources = pd.DataFrame(sources)
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    fees = pd.DataFrame(
        [
            fee_candidate(date, contract, notices, calendar)
            for date in plan["study"]["fee_dates"]
            for contract in plan["study"]["fee_contracts"]
        ]
    )
    fees = fees.merge(
        snapshots.loc[
            snapshots.kind.eq("settlement"),
            ["date", "contract", "raw_trade_fee_unit", "raw_trade_fee_ratio"],
        ],
        on=["date", "contract"],
        how="left",
        validate="one_to_one",
    )
    fees["snapshot_matches_candidate_numerically"] = [
        bool(np.isclose(row.notice_expected_rate, row.raw_trade_fee_ratio, atol=1e-15, rtol=1e-12))
        if np.isfinite([row.notice_expected_rate, row.raw_trade_fee_ratio]).all()
        else None
        for row in fees.itertuples()
    ]
    margins = snapshots.loc[snapshots.kind.eq("trading")].merge(
        snapshots.loc[snapshots.kind.eq("settlement")],
        on=["date", "contract"],
        suffixes=("_trading", "_settlement"),
        validate="one_to_one",
    )
    margins["observed_phase_rate_diff"] = ~np.isclose(
        margins.spec_long_margin_rate_trading,
        margins.spec_long_margin_rate_settlement,
        atol=1e-15,
        rtol=1e-12,
    )
    margins["automatic_holiday_restoration_performed"] = False
    sessions = pd.DataFrame(
        [
            {
                "source_id": "rb2410_april",
                "trading_date": "2024-04-01",
                "night_natural_date": notices["rb2410_april"]["effective_night_natural_date"],
                "schedule_component": "dated_fee_effective_night",
                "actual_open_at": None,
            },
            {
                "source_id": "rb_offcycle_may",
                "trading_date": "2024-05-31",
                "night_natural_date": notices["rb_offcycle_may"]["effective_night_natural_date"],
                "schedule_component": "dated_fee_effective_night",
                "actual_open_at": None,
            },
            {
                "source_id": "cny2025",
                "trading_date": None,
                "night_natural_date": "2025-01-27",
                "schedule_component": "night_cancelled",
                "actual_open_at": None,
            },
            {
                "source_id": "cny2025",
                "trading_date": "2025-02-05",
                "night_natural_date": None,
                "schedule_component": "day_reopen_auction_08:55_09:00",
                "actual_open_at": None,
            },
            {
                "source_id": "cny2025",
                "trading_date": "2025-02-06",
                "night_natural_date": "2025-02-05",
                "schedule_component": "night_resumes_next_calendar_session",
                "actual_open_at": None,
            },
        ]
    )
    sessions["historical_actual_opening_certified"] = False
    eligible, blockers = admission(dict.fromkeys(GATES, False))
    gates = pd.DataFrame(
        [
            {
                "scope": "fixed_execution_assay",
                "gate": gate,
                "satisfied": False,
                "real_execution_eligible": eligible,
                "blockers": blockers,
            }
            for gate in GATES
        ]
    )
    summary = {
        "version": __version__,
        "kind": plan["study"]["kind"],
        "sources": len(sources),
        "new_http_requests": manifest["new_requests"],
        "reused_seed_responses": 2,
        "normalized_parameter_rows": len(snapshots),
        "fee_cases": len(fees),
        "fee_cases_with_a_numeric_snapshot_comparison": int(
            fees.snapshot_matches_candidate_numerically.notna().sum()
        ),
        "fee_snapshot_matches": int(fees.snapshot_matches_candidate_numerically.eq(True).sum()),
        "margin_phase_comparisons": len(margins),
        "different_trading_and_settlement_rate_cases": int(margins.observed_phase_rate_diff.sum()),
        "source_update_after_report_date": int(
            sources.source_update_after_report_date.eq(True).sum()
        ),
        "source_supported_session_components": len(sessions),
        "certified_actual_session_openings": 0,
        "real_execution_eligible": eligible,
        "prediction_tests_added": 0,
        "registered_exploration_family_size_unchanged": len(
            pd.read_csv(paths["prior_exploration"])
        ),
        "historical_first_vintage_certified": False,
        "real_investment_backtests": 0,
        "protocol_sha256": digest(config),
        "capture_manifest_sha256": digest(manifest_path),
        "inputs": {
            name: {"path": str(paths[name]), "sha256": digest(paths[name])}
            for name in plan["input_hashes"]
        },
        "code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("execution_readiness.py", "execution_sources.py", "execution.py")
        },
    }
    if summary["registered_exploration_family_size_unchanged"] != 15:
        raise ValueError("Do not lose the existing exploration registry")
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in {
        "fee_cases": fees,
        "margin_phases": margins,
        "snapshot_parameters": snapshots,
        "source_audit": sources,
        "session_evidence": sessions,
        "admission_gates": gates,
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "notice_rules.json").write_text(
        json.dumps(notices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    display = margins[
        [
            "date",
            "contract",
            "spec_long_margin_rate_trading",
            "spec_long_margin_rate_settlement",
            "observed_phase_rate_diff",
        ]
    ]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实执行来源准入 · v{__version__}</title><style>body{{max-width:1200px;margin:35px auto;padding:24px;font:16px/1.85 system-ui,'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#193348}}section{{background:white;border:1px solid #dbe5eb;border-radius:16px;padding:24px;margin:24px 0}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:9px;border-bottom:1px solid #dbe5eb}}.scroll{{overflow:auto}}a{{color:#126b83}}</style><p>Commodity Factor Lab · v{__version__}</p><h1>手续费、保证金和交易时段：规则不能混用</h1>
<section><h2>已取得可核算的小样本证据</h2><p>{len(sources)}份来源，新增14次公开请求、复用2份旧响应。三个有日期的通知、现行交易时间负对照及十二份参数快照。解析{len(snapshots)}条固定合约参数；手续费9案例中6个可与快照数值比较，{summary["fee_snapshot_matches"]}个一致。8个交易/结算保证金比较中，{summary["different_trading_and_settlement_rate_cases"]}个比例不同。</p></section>
<section><h2>手续费按合约、到期月份和交易日生效</h2><div class="scroll">{table(fees)}</div><p>螺纹非1/5/10交割月的减费在2024年5月31日交易起生效，即5月30日晚夜盘；进入交割月前第二月首个交易日恢复。规则推导候选与快照字段相同仍不能认证全部初发版本或经纪商实际收费。尚未生效的案例保持未知，不借未来规则补回。</p></section>
<section><h2>同一天的交易参数和结算参数不能当成一份</h2><div class="scroll">{table(display)}</div><p>节前保证金从收盘结算起调整；节后恢复附带未出现单边市等条件。铜2503在2月5日结算快照为10%，并非简单倒回1月24日交易快照的9%，不能无条件复制原比例。不同合约的临近交割规则及其他调整仍需分别取证。</p><p>资金账已支持独立的开盘与结算保证金比例，以及结算参数可用时钟。旧单比例输入仍可运行；没有把结算才生效的比例倒用到开盘，也不将保证金占用记为交易亏损。</p></section>
<section><h2>时段与版本仍有门槛</h2><div class="scroll">{table(sessions)}</div><p>有日期的通知确认了费率生效夜盘的自然日、春节停盘及复市集合竞价。现行页面的21:00不能自动认证过去所有实际开盘；本轮未填入真实执行open_at或成交方向标志。两份旧参数的来源更新时间晚于经济日期，仍保留负对照；其余同日时钟也不等于历史初发证明。</p><p>真实执行资格仍为false：完整通知链、初发版本、经纪费用和账户保证金、历史时段、成交及合约参数需要分别通过。没有新增预测检验、投资回测或收益；十五项探索保持。</p><p><a href="source_audit.csv">来源与时钟</a> · <a href="notice_rules.json">公告规则</a> · <a href="snapshot_parameters.csv">快照参数</a> · <a href="margin_phases.csv">保证金两阶段</a> · <a href="session_evidence.csv">时段证据</a> · <a href="admission_gates.csv">准入门槛</a> · <a href="run.json">输入及源码指纹</a></p><p>协议：{escape(config.name)}。历史初发认证、可成交性与真实投资业绩仍未完成。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
