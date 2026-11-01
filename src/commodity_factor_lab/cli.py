"""Command line entry points. Importing this module performs no I/O."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from .catalog import FAMILIES, code_from_filename
from .config import load_config
from .data import read_csv
from .demo import generate_demo
from .factors import price_factors
from .pipeline import run
from .protocol import load_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Commodity Factor Lab / 可复现的期货因子研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("factors", help="List maintained factor definitions, roles and hypotheses")
    factor_audit = sub.add_parser(
        "factor-audit", help="Inventory all 26 legacy definitions, sources and decisions"
    )
    factor_audit.add_argument("--data", type=Path, default=Path("data/raw"))
    factor_audit.add_argument(
        "--registry", type=Path, default=Path("configs/factor_decisions.toml")
    )
    factor_audit.add_argument("--output", type=Path, required=True)
    extension = sub.add_parser(
        "factor-extension",
        help="Run the registered extension and keep source-gated hypotheses visible",
    )
    extension.add_argument("--config", type=Path, default=Path("configs/factor_extension.toml"))
    position_fetch = sub.add_parser(
        "fetch-positions", help="Resume the frozen public broker-ranking schedule"
    )
    position_fetch.add_argument("--config", type=Path, default=Path("configs/position_study.toml"))
    position_fetch.add_argument("--limit", type=int)
    position_study = sub.add_parser(
        "position-research", help="Qualify same-contract broker ranks and run fixed diagnostics"
    )
    position_study.add_argument("--config", type=Path, default=Path("configs/position_study.toml"))
    availability = sub.add_parser(
        "position-availability",
        help="Compare frozen historical and current-signal information cutoffs",
    )
    availability.add_argument(
        "--config", type=Path, default=Path("configs/position_vintage_review.toml")
    )
    spot = sub.add_parser(
        "spot-readiness", help="Replay the fixed physical-spot source assay offline"
    )
    spot.add_argument("--config", type=Path, default=Path("configs/spot_readiness.toml"))
    spot.add_argument("--sources", type=Path, default=Path("configs/spot_sources.toml"))
    for command, help_text in (
        ("fetch-nbs-probe", "Capture the fixed public NBS industrial reference source assay"),
        (
            "nbs-readiness",
            "Replay NBS periods, specifications and publication event alignment offline",
        ),
    ):
        child = sub.add_parser(command, help=help_text)
        child.add_argument("--config", type=Path, default=Path("configs/nbs_readiness.toml"))
    for command, help_text in (
        ("nbs-history-discover", "Discover and freeze the 2020-2025 official period catalog"),
        ("fetch-nbs-history", "Resume source capture from the frozen official catalog"),
        ("nbs-history", "Build and audit the completed source archive offline"),
    ):
        child = sub.add_parser(command, help=help_text)
        child.add_argument("--config", type=Path, default=Path("configs/nbs_history.toml"))
        child.add_argument("--rules", type=Path, default=Path("configs/nbs_history_rules.toml"))
    prediction = sub.add_parser(
        "nbs-prediction", help="Run the fixed descriptive industrial-reference forecast comparison"
    )
    prediction.add_argument("--config", type=Path, default=Path("configs/nbs_prediction.toml"))
    market = sub.add_parser(
        "market-benchmark",
        help="Run the frozen market simulation with explicit execution assumptions",
    )
    market.add_argument("--config", type=Path, default=Path("configs/market_benchmark.toml"))
    attribution = sub.add_parser(
        "market-attribution", help="Explain saved market books without creating a new strategy"
    )
    attribution.add_argument("--config", type=Path, default=Path("configs/market_attribution.toml"))
    capacity = sub.add_parser(
        "capacity-study", help="Replay fixed targets with past-volume execution budgets"
    )
    capacity.add_argument("--config", type=Path, default=Path("configs/capacity_study.toml"))
    capacity.add_argument(
        "--freeze", action="store_true", help="Create an immutable scope receipt before running"
    )
    evidence = sub.add_parser("factor-evidence", help="Common-sample effect sizes and uncertainty")
    evidence.add_argument("--config", type=Path, default=Path("configs/factor_evidence.toml"))
    evidence.add_argument(
        "--freeze", action="store_true", help="Record immutable scope before analysis"
    )
    for command, help_text in (
        ("fetch-execution-probe", "Capture the fixed official execution-source assay"),
        ("execution-readiness", "Audit dated fees, margin phases and session evidence offline"),
    ):
        child = sub.add_parser(command, help=help_text)
        child.add_argument("--config", type=Path, default=Path("configs/execution_readiness.toml"))
    for command, help_text in (
        ("fetch-execution-rules", "Capture the fixed historical execution rule set"),
        (
            "execution-rule-review",
            "Trace margin floors, version dates and scheduled auction phases",
        ),
    ):
        child = sub.add_parser(command, help=help_text)
        child.add_argument(
            "--config", type=Path, default=Path("configs/execution_rule_review.toml")
        )
    reconcile = sub.add_parser(
        "reconcile-contracts", help="Independent arithmetic and raw-source audit"
    )
    for name in ("baseline", "plan", "daily-directory", "output"):
        reconcile.add_argument(f"--{name}", type=Path, required=True)
    for command in ("demo", "audit", "run"):
        child = sub.add_parser(command)
        child.add_argument(
            "--config",
            type=Path,
            default=Path("configs/demo.toml" if command == "demo" else "configs/research.toml"),
        )
        child.add_argument(
            "--output", type=Path, help="Override output parent; existing runs are retained"
        )
    build = sub.add_parser(
        "build-price", help="Build trend, value and volatility factors from local OHLC CSVs"
    )
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    for command in ("study", "study-demo"):
        child = sub.add_parser(command, help="Controlled common-sample retrospective study")
        child.add_argument(
            "--config",
            type=Path,
            default=Path(
                "configs/study_demo.toml" if command == "study-demo" else "configs/study.toml"
            ),
        )
        child.add_argument("--output", type=Path)
    contract = sub.add_parser(
        "contract-panel",
        help="Build lag-selected same-contract returns from explicit contract data",
    )
    contract.add_argument("--bars", type=Path, required=True)
    contract.add_argument("--calendar", type=Path, required=True)
    contract.add_argument("--output", type=Path, required=True)
    contract.add_argument("--roll-days", type=int, default=20)
    fetch = sub.add_parser(
        "fetch-sina", help="Read public continuous prices into a new timestamped snapshot"
    )
    fetch.add_argument("--codes", nargs="+", required=True)
    fetch.add_argument("--through", required=True, help="Last completed trading date to retain")
    fetch.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser(
        "contract-research",
        help="Official-contract characteristic diagnostics, not investment performance",
    )
    pilot.add_argument("--config", type=Path, default=Path("configs/curve_pilot.toml"))
    pilot.add_argument(
        "--without-warrants", action="store_true", help="Explicitly record inventory as untestable"
    )
    annual = sub.add_parser("fetch-shfe", help="Read official completed yearly contract archives")
    annual.add_argument("--years", type=int, nargs="+", required=True)
    annual.add_argument("--output", type=Path, required=True)
    ingest = sub.add_parser("ingest-shfe", help="Normalize saved official annual ZIP archives")
    ingest.add_argument("--archives", type=Path, nargs="+", required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--codes", nargs="+", default=["cu", "al", "zn", "rb", "hc", "ru"])
    reparse = sub.add_parser(
        "reparse-shfe", help="Reparse saved contract metadata without redownloading"
    )
    reparse.add_argument("--source", type=Path, required=True)
    reparse.add_argument("--output", type=Path, required=True)
    metadata = sub.add_parser(
        "fetch-shfe-metadata", help="Read dated official contract specifications"
    )
    metadata.add_argument("--dates", nargs="+", required=True)
    metadata.add_argument("--output", type=Path, required=True)
    warehouse = sub.add_parser("fetch-warrants", help="Read predeclared weekly warehouse dates")
    warehouse.add_argument("--dates", nargs="+", required=True)
    warehouse.add_argument("--output", type=Path, required=True)
    warehouse.add_argument("--codes", nargs="+", default=["cu", "al", "zn", "rb", "hc", "ru"])
    warehouse_reparse = sub.add_parser(
        "reparse-warrants", help="Reparse immutable official warehouse responses locally"
    )
    warehouse_reparse.add_argument("--source", type=Path, required=True)
    warehouse_reparse.add_argument("--output", type=Path, required=True)
    warehouse_reparse.add_argument(
        "--codes", nargs="+", default=["cu", "al", "zn", "rb", "hc", "ru"]
    )
    ledger = sub.add_parser(
        "execution-ledger", help="Strict integer-contract daily-settlement cash ledger"
    )
    for name in ("bars", "calendar", "orders", "output"):
        ledger.add_argument(f"--{name}", type=Path, required=True)
    ledger.add_argument("--initial-cash", type=float, default=1_000_000)
    ledger.add_argument("--max-margin-fraction", type=float, default=0.5)
    contract_demo = sub.add_parser(
        "contract-demo", help="Offline synthetic contract, factor and settlement demonstration"
    )
    contract_demo.add_argument("--output", type=Path, default=Path("reports/runs/contract_demo"))
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-execution-rules":
            from .execution_rule_review import capture_rule_review

            capture_rule_review(args.config)
            return 0
        if args.command == "execution-rule-review":
            from .execution_rule_review import run_rule_review

            print(f"Historical execution rule review: {run_rule_review(args.config)}")
            return 0
        if args.command == "fetch-execution-probe":
            from .execution_sources import capture_execution_sources

            capture_execution_sources(args.config)
            return 0
        if args.command == "execution-readiness":
            from .execution_readiness import run_execution_readiness

            print(f"Execution source assay: {run_execution_readiness(args.config)}")
            return 0
        if args.command == "nbs-prediction":
            from .nbs_prediction import run_nbs_prediction

            print(f"Industrial reference forecast diagnostic: {run_nbs_prediction(args.config)}")
            return 0
        if args.command == "factor-evidence":
            from .factor_evidence import freeze_evidence, run_evidence

            result = freeze_evidence(args.config) if args.freeze else run_evidence(args.config)
            print(f"Factor evidence artifact: {result}")
            return 0
        if args.command == "capacity-study":
            from .capacity_study import freeze_capacity, run_capacity_study

            result = (
                freeze_capacity(args.config) if args.freeze else run_capacity_study(args.config)
            )
            print(f"Capacity artifact: {result}")
            return 0
        if args.command == "market-benchmark":
            from .market_benchmark import run_market_benchmark

            print(f"Assumed market benchmark: {run_market_benchmark(args.config)}")
            return 0
        if args.command == "market-attribution":
            from .market_attribution import run_market_attribution

            print(f"Saved market accounting attribution: {run_market_attribution(args.config)}")
            return 0
        if args.command == "nbs-history-discover":
            from .nbs_search import discover_history

            catalog = discover_history(args.config, args.rules)
            print(f"Frozen official catalog: {len(catalog['releases'])} articles")
            return 0
        if args.command == "fetch-nbs-history":
            from .nbs_archive import fetch_history_articles

            fetch_history_articles(args.config, args.rules)
            return 0
        if args.command == "nbs-history":
            from .nbs_history import run_nbs_history

            print(
                f"Official industrial reference history: {run_nbs_history(args.config, args.rules)}"
            )
            return 0
        if args.command == "fetch-nbs-probe":
            from .nbs_ingest import capture_nbs

            capture_nbs(args.config)
            return 0
        if args.command == "nbs-readiness":
            from .nbs_readiness import run_nbs_readiness

            print(f"Industrial reference source assay: {run_nbs_readiness(args.config)}")
            return 0
        if args.command == "spot-readiness":
            from .spot_readiness import run_spot_readiness

            print(f"Physical spot source assay: {run_spot_readiness(args.config, args.sources)}")
            return 0
        if args.command == "position-availability":
            from .position_availability import run_availability_review

            print(f"Information-set replay: {run_availability_review(args.config) / 'index.html'}")
            return 0
        if args.command == "fetch-positions":
            from .position_ingest import fetch_positions

            print(fetch_positions(args.config, args.limit))
            return 0
        if args.command == "position-research":
            from .position_study import run_position_study

            print(
                f"Same-contract position report: {run_position_study(args.config) / 'index.html'}"
            )
            return 0
        if args.command == "factor-audit":
            from .factor_audit import audit_factors

            print(
                f"Factor evidence report: {audit_factors(args.data, args.registry, args.output) / 'index.html'}"
            )
            return 0
        if args.command == "factor-extension":
            from .factor_extension import run_extension

            print(f"Extension report: {run_extension(args.config) / 'index.html'}")
            return 0
        if args.command == "reconcile-contracts":
            from .reconciliation import reconcile_sources

            print(
                f"Source reconciliation: {reconcile_sources(args.baseline, args.plan, args.daily_directory, args.output) / 'index.html'}"
            )
            return 0
        if args.command == "factors":
            from .factor_definitions import definitions

            print(json.dumps(definitions(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "contract-demo":
            from .contract_demo import run_contract_demo

            print(
                f"Synthetic contract report: {run_contract_demo(args.output.resolve()) / 'index.html'}"
            )
            return 0
        if args.command == "contract-research":
            from .contract_research import run_contract_research

            print(
                f"Contract characteristic report: {run_contract_research(args.config, args.without_warrants) / 'index.html'}"
            )
            return 0
        if args.command in ("fetch-shfe", "ingest-shfe", "reparse-shfe", "fetch-shfe-metadata"):
            from .official import fetch_annual, fetch_metadata, ingest_archives, reparse_metadata

            if args.command == "fetch-shfe":
                result = fetch_annual(args.years, args.output)
            elif args.command == "ingest-shfe":
                result = ingest_archives(args.archives, args.output, tuple(args.codes))
            elif args.command == "reparse-shfe":
                result = reparse_metadata(args.source, args.output)
            else:
                result = fetch_metadata(args.dates, args.output)
            print(f"Official source artifact: {result}")
            return 0
        if args.command in ("fetch-warrants", "reparse-warrants"):
            from .warrants import fetch_warrants, reparse_warrants

            result = (
                fetch_warrants(args.dates, args.output, tuple(args.codes))
                if args.command == "fetch-warrants"
                else reparse_warrants(args.source, args.output, tuple(args.codes))
            )
            print(f"Warehouse source snapshot: {result}")
            return 0
        if args.command == "execution-ledger":
            from .execution import settle_portfolio

            if args.output.exists():
                raise FileExistsError("Ledger output directory must be new")
            result = settle_portfolio(
                read_csv(args.bars),
                read_csv(args.calendar),
                read_csv(args.orders),
                args.initial_cash,
                args.max_margin_fraction,
            )
            args.output.mkdir(parents=True)
            for name in ("daily", "trades", "rejected", "holdings"):
                getattr(result, name).to_csv(args.output / f"{name}.csv", index=name == "daily")
            (args.output / "manifest.json").write_text(
                json.dumps(
                    {
                        "inputs": [
                            {
                                "file": str(p.resolve()),
                                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            }
                            for p in (args.bars, args.calendar, args.orders)
                        ],
                        "initial_cash": args.initial_cash,
                        "max_margin_fraction": args.max_margin_fraction,
                        "source_certification": "not performed by the ledger",
                        "claim_eligible": False,
                        "policy": "complete product targets, atomic failed rolls, next-open margin liquidation, terminal opening liquidation",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Settlement ledger: {args.output}")
            return 0
        if args.command == "fetch-sina":
            from .snapshot import fetch_snapshot

            print(
                f"Saved provider snapshot: {fetch_snapshot(args.output, args.codes, args.through)}"
            )
            return 0
        if args.command in ("study", "study-demo"):
            from .study import run_study

            config = load_protocol(args.config)
            if args.output:
                config = replace(config, output=args.output.resolve())
            if args.command == "study-demo":
                generate_demo(config.data)
            directory = run_study(config)
            print(f"Controlled retrospective report: {directory / 'index.html'}")
            return 0
        if args.command == "contract-panel":
            from .contracts import same_contract_panel

            output = args.output.resolve()
            if output.exists():
                raise FileExistsError("Contract output directory must be new")
            panel = same_contract_panel(
                read_csv(args.bars), read_csv(args.calendar), args.roll_days
            )
            output.mkdir(parents=True)
            panel.to_csv(output / "contract_returns.csv", index=False)
            metadata = {
                "roll_days": args.roll_days,
                "selection": "previous session known open interest, real expiry",
                "inputs": [
                    {"file": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (args.bars, args.calendar)
                ],
                "limitations": [
                    "not a margin simulator",
                    "roll trade legs recorded but costs not deducted",
                    "timestamps and source authenticity require external verification",
                ],
            }
            (output / "manifest.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Same-contract return panel: {output}")
            return 0
        if args.command == "build-price":
            source, target = args.input.resolve(), args.output.resolve()
            if target == source or target.is_relative_to(source) or source.is_relative_to(target):
                raise ValueError("Input and output must be separate directories")
            files = sorted(source.glob("*.csv"))
            if not files:
                raise ValueError("No OHLC CSV files found")
            for path in files:
                code_from_filename(path.name)
                for family in ("trend", "value", "volatility"):
                    destination = target / family / path.name
                    if destination.exists():
                        raise FileExistsError(
                            f"Refusing to overwrite existing features: {destination}"
                        )
            for path in files:
                bars = price_factors(read_csv(path))
                for family in ("trend", "value", "volatility"):
                    directory = target / family
                    directory.mkdir(parents=True, exist_ok=True)
                    destination = directory / path.name
                    if destination.exists():
                        raise FileExistsError(
                            f"Refusing to overwrite existing features: {destination}"
                        )
                    bars[
                        ["date", "open", "high", "low", "close", *FAMILIES[family].factors]
                    ].to_csv(destination, index=False)
            print(f"Built price factors: {target}")
            return 0
        config = load_config(args.config)
        if args.output:
            config = replace(config, output=args.output.resolve())
        if args.command == "demo":
            generate_demo(config.data)
        directory = run(config, audit_only=args.command == "audit")
        print(f"Run saved: {directory}")
        if args.command != "audit":
            print(f"Open report: {directory / 'index.html'}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
