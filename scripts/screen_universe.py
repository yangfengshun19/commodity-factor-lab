"""Reproduce the frozen availability screen without reading evaluation returns."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from commodity_factor_lab.contract_research import verify_receipt
from commodity_factor_lab.curve_factors import CurveRules, contract_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/validation_plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Screen output directory must be new")
    inputs = [verify_receipt(p) for p in (args.bars, args.metadata)]
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bars = pd.read_csv(args.bars, parse_dates=["date"])
    bars = bars.loc[bars.date.le("2022-12-31")].copy()
    dates = sorted(bars.date.unique())
    calendar = pd.DataFrame(
        {"date": dates, "signal_cutoff": [f"{d:%Y-%m-%d}T20:30:00+08:00" for d in dates]}
    )
    curves = contract_curves(
        bars, pd.read_csv(args.metadata), calendar, CurveRules(min_days=60), retrospective=True
    )
    curves["common_features"] = (
        curves[["carry", "momentum_126d", "basis_momentum"]].notna().all(axis=1)
    )
    rows = []
    for code in plan["expansion_candidates"]:
        product = bars.loc[bars.code.eq(code)]
        history = curves.loc[curves.code.eq(code) & curves.date.between("2022-01-01", "2022-12-31")]
        fraction = history.common_features.mean() if len(history) else 0.0
        first = product.date.min()
        retain = pd.notna(first) and first <= pd.Timestamp("2020-01-02") and fraction >= 0.8
        rows.append(
            {
                "code": code,
                "first_observed": first,
                "pre_2023_rows": len(product),
                "screening_sessions": len(history),
                "common_feature_fraction_2022": fraction,
                "retained": retain,
                "rule": "listed_by_sample_start_and_80pct_2022_feature_coverage",
            }
        )
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True)
    result.to_csv(args.output / "screen.csv", index=False)
    retained = result.loc[result.retained, "code"].tolist()
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "retained_codes": retained,
        "inputs": inputs,
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule_note": "listed in the legacy rule label means first observed; IPO dates are not certified here",
        "uses_future_labels_or_performance": False,
        "latest_observation_used": "2022-12-31",
        "claim_eligible": False,
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Coverage-only screen retained {len(retained)} products: {', '.join(retained)}")


if __name__ == "__main__":
    main()
