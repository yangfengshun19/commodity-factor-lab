"""A run produces an immutable folder containing inputs, parameters and outputs."""

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .catalog import FAMILIES
from .config import ResearchConfig
from .data import load_family
from .engine import evaluate_factor, summarize
from .report import plot_overview, write_report


def run(config: ResearchConfig, audit_only: bool = False) -> Path:
    if not config.data.is_dir():
        raise ValueError(
            f"Data directory does not exist: {config.data}. Run the demo or import local snapshots."
        )
    synthetic = (config.data / "SYNTHETIC_DATA.json").exists()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = config.output / stamp
    directory.mkdir(parents=True, exist_ok=False)
    quality, inputs, summaries, ledgers = [], [], [], {}
    for family in config.families:
        loaded = load_family(config, family)
        quality.extend(loaded.quality)
        inputs.extend(loaded.inputs)
        panel = loaded.panel
        print(f"{family}: {len(panel):,} accepted rows", flush=True)
        if audit_only or panel.empty:
            continue
        sectors = [("all_commodities", panel.loc[panel["sector"] != "financial"])]
        sectors.extend(panel.groupby("sector", sort=True))
        for sector, subset in sectors:
            if subset.empty:
                continue
            for factor in FAMILIES[family].factors:
                result = evaluate_factor(subset, factor, config)
                for record in summarize(result, config):
                    summaries.append(
                        {
                            "family": family,
                            "sector": sector,
                            "factor": factor,
                            "direction": config.directions.get(factor, 1),
                            "claim_eligible": False,
                            "evidence_status": "synthetic_demo"
                            if synthetic
                            else "legacy_snapshot_diagnostic",
                            **record,
                        }
                    )
                key = (family, sector, factor)
                stem = "__".join(key)
                (directory / "ic").mkdir(exist_ok=True)
                result.ic.to_csv(directory / "ic" / f"{stem}.csv", index_label="date")
                if not result.ledger.empty:
                    (directory / "daily").mkdir(exist_ok=True)
                    result.ledger.to_csv(directory / "daily" / f"{stem}.csv", index_label="date")
                    ledgers[key] = result.ledger
                    (directory / "weights").mkdir(exist_ok=True)
                    result.weights.reindex(result.ledger.index).to_csv(
                        directory / "weights" / f"{stem}.csv",
                        index_label="date",
                        float_format="%.10g",
                    )
                (directory / "layers").mkdir(exist_ok=True)
                result.layers.to_csv(
                    directory / "layers" / f"{stem}.csv", index_label="signal_date"
                )
    quality_df = pd.DataFrame(quality)
    quality_df.to_csv(directory / "quality.csv", index=False, encoding="utf-8-sig")
    snapshot = asdict(config)
    snapshot["data"] = str(config.data)
    snapshot["output"] = str(config.output)
    metadata = {
        "version": __version__,
        "created_utc": stamp,
        "synthetic": synthetic,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": version("matplotlib"),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "synthetic_metadata": json.loads(
            (config.data / "SYNTHETIC_DATA.json").read_text(encoding="utf-8")
        )
        if synthetic
        else None,
        "config": snapshot,
        "inputs": inputs,
        "audit_only": audit_only,
        "execution": "close signal -> publication lag -> next open -> following open",
        "cost_model": "single-sided traded notional, drift-aware, fixed rate",
        "confirmed_investment_findings": 0,
        "research_role": "legacy factor diagnostics; use study for controlled comparisons",
    }
    (directory / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not audit_only:
        if not summaries:
            raise ValueError(f"No analyzable data. Inspect {directory / 'quality.csv'}")
        summary = pd.DataFrame(summaries)
        summary.to_csv(directory / "summary.csv", index=False, encoding="utf-8-sig")
        plot_overview(summary, ledgers, directory, synthetic)
        write_report(directory, summary, quality_df, synthetic)
    return directory
