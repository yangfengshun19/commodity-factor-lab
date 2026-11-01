"""Read-only Sina continuous-series snapshots with raw responses and hashes.

Endpoint follows the public AKShare futures_zh_daily_sina implementation. These
are retrospectively downloaded continuous prices, never certified contract P&L.
No network operation occurs on import or during the demo.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .catalog import SECTOR_MAP
from .data import normalize_snapshot


def parse_sina_daily(payload: bytes) -> pd.DataFrame:
    text = payload.decode("utf-8")
    opening, closing = text.find("(["), text.rfind("]")
    if opening < 0 or closing < opening:
        raise ValueError("Unexpected Sina JSONP response")
    records = json.loads(text[opening + 1 : closing + 1])
    if not isinstance(records, list) or not records:
        raise ValueError("No daily price records returned")
    frame = pd.DataFrame(records).rename(
        columns={
            "d": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "p": "open_interest",
            "s": "settlement",
        }
    )
    cleaned, _ = normalize_snapshot(frame, ())
    if "volume" not in frame:
        raise ValueError("Provider response lacks volume")
    numeric_volume = pd.to_numeric(frame["volume"], errors="coerce")
    volume = pd.Series(numeric_volume.to_numpy(), index=pd.to_datetime(frame["date"]))
    cleaned["volume"] = cleaned["date"].map(volume)
    if cleaned.empty:
        raise ValueError("No valid dated bars in provider response")
    return cleaned


def fetch_snapshot(output: Path, codes: list[str], through: str) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Snapshot directory must be new; no data is overwritten")
    codes = sorted(set(code.lower() for code in codes))
    if not codes or any(
        code not in SECTOR_MAP or SECTOR_MAP[code] == "financial" for code in codes
    ):
        raise ValueError("Use registered commodity product codes")
    cutoff = pd.Timestamp(through)
    if pd.isna(cutoff):
        raise ValueError("Invalid final complete session")
    output.mkdir(parents=True)
    (output / "responses").mkdir()
    (output / "trend").mkdir()
    metadata = {
        "provider": "Sina Finance",
        "kind": "continuous_price_proxy",
        "through": through,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": [],
        "point_in_time_certified": False,
        "contract_roll_verified": False,
        "reference": "https://github.com/akfamily/akshare/blob/main/akshare/futures/futures_zh_sina.py",
    }
    try:
        for number, code in enumerate(codes, 1):
            url = (
                "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_data=/InnerFuturesNewService.getDailyKLine?symbol="
                + code.upper()
                + "0"
            )
            record = {"code": code, "url": url, "status": "failed"}
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": "CommodityFactorLab/0.2 research",
                        "Referer": "https://finance.sina.com.cn/",
                    },
                )
                with urlopen(request, timeout=20) as response:
                    payload = response.read(5_000_001)
                if len(payload) > 5_000_000:
                    raise ValueError("Unexpectedly large daily response")
                raw_path = output / "responses" / f"{code}.jsonp"
                raw_path.write_bytes(payload)
                frame = parse_sina_daily(payload)
                frame = frame.loc[frame["date"] <= cutoff]
                if frame.empty:
                    raise ValueError("No prices on or before the declared final session")
                path = output / "trend" / f"Sina_{code}_daily.csv"
                frame.to_csv(path, index=False)
                record.update(
                    status="ok",
                    rows=len(frame),
                    start=str(frame["date"].min().date()),
                    end=str(frame["date"].max().date()),
                    raw_sha256=hashlib.sha256(payload).hexdigest(),
                    normalized_file=path.relative_to(output).as_posix(),
                    normalized_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            except (ValueError, KeyError, OSError, HTTPError, URLError) as error:
                record["error"] = str(error)
            record["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["records"].append(record)
            print(f"snapshot {number}/{len(codes)} {code}: {record['status']}", flush=True)
            if number < len(codes):
                time.sleep(0.4)
    finally:
        (output / "snapshot.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return output
