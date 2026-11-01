"""SHFE annual contract bars and dated contract metadata, with immutable receipts.

Annual files are revised historical exports, not point-in-time data. Excel merged
contract labels are carried down; prices and trading dates are never filled.
"""

import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd

INDEX_URL = "https://www.shfe.cn/reports/tradedata/datadownload/download.json"
SHFE_CODES = {
    "cu",
    "al",
    "zn",
    "pb",
    "ni",
    "sn",
    "rb",
    "hc",
    "ru",
    "bu",
    "fu",
    "sp",
    "au",
    "ag",
    "ss",
    "ao",
    "br",
    "wr",
}
PILOT_CODES = ("cu", "al", "zn", "rb", "hc", "ru")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_public(url: str, maximum: int = 100_000_000) -> bytes:
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in {
        "www.shfe.cn",
        "www.shfe.com.cn",
    }:
        raise ValueError("Only documented official SHFE HTTPS sources are allowed")
    request = Request(url, headers={"User-Agent": "CommodityFactorLab/0.3 research"})
    chunks, total, start = [], 0, time.monotonic()
    with urlopen(request, timeout=25) as response:
        while chunk := response.read(262144):
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum or time.monotonic() - start > 300:
                raise ValueError("Official response exceeds download size/time budget")
    return b"".join(chunks)


def parse_annual_excel(payload: bytes, codes: tuple[str, ...]) -> pd.DataFrame:
    """Read a single official workbook; only merged identity cells may be filled."""
    book = pd.ExcelFile(io.BytesIO(payload), engine="calamine")
    output = []
    aliases = {
        "合约": "contract",
        "日期": "date",
        "交易日期": "date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "结算价": "settle",
        "前结算": "pre_settle",
        "成交量": "volume",
        "持仓量": "open_interest",
    }
    for sheet in book.sheet_names:
        raw = pd.read_excel(book, sheet_name=sheet, header=None)
        header = next(
            (
                i
                for i in range(min(12, len(raw)))
                if "合约" in raw.iloc[i].astype(str).str.strip().tolist()
                and {"日期", "交易日期"}.intersection(raw.iloc[i].astype(str).str.strip().tolist())
            ),
            None,
        )
        if header is None:
            continue
        names = [aliases.get(str(v).strip(), f"unused_{i}") for i, v in enumerate(raw.iloc[header])]
        frame = raw.iloc[header + 1 :].copy()
        frame.columns = names
        if set(aliases.values()) - set(frame):
            raise ValueError(f"Official workbook schema changed in {sheet}")
        frame["contract"] = frame["contract"].ffill().astype(str).str.strip().str.lower()
        frame["code"] = frame["contract"].str.extract(r"^([a-z]+)\d{4}$", expand=False)
        frame = frame.loc[frame["code"].isin(codes)].copy()
        # Ignore bilingual headers/footers, but do not discard malformed product rows.
        text_dates = frame["date"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        dates = pd.to_datetime(text_dates, format="%Y%m%d", errors="coerce")
        summary = text_dates.isin(["nan", "小计", "合计", "总计", "Total"])
        if (dates.isna() & ~summary).any():
            raise ValueError("Invalid date in selected official contract rows")
        frame = frame.loc[dates.notna()].copy()
        frame["date"] = dates.loc[dates.notna()]
        for col in set(aliases.values()) - {"date", "contract"}:
            frame[col] = pd.to_numeric(
                frame[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
        output.append(frame[["date", "code", *[x for x in aliases.values() if x != "date"]]])
    if not output:
        raise ValueError("No recognised official annual worksheet")
    return pd.concat(output, ignore_index=True)


def parse_annual_zip(path: Path, codes: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            # INE is intentionally outside this pilot's exchange and calendar.
            if "ine" in member.filename.lower() or not member.filename.lower().endswith(
                (".xls", ".xlsx")
            ):
                continue
            if member.file_size > 180_000_000:
                raise ValueError("Unexpectedly large annual workbook")
            frame = parse_annual_excel(archive.read(member), codes)
            frame["source_member"] = member.filename
            frames.append(frame)
    if not frames:
        raise ValueError("No SHFE contract worksheets in official annual archive")
    result = pd.concat(frames, ignore_index=True).sort_values(["date", "code", "contract"])
    if result.duplicated(["date", "contract"]).any():
        raise ValueError("Duplicate official contract/date keys; no silent deduplication")
    return result


def parse_contract_metadata(payload: bytes, source: str) -> pd.DataFrame:
    data = json.loads(payload)
    frame = pd.DataFrame(data["ContractBaseInfo"])
    frame = frame.rename(
        columns={
            "INSTRUMENTID": "contract",
            "EXPIREDATE": "expiry",
            "OPENDATE": "listed",
            "UPDATE_DATE": "metadata_available_at",
        }
    )
    frame["contract"] = frame["contract"].str.strip().str.lower()
    frame["code"] = frame["contract"].str.extract(r"^([a-z]+)\d{4}$", expand=False)
    frame = frame.loc[frame["code"].isin(SHFE_CODES)].copy()
    for col in ("expiry", "listed"):
        frame[col] = pd.to_datetime(frame[col], format="%Y%m%d", errors="raise")

    def local_clock(value):
        # A date alone is not evidence that data was known at midnight.
        if not isinstance(value, str) or not re.search(r"\d{2}:\d{2}:\d{2}", value):
            return pd.NaT
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Shanghai")
        return timestamp.tz_convert("UTC")

    file_clock = local_clock(data.get("update_date"))
    row_clocks = frame.get("metadata_available_at", pd.Series(None, index=frame.index)).map(
        local_clock
    )
    available = []
    sources = []
    for row_clock in row_clocks:
        clocks = [v for v in (row_clock, file_clock) if pd.notna(v)]
        available.append(max(clocks) if clocks else pd.NaT)
        sources.append(
            "max_row_and_file_update"
            if len(clocks) == 2
            else (
                "file_update"
                if pd.notna(file_clock)
                else ("row_update" if pd.notna(row_clock) else "unknown")
            )
        )
    frame["metadata_available_at"] = pd.to_datetime(available, utc=True)
    frame["timestamp_source"] = sources
    frame["timing_status"] = frame.metadata_available_at.notna().map(
        {True: "reported_update_clock", False: "unknown_clock"}
    )
    frame["source"] = source
    if frame.duplicated(["contract", "metadata_available_at"]).any():
        raise ValueError("Duplicate contract metadata publication keys")
    return frame[
        [
            "contract",
            "code",
            "expiry",
            "listed",
            "metadata_available_at",
            "timestamp_source",
            "timing_status",
            "source",
        ]
    ]


def reparse_metadata(source: Path, output: Path) -> Path:
    """Rebuild derived metadata from saved responses; preserve the original failures."""
    if output.exists():
        raise FileExistsError("Derived metadata directory must be new")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    records, frames = [], []
    for old in manifest["records"]:
        path = source / "responses" / f"{old['date']}.json"
        record = {
            "query_date": old["date"],
            "url": old["url"],
            "original_status": old["status"],
            "status": "failed",
        }
        try:
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if old.get("sha256", checksum) != checksum:
                raise ValueError("Saved response no longer matches original receipt")
            if str(json.loads(raw).get("report_date")) != old["date"]:
                raise ValueError("Response report date differs from requested historical date")
            frame = parse_contract_metadata(raw, old["url"])
            frames.append(frame)
            record.update(
                status="ok",
                raw_sha256=checksum,
                rows=len(frame),
                unknown_clocks=int(frame.metadata_available_at.isna().sum()),
            )
        except (ValueError, KeyError, OSError) as error:
            record["error"] = str(error)
        records.append(record)
    output.mkdir(parents=True)
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(output / "metadata.csv", index=False)
    receipt = {
        "kind": "reparsed_official_contract_metadata",
        "input_manifest": str((source / "manifest.json").resolve()),
        "input_manifest_sha256": digest(source / "manifest.json"),
        "records": records,
        "timestamp_timezone_assumption": "source clocks without offset are Asia/Shanghai",
        "point_in_time_certified": False,
        "normalized_sha256": digest(output / "metadata.csv") if frames else None,
    }
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return output


def ingest_archives(
    archives: list[Path], output: Path, codes: tuple[str, ...] = PILOT_CODES
) -> Path:
    if output.exists() or not codes or set(codes) - SHFE_CODES:
        raise ValueError("Use a new output directory and registered SHFE codes")
    frames, inputs = [], []
    for path in archives:
        print(f"Reading official archive: {path.name}", flush=True)
        frame = parse_annual_zip(path, codes)
        frames.append(frame)
        inputs.append({"file": str(path.resolve()), "sha256": digest(path)})
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code", "contract"])
    if panel.duplicated(["date", "contract"]).any():
        raise ValueError("Annual archives overlap; refuse silent deduplication")
    output.mkdir(parents=True)
    path = output / "bars.csv"
    panel.to_csv(path, index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "official_revised_contract_bars",
                "exchange": "SHFE",
                "codes": list(codes),
                "inputs": inputs,
                "normalized_sha256": digest(path),
                "rows": len(panel),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "point_in_time_certified": False,
                "execution_ready": False,
                "limitations": [
                    "annual revisions not historic vintages",
                    "expiry metadata must be joined as of publication",
                    "actual session openings and historical costs/margins not yet certified",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def fetch_metadata(dates: list[str], output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Metadata snapshot directory must be new")
    days = sorted(set(dates))
    if not days:
        raise ValueError("At least one metadata query date is required")
    for day in days:
        if not re.fullmatch(r"\d{8}", day):
            raise ValueError("Metadata query dates must be YYYYMMDD")
        datetime.strptime(day, "%Y%m%d")
    output.mkdir(parents=True)
    (output / "responses").mkdir()
    records, frames = [], []
    for date in days:
        url = f"https://www.shfe.com.cn/data/busiparamdata/future/ContractBaseInfo{date}.dat"
        record = {"date": date, "url": url, "status": "failed"}
        try:
            raw = read_public(url, maximum=2_000_000)
            (output / "responses" / f"{date}.json").write_bytes(raw)
            record["sha256"] = hashlib.sha256(raw).hexdigest()
            if str(json.loads(raw).get("report_date")) != date:
                raise ValueError("Metadata response report date differs from query")
            frame = parse_contract_metadata(raw, url)
            frames.append(frame)
            record.update(status="ok", sha256=hashlib.sha256(raw).hexdigest())
        except (ValueError, KeyError, OSError) as error:
            record["error"] = str(error)
        record["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
        records.append(record)
        print(f"Contract metadata {date}: {record['status']}", flush=True)
        time.sleep(0.2)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output / "metadata.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "official_contract_metadata",
                "planned_dates": days,
                "records": records,
                "normalized_sha256": digest(output / "metadata.csv") if frames else None,
                "timestamp_timezone_assumption": "exchange local timestamps are Asia/Shanghai",
                "point_in_time_certified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def fetch_annual(years: list[int], output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Annual download directory must be new")
    index_raw = read_public(INDEX_URL, maximum=1_000_000)
    listing = json.loads(index_raw)["data"]["historicalData"]
    selected = []
    for year in sorted(set(years)):
        candidates = [r for r in listing if r["year"] == str(year) and r["type"] == "1"]
        if len(candidates) != 1:
            raise ValueError(f"No unique completed official futures year {year}")
        selected.append((year, urljoin(INDEX_URL, candidates[0]["url"])))
    output.mkdir(parents=True)
    (output / "download.json").write_bytes(index_raw)
    records = []
    try:
        for year, url in selected:
            record = {"year": year, "url": url, "status": "failed"}
            try:
                raw = read_public(url)
                if not zipfile.is_zipfile(io.BytesIO(raw)):
                    raise ValueError("Official yearly response is not a ZIP archive")
                path = output / f"{year}.zip"
                path.write_bytes(raw)
                record.update(status="ok", sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            except (ValueError, OSError) as error:
                record["error"] = str(error)
            record["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
            records.append(record)
            print(f"Official year {year}: {record['status']}", flush=True)
    finally:
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "index_url": INDEX_URL,
                    "index_sha256": hashlib.sha256(index_raw).hexdigest(),
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return output
