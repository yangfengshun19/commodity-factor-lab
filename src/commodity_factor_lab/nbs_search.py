"""Public NBS delegated search metadata only. Search summaries never become price inputs."""

import hashlib
import http.client
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from .nbs_archive import (
    HistoryCapture,
    discover_static_indexes,
    freeze_history_catalog,
    history_plan,
    official_links,
    period_from_title,
)
from .nbs_ingest import digest


class PublicNbsSearch:
    def __init__(self, config: Path):
        self.plan, self.paths = history_plan(config)
        self.directory = self.paths["raw"] / "search_provider"
        self.directory.mkdir(exist_ok=True)
        self.receipt = self.directory / "capture_manifest.json"
        self.connection, self.last_start = None, 0.0
        self.manifest = (
            json.loads(self.receipt.read_text(encoding="utf-8"))
            if self.receipt.exists()
            else {
                "protocol_sha256": digest(config),
                "records": [],
                "requests": 0,
                "access_stop": False,
                "purpose": "official_link_discovery_only_not_price_ingestion",
            }
        )
        if self.manifest["protocol_sha256"] != digest(config) or self.manifest["access_stop"]:
            raise ValueError("Search capture plan changed or an access stop remains active")
        for item in self.manifest["records"]:
            if "file" in item and digest(self.directory / item["file"]) != item["sha256"]:
                raise ValueError("Saved public search metadata has changed")
        site = json.loads((self.directory / "site_config.json").read_text(encoding="utf-8"))
        self.max_pages = next(
            tab["maxPage"] for tab in site["data"]["tabs"] if tab["tabName"] == "all"
        )
        if not 1 <= self.max_pages <= 10:
            raise ValueError("Public search page-limit evidence is missing or changed")

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def checkpoint(self):
        self.receipt.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def request(self, identifier: str, endpoint: str, params: dict) -> dict:
        if (
            not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier)
            or endpoint not in {"s", "combin"}
            or params["siteCode"] != "bm36000002"
            or not 1 <= params.get("page", 1) <= self.max_pages
        ):
            raise ValueError("Only the documented public search/link expansion is allowed")
        for record in self.manifest["records"]:
            if record["id"] == identifier:
                if (
                    record["params"] != params
                    or record["endpoint"] != endpoint
                    or record["status"] != "ok"
                ):
                    raise ValueError("Cached search identity changed or failed")
                return json.loads((self.directory / record["file"]).read_text(encoding="utf-8"))
        if self.manifest["access_stop"]:
            raise ValueError("Public search access stop remains active")
        discovery = json.loads(
            (self.paths["raw"] / "discovery/manifest.json").read_text(encoding="utf-8")
        )
        ancillary = len(list(self.directory.glob("*.receipt.json"))) + len(
            json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        )
        if (
            discovery["requests"] + ancillary + self.manifest["requests"]
            >= self.plan["discovery"]["maximum_requests"]
        ):
            raise ValueError("Shared metadata-discovery request budget exhausted")
        time.sleep(
            max(
                0,
                self.plan["collection"]["minimum_interval_seconds"]
                - (time.monotonic() - self.last_start),
            )
        )
        self.last_start = time.monotonic()
        record = {
            "id": identifier,
            "endpoint": endpoint,
            "params": params,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.manifest["requests"] += 1
        try:
            if self.connection is None:
                self.connection = http.client.HTTPSConnection("api.so-gov.cn", timeout=25)
            self.connection.request(
                "POST",
                "/query/" + endpoint,
                body=urlencode(params),
                headers={
                    "User-Agent": "CommodityFactorLab public statistical link discovery",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            response = self.connection.getresponse()
            content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise ValueError("Search metadata exceeds its response budget")
            file = identifier + ".json"
            if (self.directory / file).exists():
                raise FileExistsError("Unregistered search result file exists")
            (self.directory / file).write_bytes(content)
            record.update(
                file=file,
                sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content),
                http_status=response.status,
            )
            if response.status in (403, 429):
                self.manifest["access_stop"] = True
                raise ValueError("Public search access limited; do not retry another query")
            result = json.loads(content)
            if response.status != 200 or not result.get("ok"):
                self.manifest["access_stop"] = True
                raise ValueError("Public search rejected the request; do not bypass validation")
            record["status"] = "ok"
        except (OSError, http.client.HTTPException, ValueError) as error:
            self.close()
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.manifest["records"].append(record)
            self.checkpoint()
            raise
        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["records"].append(record)
        self.checkpoint()
        return result


def public_search_links(result: dict, receipt: str) -> list[dict]:
    observations = []
    for item in result.get("resultDocs", []):
        data = item.get("data", {})
        period = period_from_title(data.get("titleO", ""))
        url = data.get("url", "")
        parsed = urlsplit(url)
        if (
            period is not None
            and 2020 <= int(period[:4]) <= 2025
            and parsed.hostname == "www.stats.gov.cn"
            and parsed.scheme in {"http", "https"}
        ):
            observations.append(
                {
                    "period": period,
                    "url": "https://" + parsed.netloc + parsed.path,
                    "title": data["titleO"],
                    "discovered_on": receipt,
                    "observed_url": url,
                    "search_reference": item.get("id"),
                    "search_md5_title": data.get("md5Title"),
                    "search_combined_copies": data.get("combins", 1),
                }
            )
    return observations


def discover_early_history(config: Path, rules: Path | None = None) -> dict:
    capture = PublicNbsSearch(config)
    static = json.loads(
        (capture.paths["raw"] / "static_discovery.json").read_text(encoding="utf-8")
    )
    observations = list(static["observations"])
    try:
        if rules is None:
            years = (2020, 2021)
        else:
            import tomllib

            cancelled = {
                item["period"]
                for item in tomllib.loads(rules.read_text(encoding="utf-8"))["cancellations"]
            }
            located = {item["period"] for item in observations}
            years = [
                year
                for year in range(2020, 2026)
                if any(
                    f"{year}-{month:02d}-{part}" not in cancelled | located
                    for month in range(1, 13)
                    for part in capture.plan["history"]["parts"]
                )
            ]
        for year in years:
            # This exact AND form is generated by the public site's advanced-search UI.
            base = {
                "siteCode": "bm36000002",
                "tab": "all",
                "qt": f'("{year}年" AND "流通领域重要生产资料市场价格变动情况")',
                "adv": 1,
                "keyplace": 1,
                "sort": "dateDesc",
                "pageSize": 20,
            }
            first = capture.request(f"year_{year}_page_1", "s", {**base, "page": 1})
            pages = math.ceil(first["totalHits"] / max(1, first["currentHits"]))
            if not 1 <= pages <= capture.max_pages:
                raise ValueError(
                    "Refine the normal public query; never exceed its advertised page limit"
                )
            for page in range(1, pages + 1):
                result = (
                    first
                    if page == 1
                    else capture.request(f"year_{year}_page_{page}", "s", {**base, "page": page})
                )
                found = public_search_links(result, f"year_{year}_page_{page}")
                observations.extend(found)
                print(
                    f"Public year query {year} page {page}/{pages}; {len(found)} primary links",
                    flush=True,
                )
        # Normal duplicate-expansion links, only where the static catalog has no modern primary page.
        modern = {
            item["period"]
            for item in observations
            if "/sj/zxfb/" in item["url"] or item.get("seed")
        }
        candidates = {
            item["period"]: item
            for item in observations
            if item.get("search_reference")
            and item["period"] not in modern
            and item.get("search_combined_copies", 1) > 1
        }
        for period, item in sorted(candidates.items()):
            params = {
                "siteCode": "bm36000002",
                "tab": "all",
                "qt": "流通领域重要生产资料市场价格变动情况",
                "page": 1,
                "pageSize": 20,
                "reference": item["search_reference"],
                "combinField": "MD5TITLE",
                "combinValue": item["search_md5_title"],
            }
            result = capture.request("copies_" + period, "combin", params)
            found = public_search_links(result, "copies_" + period)
            observations.extend(found)
            print(f"Public duplicate links {period}: {len(found)}", flush=True)
        output = {
            "observations": observations,
            "distinct_periods": len({item["period"] for item in observations}),
            "search_metadata_used_for_prices": False,
            "public_page_limit_respected": capture.max_pages,
        }
        (capture.paths["raw"] / "discovered_links.json").write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output
    finally:
        capture.close()


def bootstrap_search_metadata(config: Path) -> None:
    """Read only interface metadata explicitly referenced by the public official search page."""
    capture = HistoryCapture(config, "discovery")
    directory = capture.paths["raw"] / "search_provider"
    directory.mkdir(exist_ok=True)
    stop = directory / "bootstrap_access_stop.json"
    if stop.exists():
        raise ValueError("A previous public-search bootstrap access stop remains active")
    manifest_path = directory / "manifest.json"
    sources = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    )

    def read_public(host, path, method="GET", body=None):
        if host not in {"p.so-gov.cn", "api.so-gov.cn"}:
            raise ValueError("Unexpected delegated-search interface host")
        discovery_count = capture.manifest["requests"]
        search_receipt = directory / "capture_manifest.json"
        search_count = (
            json.loads(search_receipt.read_text(encoding="utf-8"))["requests"]
            if search_receipt.exists()
            else 0
        )
        if (
            discovery_count
            + search_count
            + len(sources)
            + len(list(directory.glob("*.receipt.json")))
            >= capture.plan["discovery"]["maximum_requests"]
        ):
            raise ValueError("Shared public-metadata request budget exhausted")
        time.sleep(capture.plan["collection"]["minimum_interval_seconds"])
        connection = http.client.HTTPSConnection(host, timeout=25)
        record = {
            "url": "https://" + host + path,
            "method": method,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            connection.request(
                method,
                path,
                body=body,
                headers={
                    "User-Agent": "CommodityFactorLab public search interface inspection",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json,text/javascript",
                },
            )
            response = connection.getresponse()
            content = response.read(1_000_001)
            record.update(
                http_status=response.status,
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            if response.status != 200 or len(content) > 1_000_000:
                stop.write_text(json.dumps(record, indent=2), encoding="utf-8")
                raise ValueError(
                    "Public-search interface rejected the read; inspect retained receipt"
                )
            return content, record
        finally:
            connection.close()

    try:
        schedule_url = capture.plan["discovery"]["schedule_index"].rstrip("/") + "/index.html"
        listing = capture.get("schedule_index", schedule_url, "index")
        links = official_links(listing, schedule_url)
        for year in range(2020, 2026):
            urls = sorted(
                {
                    item["url"]
                    for item in links
                    if item["title"].startswith(str(year) + "年国家统计局")
                }
            )
            if len(urls) != 1:
                raise ValueError("Official yearly schedule link is missing or ambiguous")
            capture.get("schedule_" + str(year), urls[0], "schedule")
        search_url = "https://www.stats.gov.cn/search/s?" + urlencode(
            {"qt": "流通领域重要生产资料市场价格变动情况"}
        )
        page = capture.get("search_bootstrap", search_url, "index").decode("utf-8")
        if "https://api.so-gov.cn/" not in page or "bm36000002" not in page:
            raise ValueError(
                "The current official search interface does not establish this public provider"
            )
        for name in ("api", "query", "util"):
            url = f"https://p.so-gov.cn/web/api/{name}.js"
            if url not in page:
                raise ValueError(
                    "An interface script is no longer referenced by the official search page"
                )
            path = directory / (name + ".js")
            sidecar = directory / (name + ".receipt.json")
            if path.exists():
                receipt = next((item for item in sources if item["url"] == url), None)
                if receipt is None and sidecar.exists():
                    receipt = json.loads(sidecar.read_text(encoding="utf-8"))
                if (
                    receipt is None
                    or receipt["http_status"] != 200
                    or digest(path) != receipt["sha256"]
                ):
                    raise ValueError("Cached interface script lacks matching provenance")
                continue
            content, receipt = read_public("p.so-gov.cn", f"/web/api/{name}.js")
            path.write_bytes(content)
            sources.append(receipt)
            manifest_path.write_text(
                json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        site = directory / "site_config.json"
        receipt_path = directory / "site_config.receipt.json"
        if site.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["http_status"] != 200 or digest(site) != receipt["sha256"]:
                raise ValueError("Cached public-site configuration changed")
        else:
            content, receipt = read_public(
                "api.so-gov.cn", "/query/site", "POST", urlencode({"siteCode": "bm36000002"})
            )
            site.write_bytes(content)
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            if not json.loads(content).get("ok"):
                stop.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                raise ValueError("Public-site configuration rejected the request")
    finally:
        capture.close()


def discover_history(config: Path, rules: Path) -> dict:
    _, paths = history_plan(config)
    if (paths["raw"] / "catalog.json").exists():
        return freeze_history_catalog(config, rules)
    bootstrap_search_metadata(config)
    discover_static_indexes(config)
    discover_early_history(config, rules)
    return freeze_history_catalog(config, rules)
