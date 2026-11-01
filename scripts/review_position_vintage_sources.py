"""Verify saved endpoint probes and distinguish a current copy from a historical capture."""

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import pandas as pd


def review(config: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Vintage evidence output must be new")
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    probe = root / plan["paths"]["probe"]
    original = root / plan["paths"]["raw"]
    manifest_file = original / "manifest.json"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if digest(manifest_file) != plan["input_hashes"]["raw_manifest"]:
        raise ValueError("Original ranking receipt has changed")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = {row["date"]: row["sha256"] for row in manifest["records"] if row["status"] == "ok"}
    probe_file = probe / "probe_manifest.json"
    saved = json.loads(probe_file.read_text(encoding="utf-8"))
    cases = []
    for row in saved["records"]:
        result = {
            "name": row["name"],
            "url": row["url"],
            "http_status": row.get("http_status"),
            "report_date": row.get("report_date"),
            "json_update_date": row.get("update_date"),
            "http_last_modified": row.get("headers", {}).get("Last-Modified"),
            "identical_saved_current_copy": False,
            "historical_capture_proved": False,
            "status": "unavailable",
            "error": row.get("error", ""),
        }
        if "sha256" in row:
            source = probe / "responses" / f"{row['name']}.bin"
            if digest(source) != row["sha256"]:
                raise ValueError("Probe response no longer matches its receipt")
            if row["name"].startswith("alias_") and row.get("http_status") == 200:
                date = row["name"].split("_")[1]
                first = original / "responses" / f"{date}.json"
                if digest(first) != expected[date]:
                    raise ValueError("Original ranking response has changed")
                result["identical_saved_current_copy"] = digest(first) == row["sha256"]
                result["status"] = (
                    "current_copy_matches"
                    if result["identical_saved_current_copy"]
                    else "different_current_copy_needs_investigation"
                )
            elif row["name"].startswith("archive_") and row.get("http_status") == 200:
                index = json.loads(source.read_bytes())
                result["status"] = (
                    "no_index_entries_for_queried_url_range"
                    if index == []
                    else "index_entries_need_capture_validation"
                )
            elif row["name"] == "official_pm_script":
                result["status"] = "current_frontend_source_only"
        cases.append(result)
    output.mkdir(parents=True)
    frame = pd.DataFrame(cases)
    frame.to_csv(output / "source_cases.csv", index=False, encoding="utf-8-sig")
    result = {
        "kind": "bounded_position_vintage_source_review",
        "probe_cases": len(cases),
        "matching_current_alias_copies": int(frame.identical_saved_current_copy.sum()),
        "historical_captures_validated": 0,
        "all_44_dates_researched_exhaustively": False,
        "availability_semantics_documented_as_initial_publication": False,
        "protocol_sha256": digest(config),
        "probe_manifest_sha256": digest(probe_file),
        "original_manifest_sha256": digest(manifest_file),
        "script_sha256": digest(Path(__file__)),
        "claim_eligible": False,
        "limitation": "Current copies and HTTP modification metadata do not establish initial release or numeric revisions; only two exact archive URLs were queried",
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shown = frame.loc[
        frame.name.str.startswith("alias_"),
        ["report_date", "json_update_date", "http_last_modified", "identical_saved_current_copy"],
    ]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>历史版本来源核查</title><style>body{{font:16px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;color:#173149;max-width:1200px;margin:40px auto;padding:20px}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}a{{color:#147d92}}</style><h1>本次公开入口核查未找回历史初发版本</h1><p>旧接口四个样本请求均为 DNS 解析失败；官方别名四个样本与原下载完整响应字节一致。两条具体 URL 的公开快照索引在查询时段内返回空列表，不能据此断言全部互联网档案不存在。</p>{shown.to_html(index=False)}<p>其中正常对照日 2024-02-19 的 JSON 更新时间在当日，但 HTTP Last-Modified 为 2024-03-20。两个时钟都不能单独证明首次发布时间或数值何时修订。对后面三个异常案例，两者时间相符；没有找到首次发布语义的官方说明。</p><p>官方当前前端使用报告日期展示标题，旧、新路径由 AKShare 官方源码与交易所脚本核对。当前副本相同不能充当历史时间戳证据，本轮恢复的原日历史版本数为零。</p><a href="source_cases.csv">全部请求与状态</a> · <a href="audit.json">校验与范围</a></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/position_vintage_review.toml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(review(args.config, args.output))
