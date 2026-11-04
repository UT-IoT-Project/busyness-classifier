"""Download one real data-collection session from the API.

A session is a period in which the nodes were sending data and a person was
setting the occupancy level (busy_level). Each session is saved in its own file,
data/raw/<name>.json, in the same format as the API. train.py uses each file as
one group for validation: it trains on some sessions and tests on another.

Run from the repository root (Windows PowerShell):
    $env:PYTHONPATH = "src"
    python -m occupancy.download --session session1 --start "2026-11-03 10:00" --end "2026-11-03 12:30"

Times are in Amsterdam local time, as the API expects.
"""
import argparse
import json
from datetime import datetime, timedelta

from occupancy.api_client import ApiSource, format_time
from occupancy.train import ROOT, load_config

MAX_CHUNK = timedelta(hours=24)        # the API does not accept more than 24 h per request
ONE_MS = timedelta(milliseconds=1)


def download(source, start, end):
    """Request the period in chunks of at most 24 h and merge the readings."""
    merged = {}
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + MAX_CHUNK, end)
        print("Requesting", format_time(chunk_start), "->", format_time(chunk_end))
        data = source.fetch(chunk_start, chunk_end)
        if data is None:
            print("  no data in this chunk")
        else:
            for sensor, readings in data.items():
                merged.setdefault(sensor, []).extend(readings)
        # Both ends are included: the next chunk starts 1 ms later
        chunk_start = chunk_end + ONE_MS
    return merged


def main():
    parser = argparse.ArgumentParser(description="Download a session of real data")
    parser.add_argument("--session", required=True, help="session name (used as file name)")
    parser.add_argument("--start", required=True, help='start, "YYYY-MM-DD HH:MM" (Amsterdam time)')
    parser.add_argument("--end", required=True, help='end, "YYYY-MM-DD HH:MM" (Amsterdam time)')
    args = parser.parse_args()

    config = load_config()
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
    if end <= start:
        raise SystemExit("The end must be after the start")

    data = download(ApiSource(config), start, end)
    if not any(data.values()):
        raise SystemExit("The API returned no data for that period: nothing saved")

    labels = data.get(config["cleaning"]["label"]["sensor"], [])
    print({sensor: len(readings) for sensor, readings in data.items()})
    if not labels:
        print("WARNING: the session has no label (busy_level): it cannot be used for training")

    path = ROOT / "data" / "raw" / f"{args.session}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    session = {"session": args.session,
               "building": config["api"]["building"],
               "room": config["api"]["room"],
               "start": format_time(start),
               "end": format_time(end),
               "data": data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f)
    print("Saved to", path)


if __name__ == "__main__":
    main()
