"""
Fetch real bill metadata from Congress.gov API for HR/S bills, 110th-114th Congress,
filtered to economic policy areas.

Two-stage filter (keeps API calls tractable):
  1. Title keyword prefilter (cheap, no extra API calls) narrows ~5-15k bills/Congress
     down to a candidate set.
  2. Official policyArea lookup (one API call per candidate) confirms membership in
     an economic policy area, matching the categories used in the report:
     Taxation, Labor and Employment, Foreign Trade and International Finance,
     Economics and Public Finance, Finance and Financial Sector.

Caches raw API responses to data/raw/bills_cache/ so re-runs are cheap and resumable.

Usage:
    python scripts/fetch_real_bills.py --congresses 114          # pilot / smoke test
    python scripts/fetch_real_bills.py --congresses 110,111,112,113,114
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

CACHE_DIR = "data/raw/bills_cache"
OUTPUT_PATH = "data/processed/bills_metadata_real.csv"
BILL_TYPES = ["hr", "s"]

# Official Congress.gov "Policy Area" categories that correspond to the report's
# economic-legislation scope (taxation, labor, trade, budget, economic policy).
ECONOMIC_POLICY_AREAS = {
    "Taxation",
    "Labor and Employment",
    "Foreign Trade and International Finance",
    "Economics and Public Finance",
    "Finance and Financial Sector",
    "Commerce",
}

# Cheap title prefilter before spending an API call on the subjects endpoint.
TITLE_KEYWORDS = re.compile(
    r"\b(tax|taxation|tariff|trade|import|export|customs|duty|duties|"
    r"labor|employment|unemployment|wage|workforce|worker|"
    r"budget|appropriation|spending|deficit|debt ceiling|"
    r"economic|economy|fiscal|financial|finance|revenue|"
    r"commerce|bank|banking|credit|corporation|business|industry)\b",
    re.IGNORECASE,
)


def get_api_key() -> str:
    key = os.environ.get("CONGRESS_API_KEY")
    if key:
        return key
    # Fall back to .env in repo root
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.strip().startswith("CONGRESS_API_KEY"):
                    return line.strip().split("=", 1)[1].strip()
    raise RuntimeError("CONGRESS_API_KEY not found in environment or .env")


def fetch_bill_list(congress: int, bill_type: str, api_key: str, delay: float = 0.4) -> list:
    """Paginate through all bills of one type for one Congress."""
    cache_path = f"{CACHE_DIR}/list_{congress}_{bill_type}.json"
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    base_url = f"https://api.congress.gov/v3/bill/{congress}/{bill_type}"
    all_bills = []
    offset = 0
    while True:
        params = {"api_key": api_key, "limit": 250, "offset": offset, "format": "json"}
        resp = requests.get(base_url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        bills = data.get("bills", [])
        if not bills:
            break
        all_bills.extend(bills)
        offset += 250
        if not data.get("pagination", {}).get("next"):
            break
        time.sleep(delay)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_bills, f)
    print(f"  fetched {len(all_bills)} {bill_type.upper()} bills for Congress {congress}")
    return all_bills


def fetch_policy_area(congress: int, bill_type: str, bill_number, api_key: str) -> str | None:
    cache_path = f"{CACHE_DIR}/subj_{congress}_{bill_type}_{bill_number}.json"
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        return cached.get("policyArea")

    url = f"https://api.congress.gov/v3/bill/{congress}/{bill_type}/{bill_number}/subjects"
    params = {"api_key": api_key, "format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        policy_area = data.get("subjects", {}).get("policyArea", {}).get("name")
    except requests.exceptions.RequestException as e:
        print(f"    WARN: subjects fetch failed for {bill_type}{bill_number}-{congress}: {e}")
        policy_area = None

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"policyArea": policy_area}, f)
    return policy_area


def fetch_policy_areas_concurrent(jobs: list, api_key: str, max_workers: int = 6) -> dict:
    """
    jobs: list of (congress, bill_type, bill_number) tuples.
    Returns {(congress, bill_type, bill_number): policy_area_or_None}.
    Cache hits are resolved synchronously first (no network); only cache
    misses go through the thread pool, which keeps concurrency modest and
    respectful of Congress.gov's rate limit.
    """
    results = {}
    to_fetch = []
    for congress, bill_type, number in jobs:
        cache_path = f"{CACHE_DIR}/subj_{congress}_{bill_type}_{number}.json"
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                results[(congress, bill_type, number)] = json.load(f).get("policyArea")
        else:
            to_fetch.append((congress, bill_type, number))

    if not to_fetch:
        return results

    print(f"    fetching {len(to_fetch)} uncached subjects ({max_workers} workers)...")
    done_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(fetch_policy_area, c, bt, n, api_key): (c, bt, n)
            for c, bt, n in to_fetch
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"    WARN: {key} failed: {e}")
                results[key] = None
            done_count += 1
            if done_count % 100 == 0:
                print(f"      ...{done_count}/{len(to_fetch)} fetched")

    return results


def process_congress(congress: int, api_key: str) -> pd.DataFrame:
    print(f"\n=== Congress {congress} ===")
    rows = []
    for bill_type in BILL_TYPES:
        bills = fetch_bill_list(congress, bill_type, api_key)
        candidates = [b for b in bills if TITLE_KEYWORDS.search(b.get("title", "") or "")]
        print(f"  {bill_type.upper()}: {len(bills)} total -> {len(candidates)} title-keyword candidates")

        jobs = [(congress, bill_type, b.get("number")) for b in candidates]
        policy_areas = fetch_policy_areas_concurrent(jobs, api_key)

        for bill in candidates:
            number = bill.get("number")
            policy_area = policy_areas.get((congress, bill_type, number))
            if policy_area not in ECONOMIC_POLICY_AREAS:
                continue

            latest_action_text = (bill.get("latestAction", {}) or {}).get("text", "") or ""
            passed = 1 if "became public law" in latest_action_text.lower() else 0

            rows.append({
                "congress": congress,
                "bill_type": bill_type.upper(),
                "bill_number": number,
                "bill_id": f"{bill_type.upper()}{number}-{congress}",
                "title": bill.get("title", ""),
                "status": latest_action_text,
                "status_date": (bill.get("latestAction", {}) or {}).get("actionDate", ""),
                "policy_area": policy_area,
                "passed": passed,
            })

        kept = sum(1 for r in rows if r["congress"] == congress and r["bill_type"] == bill_type.upper())
        print(f"  {bill_type.upper()}: {kept} confirmed economic-policy bills")

    df = pd.DataFrame(rows)
    print(f"  Congress {congress}: {len(df)} bills kept "
          f"(pass rate {df['passed'].mean():.1%})" if len(df) else f"  Congress {congress}: 0 bills kept")
    return df


def main(congresses: list):
    api_key = get_api_key()
    all_dfs = []
    for c in congresses:
        df_c = process_congress(c, api_key)
        all_dfs.append(df_c)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # merge with any prior run (e.g. pilot congress) instead of clobbering
    if os.path.exists(OUTPUT_PATH):
        prior = pd.read_csv(OUTPUT_PATH)
        combined = pd.concat([prior, combined], ignore_index=True)
        combined = combined.drop_duplicates(subset=["bill_id"], keep="last")

    combined = combined.sort_values(["congress", "bill_type", "bill_number"])
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} economic bills -> {OUTPUT_PATH}")
    print(f"Overall pass rate: {combined['passed'].mean():.1%}")
    print(combined.groupby("congress")["passed"].agg(["count", "mean"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congresses", type=str, default="110,111,112,113,114",
                        help="Comma-separated list of Congress numbers")
    args = parser.parse_args()
    congress_list = [int(c.strip()) for c in args.congresses.split(",")]
    main(congress_list)
