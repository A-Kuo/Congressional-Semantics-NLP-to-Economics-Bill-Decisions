"""
Link Stanford Congressional Record floor speeches to real bills via in-text
bill-number mentions (e.g. "H.R. 1234", "S. 815").

The Stanford corpus (data/raw/speeches/speeches_{congress}.txt) has NO bill_id
field -- it is organized by date/speaker only. This script recovers bill
linkage by regex-matching explicit bill references inside each speech and
keeping only mentions of bills already confirmed as economic-policy bills
(from scripts/fetch_real_bills.py -> data/processed/bills_metadata_real.csv).

Caveats (documented, not hidden):
  - A speech that mentions a bill is not proof the speech is substantively
    "about" that bill (could be a passing procedural reference).
  - A speech mentioning multiple bills is linked to all of them.
  - Only explicit numeric citations are captured; bills discussed only by
    title/nickname ("the Recovery Act") are missed.

Usage:
    python scripts/link_speeches_to_bills.py --congresses 114
    python scripts/link_speeches_to_bills.py --congresses 110,111,112,113,114
"""

import argparse
import os
import re

import pandas as pd

BILLS_PATH = "data/processed/bills_metadata_real.csv"
SPEECHES_DIR = "data/raw/speeches"
SPEAKERMAP_DIR = "data/raw/speakermap"
OUTPUT_DIR = "data/raw/speech_links"

HR_RE = re.compile(r"\bH\.?\s*R\.?\s*(\d{1,5})\b")
S_RE = re.compile(r"(?<!U\.)\bS\.\s?(\d{1,5})\b")


def load_bill_number_sets(congress: int, bills_df: pd.DataFrame) -> dict:
    """Return {'HR': {123, 456, ...}, 'S': {789, ...}} for one Congress."""
    sub = bills_df[bills_df["congress"] == congress]
    return {
        "HR": set(sub[sub["bill_type"] == "HR"]["bill_number"].astype(int)),
        "S": set(sub[sub["bill_type"] == "S"]["bill_number"].astype(int)),
    }


def link_congress(congress: int, bill_sets: dict) -> pd.DataFrame:
    speeches_path = f"{SPEECHES_DIR}/speeches_{congress}.txt"
    if not os.path.exists(speeches_path):
        print(f"  WARNING: {speeches_path} not found, skipping")
        return pd.DataFrame(columns=["speech_id", "bill_id", "congress"])

    hr_set, s_set = bill_sets["HR"], bill_sets["S"]
    rows = []
    n_lines = 0
    n_matched = 0

    with open(speeches_path, encoding="utf-8", errors="replace") as f:
        next(f, None)  # header: speech_id|speech
        for line in f:
            n_lines += 1
            parts = line.rstrip("\n").split("|", 1)
            if len(parts) != 2:
                continue
            speech_id, text = parts

            bill_ids_found = set()
            for num in HR_RE.findall(text):
                n = int(num)
                if n in hr_set:
                    bill_ids_found.add(f"HR{n}-{congress}")
            for num in S_RE.findall(text):
                n = int(num)
                if n in s_set:
                    bill_ids_found.add(f"S{n}-{congress}")

            if bill_ids_found:
                n_matched += 1
                for bill_id in bill_ids_found:
                    rows.append({"speech_id": speech_id, "bill_id": bill_id, "congress": congress})

            if n_lines % 100000 == 0:
                print(f"    ...{n_lines} speeches scanned, {n_matched} matched so far")

    print(f"  Congress {congress}: {n_lines} speeches scanned, "
          f"{n_matched} contained a tracked economic-bill mention "
          f"({len(rows)} speech-bill links)")
    return pd.DataFrame(rows)


def attach_speaker_party(links_df: pd.DataFrame, congress: int) -> pd.DataFrame:
    """Join speaker party from the SpeakerMap file (speech_id -> party)."""
    sm_path = f"{SPEAKERMAP_DIR}/{congress}_SpeakerMap.txt"
    if not os.path.exists(sm_path) or links_df.empty:
        links_df["party"] = "Unknown"
        return links_df

    sm = pd.read_csv(sm_path, sep="|", dtype=str, usecols=["speech_id", "party"])
    sm["speech_id"] = sm["speech_id"].astype(str)
    links_df["speech_id"] = links_df["speech_id"].astype(str)
    merged = links_df.merge(sm, on="speech_id", how="left")
    merged["party"] = merged["party"].fillna("Unknown")
    return merged


def main(congresses: list):
    bills_df = pd.read_csv(BILLS_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for congress in congresses:
        print(f"\n=== Linking speeches for Congress {congress} ===")
        bill_sets = load_bill_number_sets(congress, bills_df)
        print(f"  Tracking {len(bill_sets['HR'])} HR bills, {len(bill_sets['S'])} S bills")

        links = link_congress(congress, bill_sets)
        links = attach_speaker_party(links, congress)

        out_path = f"{OUTPUT_DIR}/links_{congress}.csv"
        links.to_csv(out_path, index=False)
        print(f"  Saved {len(links)} links -> {out_path}")
        print(f"  Unique bills with >=1 linked speech: {links['bill_id'].nunique() if len(links) else 0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congresses", type=str, default="110,111,112,113,114")
    args = parser.parse_args()
    congress_list = [int(c.strip()) for c in args.congresses.split(",")]
    main(congress_list)
