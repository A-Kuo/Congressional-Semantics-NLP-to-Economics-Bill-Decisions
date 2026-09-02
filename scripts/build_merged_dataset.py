"""
Final assembly: combine bills_metadata_real.csv + data/raw/speech_links/links_*.csv
+ raw speech text -> one bill-level row per bill, matching the schema expected
by run_pipeline.py (data/processed/bills_speeches_merged.csv and
bills_speeches_preprocessed.csv).

For each Congress, re-reads the raw speeches file ONCE, pulling out only the
text for speech_ids that appear in that Congress's link table (keeps memory
bounded instead of loading the whole multi-hundred-MB file into a DataFrame).

Usage:
    python scripts/build_merged_dataset.py --congresses 110,111,112,113,114
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, ".")

BILLS_PATH = "data/processed/bills_metadata_real.csv"
LINKS_DIR = "data/raw/speech_links"
SPEECHES_DIR = "data/raw/speeches"
MERGED_OUT = "data/processed/bills_speeches_merged.csv"
PREPROCESSED_OUT = "data/processed/bills_speeches_preprocessed.csv"


def load_speech_texts(congress: int, wanted_ids: set) -> dict:
    """Single pass over the raw speeches file, keeping only wanted speech_ids."""
    path = f"{SPEECHES_DIR}/speeches_{congress}.txt"
    texts = {}
    if not os.path.exists(path) or not wanted_ids:
        return texts
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("|", 1)
            if len(parts) != 2:
                continue
            speech_id, text = parts
            if speech_id in wanted_ids:
                texts[speech_id] = text
    return texts


def build_for_congress(congress: int, bills_df: pd.DataFrame) -> pd.DataFrame:
    links_path = f"{LINKS_DIR}/links_{congress}.csv"
    if not os.path.exists(links_path):
        print(f"  WARNING: {links_path} missing, skipping Congress {congress}")
        return pd.DataFrame()

    links = pd.read_csv(links_path, dtype={"speech_id": str})
    if links.empty:
        print(f"  Congress {congress}: no links found")
        return pd.DataFrame()

    wanted_ids = set(links["speech_id"].unique())
    print(f"  Congress {congress}: loading text for {len(wanted_ids)} speeches...")
    texts = load_speech_texts(congress, wanted_ids)
    links["speech_text"] = links["speech_id"].map(texts).fillna("")

    agg = links.groupby("bill_id").agg(
        speeches_combined=("speech_text", lambda s: " ".join(s)),
        num_speakers=("speech_id", "nunique"),
        speakers_parties=("party", lambda s: "|".join(s.astype(str))),
    ).reset_index()

    def majority_party(parties_str):
        parties = [p for p in parties_str.split("|") if p and p != "Unknown"]
        if not parties:
            return "Unknown"
        from collections import Counter
        return Counter(parties).most_common(1)[0][0]

    agg["majority_party"] = agg["speakers_parties"].apply(majority_party)

    bills_c = bills_df[bills_df["congress"] == congress]
    merged = bills_c.merge(agg, on="bill_id", how="inner")
    print(f"  Congress {congress}: {len(merged)} bills with linked speech text")
    return merged


def main(congresses: list):
    bills_df = pd.read_csv(BILLS_PATH)
    all_parts = []
    for congress in congresses:
        part = build_for_congress(congress, bills_df)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        print("No bills with linked speeches found. Aborting.")
        return

    merged = pd.concat(all_parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["bill_id"])
    merged = merged.dropna(subset=["speeches_combined", "title"])
    merged = merged[merged["speeches_combined"].str.strip() != ""]

    merged["speeches_combined"] = merged["speeches_combined"].str.lower()
    merged["speech_length"] = merged["speeches_combined"].str.len()
    merged["speech_word_count"] = merged["speeches_combined"].str.split().str.len()
    merged["has_economic_subject"] = True

    os.makedirs(os.path.dirname(MERGED_OUT), exist_ok=True)
    merged.to_csv(MERGED_OUT, index=False)
    print(f"\nSaved merged dataset: {MERGED_OUT}")
    print(f"  Total bills: {len(merged)}")
    print(f"  Pass rate: {merged['passed'].mean():.1%}  ({merged['passed'].sum()} passed)")
    print(merged.groupby("congress")["passed"].agg(["count", "mean"]))

    # Preprocessed version (mirrors notebook 02 output)
    from src.nlp_utils import extract_text_features
    preprocessed = extract_text_features(merged, text_column="speeches_combined")
    preprocessed.to_csv(PREPROCESSED_OUT, index=False)
    print(f"\nSaved preprocessed dataset: {PREPROCESSED_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congresses", type=str, default="110,111,112,113,114")
    args = parser.parse_args()
    congress_list = [int(c.strip()) for c in args.congresses.split(",")]
    main(congress_list)
