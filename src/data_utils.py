"""Data loading, cleaning, and merging utilities."""

import os
import json
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time


ECONOMIC_SUBJECTS = [
    "Taxation",
    "Labor and Employment",
    "Trade",
    "Economics and Public Finance",
    "Budget and Appropriations",
]

BILL_TYPES = ["HR", "S", "HB", "SB"]  # House and Senate bills


def get_congress_gov_api_key() -> str:
    """Get Congress.gov API key from environment variable or prompt."""
    api_key = os.environ.get("CONGRESS_API_KEY")
    if not api_key:
        raise ValueError(
            "CONGRESS_API_KEY environment variable not set. "
            "Get a free key at https://api.congress.gov"
        )
    return api_key


def fetch_bills_from_congress_gov(
    congress: int, api_key: str, delay: float = 0.5
) -> pd.DataFrame:
    """
    Fetch bill metadata from Congress.gov API for a given Congress.

    Args:
        congress: Congress number (e.g., 110 for 110th Congress)
        api_key: Congress.gov API key
        delay: Delay between requests in seconds (rate limiting)

    Returns:
        DataFrame with columns: bill_id, title, congress, bill_type,
                               status, passed, summary
    """
    base_url = "https://api.congress.gov/v3/bill"
    params = {
        "api_key": api_key,
        "limit": 250,  # Max per request
        "format": "json",
    }

    all_bills = []
    offset = 0

    print(f"Fetching bills for Congress {congress}...")

    while True:
        params["offset"] = offset
        url = f"{base_url}/{congress}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Congress.gov: {e}")
            break

        data = response.json()
        if "bills" not in data or not data["bills"]:
            break

        for bill in data["bills"]:
            bill_data = {
                "congress": bill.get("congress"),
                "bill_type": bill.get("type", "").upper(),
                "bill_number": bill.get("number"),
                "bill_id": f"{bill.get('type', '').upper()}{bill.get('number', '')}-{congress}",
                "title": bill.get("title", ""),
                "summary": bill.get("summaries", [{}])[0].get("text", "") if bill.get("summaries") else "",
                "status": bill.get("latestAction", {}).get("text", ""),
                "status_date": bill.get("latestAction", {}).get("actionDate", ""),
                "subjects": [s.get("name") for s in bill.get("subjects", [])],
            }

            # Determine if bill passed
            latest_action_text = bill.get("latestAction", {}).get("text", "").lower()
            bill_data["passed"] = 1 if "became public law" in latest_action_text else 0

            all_bills.append(bill_data)

        print(f"  Fetched {len(all_bills)} bills so far...")
        offset += 250
        time.sleep(delay)  # Rate limiting

    df = pd.DataFrame(all_bills)

    # Filter to economic subjects
    def has_economic_subject(subjects_list):
        if pd.isna(subjects_list):
            return False
        return any(subj in subjects_list for subj in ECONOMIC_SUBJECTS)

    df["has_economic_subject"] = df["subjects"].apply(has_economic_subject)
    df = df[df["has_economic_subject"]].copy()

    print(f"Filtered to {len(df)} bills with economic subjects")

    return df


def fetch_stanford_speeches(
    congress_list: List[int], cache_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Fetch Stanford Congressional Record speeches.

    Note: Stanford provides bulk download. This is a placeholder for manual download.
    Download from: https://data.stanford.edu/congress_text

    Args:
        congress_list: List of Congress numbers to include
        cache_dir: Directory to store downloaded data

    Returns:
        DataFrame with columns: congress, speaker_name, speaker_party,
                               speaker_state, speech_date, speech_text, bill_id
    """
    print("Stanford Congressional Record speeches must be downloaded manually.")
    print("Visit: https://data.stanford.edu/congress_text")
    print("Download format: JSON per Congress")
    print("")

    speeches_list = []

    for congress in congress_list:
        file_path = os.path.join(cache_dir, f"congress_{congress}_speeches.json")

        if not os.path.exists(file_path):
            print(f"WARNING: {file_path} not found.")
            print(f"  Please download from Stanford and save to {file_path}")
            continue

        print(f"Loading speeches from {file_path}...")

        try:
            with open(file_path, "r") as f:
                speeches = json.load(f)

            # Normalize to list of dicts if needed
            if isinstance(speeches, dict):
                speeches = [speeches]

            for speech in speeches:
                speech_data = {
                    "congress": congress,
                    "speaker_name": speech.get("speaker", {}).get("name", ""),
                    "speaker_party": speech.get("speaker", {}).get("party", ""),
                    "speaker_state": speech.get("speaker", {}).get("state", ""),
                    "speaker_id": speech.get("speaker", {}).get("bioguide_id", ""),
                    "speech_date": speech.get("date", ""),
                    "speech_text": speech.get("text", ""),
                    "bill_id": speech.get("bill_id", ""),
                }
                speeches_list.append(speech_data)

        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            continue

    df = pd.DataFrame(speeches_list)
    print(f"Loaded {len(df)} total speeches")

    return df


def merge_bills_and_speeches(
    bills_df: pd.DataFrame, speeches_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge bill metadata with speeches.

    Args:
        bills_df: DataFrame from fetch_bills_from_congress_gov
        speeches_df: DataFrame from fetch_stanford_speeches

    Returns:
        Merged DataFrame with one row per bill, speech text aggregated
    """
    print("Merging bills with speeches...")

    # Aggregate speeches by bill_id
    speeches_agg = speeches_df.groupby("bill_id").agg({
        "speech_text": " ".join,  # Concatenate all speeches for a bill
        "speaker_name": lambda x: "|".join(x),  # Pipe-separated speaker names
        "speaker_party": lambda x: "|".join(x),  # Pipe-separated parties
        "congress": "first",
    }).reset_index()

    speeches_agg.columns = ["bill_id", "speeches_combined", "speakers_names", "speakers_parties", "congress"]

    # Merge
    merged = bills_df.merge(speeches_agg, on=["bill_id", "congress"], how="inner")

    print(f"Merged to {len(merged)} bills with associated speeches")

    return merged


def clean_bill_speeches(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess merged data.

    Args:
        merged_df: Output from merge_bills_and_speeches

    Returns:
        Cleaned DataFrame
    """
    df = merged_df.copy()

    # Remove duplicates
    df = df.drop_duplicates(subset=["bill_id"])

    # Remove rows with missing speech text or title
    df = df.dropna(subset=["speeches_combined", "title"])

    # Clean text
    df["speeches_combined"] = df["speeches_combined"].str.lower()

    # Add features
    df["speech_length"] = df["speeches_combined"].str.len()
    df["speech_word_count"] = df["speeches_combined"].str.split().str.len()
    df["num_speakers"] = df["speakers_names"].str.split("|").str.len()

    # Determine majority speaker party
    def get_majority_party(parties_str):
        if pd.isna(parties_str):
            return "Unknown"
        parties = parties_str.split("|")
        from collections import Counter
        counts = Counter(parties)
        return counts.most_common(1)[0][0] if counts else "Unknown"

    df["majority_party"] = df["speakers_parties"].apply(get_majority_party)

    print(f"Cleaned data: {len(df)} bills after deduplication and null removal")

    return df


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


def load_processed_data(path: str) -> pd.DataFrame:
    """Load processed data from CSV."""
    return pd.read_csv(path)
