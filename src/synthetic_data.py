"""
Synthetic Congressional bill-speech dataset generator for pipeline testing.

Generates realistic-looking bill/speech data that exercises the full NLP pipeline.
Run directly to create data/processed/bills_speeches_merged.csv.
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── vocabulary pools ──────────────────────────────────────────────────────────
# Large vocabularies ensure TF-IDF has sufficient vocabulary diversity
# (avoids max_df pruning due to noise bleeding into too many documents).

PASSED_WORDS = [
    "bipartisan", "unanimous", "compromise", "cooperative", "constructive",
    "affordable", "productive", "consensus", "prosperity", "efficient",
    "competitive", "innovation", "workforce", "strengthened", "beneficial",
    "streamlined", "targeted", "collaborative", "inclusive", "empowering",
    "sustainable", "resilient", "transparent", "accountable", "equitable",
    "leveraging", "optimized", "modernizing", "pioneering", "unifying",
    "endorsed", "praised", "celebrated", "championed", "welcomed",
    "rectified", "bolstered", "expanded", "secured", "enacted",
    "ratified", "approved", "passed", "finalized", "concluded",
]

FAILED_WORDS = [
    "controversial", "partisan", "deficit", "costly", "objectionable",
    "reckless", "irresponsible", "wasteful", "divisive", "unacceptable",
    "misguided", "problematic", "unsustainable", "unconstitutional", "burdensome",
    "overreaching", "shortsighted", "unworkable", "inflationary", "regressive",
    "overreaching", "overcomplicated", "excessive", "imbalanced", "incompatible",
    "ineffective", "unpopular", "unfunded", "undermining", "penalizing",
    "destabilizing", "obstructed", "tabled", "shelved", "delayed",
    "rejected", "vetoed", "killed", "blocked", "stalled",
    "withdrawn", "abandoned", "postponed", "defeated", "failed",
]

# Large neutral pool — words appear randomly across ALL docs, so they will be
# filtered by max_df=0.95; only signal words survive after pruning.
_NEUTRAL_POOL_A = [
    "tax", "budget", "spending", "revenue", "trade", "tariff", "wage",
    "labor", "employment", "appropriation", "federal", "congress",
    "committee", "legislation", "policy", "economic", "fiscal",
    "financial", "market", "industry", "business", "commerce",
    "regulation", "program", "fund", "allocation", "provision",
    "benefit", "cost", "rate", "income", "corporation", "sector",
    "amendment", "reform", "proposal", "measure", "act", "bill",
    "hearing", "session", "vote", "chamber", "senate", "house",
    "representative", "senator", "speaker", "majority", "minority",
    "conference", "committee", "caucus", "leadership", "administration",
]

_NEUTRAL_POOL_B = [
    "investment", "opportunity", "jobs", "growth", "support",
    "improve", "strengthen", "concerns", "important", "significant",
    "believe", "think", "consider", "review", "examine",
    "address", "ensure", "provide", "require", "establish",
    "create", "maintain", "protect", "increase", "reduce",
    "implement", "develop", "monitor", "evaluate", "assess",
    "workers", "families", "businesses", "Americans", "citizens",
    "communities", "states", "districts", "agencies", "departments",
    "services", "resources", "funding", "support", "assistance",
]

NEUTRAL_ECONOMIC_WORDS = _NEUTRAL_POOL_A + _NEUTRAL_POOL_B

ECONOMIC_SUBJECTS = [
    "Taxation",
    "Labor and Employment",
    "Trade",
    "Economics and Public Finance",
    "Budget and Appropriations",
]

PARTIES = ["D", "R", "I"]
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
BILL_TYPES = ["HR", "S"]
CONGRESSES = [110, 111, 112, 113, 114]

PASS_ACTIONS = [
    "Became Public Law No. {n}.",
    "Signed by the President. Became Public Law.",
    "Enacted into law.",
]

FAIL_ACTIONS = [
    "Referred to committee. No further action.",
    "Failed passage in the House.",
    "Indefinitely postponed.",
    "Vetoed by the President.",
    "Died in committee.",
    "Failed cloture vote in the Senate.",
]


# ── text generation ───────────────────────────────────────────────────────────

def _build_speech(passed: bool, n_words: int = 400) -> str:
    """
    Generate a synthetic floor speech.

    Design: neutral filler dominates (high max_df → pruned by TF-IDF);
    signal words are sparse enough to survive max_df=0.95 filtering yet
    appear in enough docs to survive min_df=5.

    Word mix:
    - 70% neutral filler (will exceed max_df → pruned → invisible to models)
    - 25% outcome-specific signal  (survive max_df, carry predictive info)
    -  5% cross-class noise         (kept very small to avoid signal bleed)
    """
    words = []

    filler_count = int(n_words * 0.70)
    signal_count = int(n_words * 0.25)
    noise_count  = n_words - filler_count - signal_count  # ~5%

    # Filler: drawn from the large neutral pool
    words += random.choices(NEUTRAL_ECONOMIC_WORDS, k=filler_count)

    # Signal: drawn from the outcome-specific pool
    if passed:
        words += random.choices(PASSED_WORDS, k=signal_count)
        words += random.choices(FAILED_WORDS, k=noise_count)
    else:
        words += random.choices(FAILED_WORDS, k=signal_count)
        words += random.choices(PASSED_WORDS, k=noise_count)

    random.shuffle(words)
    return " ".join(words)


def _build_title(bill_type: str, bill_number: int, congress: int) -> str:
    topics = [
        "Tax Relief Act",
        "Jobs and Economic Growth Act",
        "Trade Adjustment Act",
        "Budget Reform and Accountability Act",
        "Workforce Investment Act",
        "Small Business Credit Act",
        "Energy and Commerce Improvement Act",
        "Federal Spending Reduction Act",
        "Export Promotion Act",
        "Fiscal Responsibility Act",
    ]
    return f"{random.choice(topics)} of {2005 + congress}"


def _latest_action(passed: bool, n: int) -> str:
    if passed:
        return random.choice(PASS_ACTIONS).format(n=n)
    return random.choice(FAIL_ACTIONS)


# ── main generator ────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_bills: int = 2000,
    pass_rate: float = 0.45,
    speeches_per_bill_mean: float = 3.0,
    output_path: str = "data/processed/bills_speeches_merged.csv",
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic Congressional bill + speech dataset.

    Args:
        n_bills: Total number of bills to generate
        pass_rate: Fraction of bills that pass
        speeches_per_bill_mean: Average number of speeches per bill
        output_path: Where to save the CSV (None to skip)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with one row per bill, speeches aggregated
    """
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    n_passed = int(n_bills * pass_rate)
    outcomes = [1] * n_passed + [0] * (n_bills - n_passed)
    random.shuffle(outcomes)

    for i, passed in enumerate(outcomes):
        congress = random.choice(CONGRESSES)
        bill_type = random.choice(BILL_TYPES)
        bill_number = random.randint(1, 9999)
        bill_id = f"{bill_type}{bill_number}-{congress}"

        # Generate 1–6 speeches per bill
        n_speeches = max(1, int(np.random.poisson(speeches_per_bill_mean)))
        speeches = [_build_speech(bool(passed)) for _ in range(n_speeches)]
        speeches_combined = " ".join(speeches)

        # Speaker metadata
        speaker_parties = [random.choice(PARTIES) for _ in range(n_speeches)]
        majority_party = max(set(speaker_parties), key=speaker_parties.count)

        rows.append({
            "bill_id": bill_id,
            "congress": congress,
            "bill_type": bill_type,
            "bill_number": bill_number,
            "title": _build_title(bill_type, bill_number, congress),
            "summary": f"A bill to {random.choice(NEUTRAL_ECONOMIC_WORDS)} the national {random.choice(NEUTRAL_ECONOMIC_WORDS)} policy.",
            "status": _latest_action(bool(passed), n=100 + i),
            "status_date": f"20{7 + (congress - 110) * 2:02d}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "passed": passed,
            "subjects": random.sample(ECONOMIC_SUBJECTS, k=random.randint(1, 3)),
            "has_economic_subject": True,
            "speeches_combined": speeches_combined,
            "speakers_names": "|".join([f"Speaker{j}" for j in range(n_speeches)]),
            "speakers_parties": "|".join(speaker_parties),
            "speech_length": len(speeches_combined),
            "speech_word_count": len(speeches_combined.split()),
            "num_speakers": n_speeches,
            "majority_party": majority_party,
        })

    df = pd.DataFrame(rows)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic dataset saved: {output_path}")
        print(f"  Bills: {len(df)}")
        print(f"  Pass rate: {df['passed'].mean():.1%}")
        print(f"  Avg speech words: {df['speech_word_count'].mean():.0f}")

    return df


def generate_preprocessed_dataset(
    n_bills: int = 2000,
    output_path: str = "data/processed/bills_speeches_preprocessed.csv",
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate preprocessed dataset (includes text_length, word_count, etc.)
    Mirrors the output of notebook 02.
    """
    from src.nlp_utils import extract_text_features, filter_economic_speeches

    df = generate_synthetic_dataset(n_bills=n_bills, output_path=None, seed=seed)
    df = filter_economic_speeches(df)
    df = extract_text_features(df, text_column="speeches_combined")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Preprocessed dataset saved: {output_path}")

    return df


if __name__ == "__main__":
    print("Generating synthetic Congressional bill-speech dataset...")
    df_raw = generate_synthetic_dataset(
        n_bills=2000,
        output_path="data/processed/bills_speeches_merged.csv",
    )
    df_pre = generate_preprocessed_dataset(
        n_bills=2000,
        output_path="data/processed/bills_speeches_preprocessed.csv",
    )
    print("\nDone. Run notebooks 03–07 to train models.")
