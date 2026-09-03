# Data Sources and Download Instructions

## Overview

This project uses two primary data sources:
1. **Bill Metadata + Outcomes** from Congress.gov API
2. **Congressional Floor Speeches** from Stanford Congressional Record Dataset

We focus on **110th–114th Congress (2007–2016)** with **economic bill subjects** (taxation, labor, trade, budget).

---

## 1. Bill Metadata & Outcomes (Congress.gov API)

### Setup

1. **Get a free API key:**
   - Visit https://api.congress.gov
   - Register for a free account
   - Copy your API key

2. **Set environment variable:**
   ```bash
   export CONGRESS_API_KEY=<your-api-key>
   ```

3. **Or save to `.env` file** in project root:
   ```
   CONGRESS_API_KEY=<your-api-key>
   ```

### Automatic Fetch (via scripts/fetch_real_bills.py)

`scripts/fetch_real_bills.py` fetches real bill metadata and determines the
economic-policy subset in two stages, because the Congress.gov list endpoint
(`/v3/bill/{congress}/{billType}`) does **not** return `subjects` inline —
subjects require one call per bill to a separate endpoint, which is too many
calls to do for every bill in a Congress:

1. **List + title-keyword prefilter (cheap):** paginate `/v3/bill/{congress}/{billType}`
   for `hr` and `s`, keep bills whose title matches an economic-keyword regex
   (tax, tariff, trade, labor, wage, budget, appropriation, fiscal, etc.).
2. **Official policyArea confirmation (one call per candidate):** for each
   keyword-matched candidate, call `/v3/bill/{congress}/{billType}/{billNumber}/subjects`
   and keep it only if the bill's official `policyArea.name` is one of:
   Taxation, Labor and Employment, Foreign Trade and International Finance,
   Economics and Public Finance, Finance and Financial Sector, Commerce.

Requires `CONGRESS_API_KEY` (see setup above). Caches every API response to
`data/raw/bills_cache/` so re-runs are cheap and resumable. Saves to
`data/processed/bills_metadata_real.csv`.

**Pass/Fail Definition:**
- **PASSED (Y=1):** `latestAction.text` contains "Became Public Law"
- **FAILED (Y=0):** All others

---

## 2. Congressional Speeches (Stanford Congressional Record)

### Download

1. **Visit Stanford Dataset:**
   - https://data.stanford.edu/congress_text
   - Data available for Congress 43–114

2. **Select 110th–114th Congress:**
   - Download the pipe-delimited bulk files for each Congress.

3. **Save to `data/raw/`:**
   ```
   data/raw/
   ├── speeches/speeches_{congress}.txt        # speech_id|speech text
   ├── descriptions/descr_{congress}.txt       # speech_id metadata (date, chamber, etc.)
   └── speakermap/{congress}_SpeakerMap.txt    # speech_id -> speaker/party
   ```

### Data Format (Stanford) — IMPORTANT: no native bill linkage

Each line of `speeches_{congress}.txt` is `speech_id|speech_text`. **The Stanford
corpus has no `bill_id` field.** It is organized purely by date/speaker; a speech
is not tagged with which bill (if any) it discusses. Any documentation or code
that assumes a `bill_id` key in this data is describing a different, bill-linked
corpus — not what Stanford actually publishes.

This project recovers bill linkage itself via `scripts/link_speeches_to_bills.py`,
which regex-matches explicit in-text bill citations (e.g. "H.R. 1234", "S. 815")
against the confirmed economic-bill list from `scripts/fetch_real_bills.py`, and
keeps only bills with at least one linked speech. See that script's docstring for
the regex patterns and known limitations (a mention is not proof of substantive
relevance; nickname-only references like "the Recovery Act" are missed).

---

## 3. Alternative: CoCoHD Dataset (Optional)

If Stanford download is slow or unavailable:

**GitHub:** https://github.com/gtfintechlab/CoCoHD

- 32,697 hearing transcripts (1997–2024)
- Already cleaned and structured
- Can be used as supplement or alternative

---

## 4. Data Merging

Once downloaded, run `notebooks/01_data_collection.ipynb` to:

1. Load Congress.gov API bills
2. Load Stanford speeches
3. Merge on `bill_id` + `congress`
4. Filter to bills with economic subjects AND associated speeches
5. Save merged dataset: `data/processed/bills_speeches_merged.csv`

**Output Schema:**
```
bill_id          : str        (e.g., "HR123-110")
congress         : int        (110–114)
bill_type        : str        (HR, S, etc.)
bill_number      : int        
title            : str        (Bill title)
summary          : str        (First summary from API)
status           : str        (Latest action text)
status_date      : str        (ISO date)
passed           : int        (0 or 1)
speakers_names   : str        (pipe-separated)
speakers_parties : str        (pipe-separated)
speeches_combined: str        (Aggregated speech text)
has_economic_subject : bool
```

---

## 5. Data Filtering & Cleaning

The cleaning pipeline in `src/data_utils.py`:

1. **Deduplicates** by `bill_id`
2. **Removes nulls** in speech text and title
3. **Lowercases** speech text
4. **Adds features:**
   - `speech_length`: character count
   - `speech_word_count`: word count
   - `num_speakers`: number of unique speakers
   - `majority_party`: party of most speakers

---

## 6. Final Dataset (this repo's real pipeline, run 2026-09-02)

**Actual merged dataset produced by `scripts/build_merged_dataset.py`:**
- **1,120 bills** (110th–114th Congress, 2007–2016)
- **6.8% pass rate** (76 passed, 1,044 failed)
- Economic-policy subject areas only, confirmed via Congress.gov's official `policyArea`
  (Taxation, Labor and Employment, Foreign Trade and International Finance, Economics
  and Public Finance, Finance and Financial Sector, Commerce)
- Bills with ≥1 linked floor speech, where linkage = a regex-matched explicit in-text
  citation ("H.R. 1234", "S. 815") in the raw Stanford Congressional Record text —
  see `scripts/link_speeches_to_bills.py`

An earlier report draft cites 1,133 bills / 12.5% pass rate for this same scope; that
number traces to the Kaggle notebook in Appendix A, not to code in this repository.
The two samples aren't expected to match exactly: this pipeline's speech-mention
regex is a real but different (broader/noisier) linkage method than whatever
produced that number. See `report/results_real_data.tex` for the full real-data
Results section.

**Note on pass rate:** The unconditional pass rate across all 8,529 economic bills
this repo fetched from Congress.gov is ~0.9% (see `data/processed/bills_metadata_real.csv`).
Conditioning on having a linked floor speech raises it to 6.8% — bills with floor
debate are more likely to pass, the selection effect the report's Data section
describes, just at a lower conditional rate than the original 12.5% since this
linkage method is broader (picks up passing procedural mentions, not just
substantive debate).

**Data size:** ~500 MB–1.5 GB (raw speeches)

---

## 7. Troubleshooting

### Congress.gov API
- **Rate limiting?** Add `time.sleep(0.5)` between requests
- **API key invalid?** Check https://api.congress.gov/v3/bill/110 in browser

### Stanford Download
- **Large file?** Download one Congress at a time
- **Slow?** Use institutional network or VPN
- **JSON parse error?** Ensure proper UTF-8 encoding

### Data Merge
- **Few matches?** Some bills lack floor speeches (sent to committee, not debated)
- **No economic subjects?** Check subject filtering in `data_utils.py`

---

## 8. File Paths (in Notebooks)

```python
# Data loading
bills_df = pd.read_csv("../data/processed/bills_metadata.csv")
speeches_df = pd.read_csv("../data/processed/speeches.csv")
merged_df = pd.read_csv("../data/processed/bills_speeches_merged.csv")

# Note: Adjust paths based on where notebooks are run
```
