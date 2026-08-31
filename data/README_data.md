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

### Automatic Download (via Notebook 01)

The data collection notebook (`notebooks/01_data_collection.ipynb`) will automatically:
- Fetch bill metadata for Congress 110–114
- Filter to economic bill subjects
- Determine pass/fail status from `latestAction` field
- Save to `data/processed/bills_metadata.csv`

### Manual Download (if needed)

Use the Congress.gov API browser: https://api.congress.gov/v3/bill/110

Key endpoints:
- `/v3/bill/{congress}/{billType}` - Get bills for a Congress
- Fields: `congress`, `number`, `type`, `title`, `summaries`, `subjects`, `latestAction`

**Pass/Fail Definition:**
- **PASSED (Y=1):** `latestAction.text` contains "Became Public Law"
- **FAILED (Y=0):** All others after session end

---

## 2. Congressional Speeches (Stanford Congressional Record)

### Download

1. **Visit Stanford Dataset:**
   - https://data.stanford.edu/congress_text
   - Data available for Congress 43–114

2. **Select 110th–114th Congress:**
   - Download JSON for each Congress
   - Files will be named: `congress_110.json`, `congress_111.json`, etc.

3. **Save to `data/raw/`:**
   ```bash
   # Example directory structure:
   data/raw/
   ├── congress_110_speeches.json
   ├── congress_111_speeches.json
   ├── congress_112_speeches.json
   ├── congress_113_speeches.json
   └── congress_114_speeches.json
   ```

### Data Format (Stanford)

Each speech record contains:
```json
{
  "speaker": {
    "name": "John Smith",
    "party": "D",
    "state": "CA",
    "bioguide_id": "S000001"
  },
  "date": "2007-01-15",
  "text": "Mr. Speaker, I rise today to speak about...",
  "bill_id": "HR123-110"  // May be empty for non-bill speeches
}
```

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

## 6. Size Expectations

**Target dataset after filtering:**
- ~2,000–5,000 bills
- ~40–60% pass rate (depends on Congress)
- 110th Congress (2007–2008): 2,000+ bills
- 114th Congress (2015–2016): 2,000+ bills

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
