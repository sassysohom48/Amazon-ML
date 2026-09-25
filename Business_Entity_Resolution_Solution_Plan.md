# Business Entity Resolution Challenge (Amazon ML 2026)
## Comprehensive End-to-End ML / NLP Solution & Execution Plan

---

## 1. Executive Strategy & Core Objectives

The goal is to build an enterprise-grade, reproducible Entity Resolution (ER) pipeline that maps every **Source 1 (S1)** business record to zero, one, or multiple matching records in **Source 2 (S2)** and **Source 3 (S3)** using only the challenge dataset.

### 1.1 Guiding Principles
* **Record Linkage First, Not LLM Fine-Tuning:** Entity resolution at this scale ($>12\text{M}$ records across train/test) requires a fast, high-recall candidate generation stage followed by a precision-calibrated gradient boosting matcher.
* **Precision is King ($F_{0.5}$ Optimization):** In $F_{0.5}$, precision is weighted twice as heavily as recall:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  A false merge (false positive) causes twice the damage of a missed match (false negative).
* **Defend Singletons Fiercely:** $5.58\%$ of S1 entities have 0 matches in ground truth. Correctly predicting empty earns a full $1.0$, while a single incorrect match drops that entity's score to $0.0$.
* **Zero-Shot Generalization for France:** France appears **only in the test set** ($\sim 259\text{k}$ S1, $\sim 703\text{k}$ S2, $\sim 732\text{k}$ S3) and is completely absent in train. The pipeline must be strictly country-agnostic and robust to French language syntax, diacritics, and postal formats.
* **Zero Cross-Country Matching Invariant:** Verified on $7.63\text{M}$ ground truth pairs ($0$ cross-country matches). All blocking and matching are strictly partitioned by country.

---

## 2. Target Project Architecture & Directory Structure

To satisfy submission requirements and local modular execution, the workspace is organized as follows:

```
Amazon-ML/
├── dataset/
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv       # 2,206,821 rows
│   │   │   ├── train_source2.tsv       # 5,034,616 rows
│   │   │   ├── train_source3.tsv       # 5,285,603 rows
│   │   │   └── train_ground_truth.tsv  # 2,206,821 rows (7,638,365 pairs)
│   │   └── test/
│   │       ├── test_source1.tsv        # 1,732,544 rows (US, India, France)
│   │       ├── test_source2.tsv        # 4,887,273 rows
│   │       └── test_source3.tsv        # 5,082,316 rows
│   ├── utils/
│   │   └── validate_submission.py      # Format & constraint validator
│   └── Documentation_template.md       # Methodology documentation
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py               # Paths, constants, thresholds, hyperparameters
│       │   ├── text_normalizer.py      # Multilingual text cleaning, suffix standardization
│       │   ├── blocking.py             # Country-partitioned inverted index & MinHash blocker
│       │   ├── feature_engineering.py  # RapidFuzz, token overlap, TF-IDF & structural features
│       │   ├── dataset.py              # Chunked streaming, training pair sampler, validation loader
│       │   ├── model.py                # LightGBM / CatBoost pairwise classifier & ranker
│       │   ├── threshold_optimizer.py  # Macro F0.5 calibrator & singleton gatekeeper
│       │   ├── inference.py            # Batched low-memory test inference engine
│       │   └── evaluator.py            # Exact macro F0.5 & candidate recall scoring harness
│       ├── notebooks/
│       │   ├── 01_eda_and_insights.ipynb
│       │   ├── 02_blocking_tuning.ipynb
│       │   ├── 03_model_experiments.ipynb
│       │   └── 04_error_analysis.ipynb
│       ├── README.md                   # Step-by-step reproduction guide
│       └── requirements.txt            # Pinned dependencies (polars, lightgbm, rapidfuzz, etc.)
│
├── output/
│   ├── candidate_pairs.tsv             # Candidate pairs before final classification
│   └── matching_results.tsv            # Final entity matches (leaderboard submission)
│
├── PROJECT_ANALYSIS.md                 # Complete dataset statistics & empirical findings
└── Business_Entity_Resolution_Solution_Plan.md
```

---

## 3. High-Level Two-Stage System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A1[Source 1 TSV]
        A2[Source 2 TSV]
        A3[Source 3 TSV]
    end

    subgraph Stage 1: Preprocessing & High-Recall Blocking
        B1[Multilingual Text Normalization<br/>- Legal Suffix Standardizer<br/>- Diacritic / Unicode Normalizer<br/>- Street / Token Standardization]
        B2[Strict Country Partition<br/>US | India | France]
        B3[Multi-Index Candidate Blocker<br/>- Exact Normalized Name<br/>- Inverted Index on Informative Tokens<br/>- Character 3/4-Gram MinHash LSH<br/>- Core Name Stem + Postal/PIN Block]
        B4[Candidate Set: candidate_pairs.tsv<br/>Target Recall: >= 98.5%<br/>Avg Candidates/S1: <= 12]
    end

    subgraph Stage 2: Feature Extraction & ML Classification
        C1[High-Performance Feature Matrix<br/>- RapidFuzz (Token Sort/Set, Ratio, WRatio)<br/>- Character & Word TF-IDF Cosine<br/>- Address Overlap (Street, PIN, State)<br/>- Missing Address & Length Interactions]
        C2[Gradient Boosted Tree Matcher<br/>LightGBM / CatBoost Binary Classifier<br/>Optimized for Precision]
        C3[Macro F0.5 Calibration & Post-Processing<br/>- Global / Country Threshold Optimization<br/>- Singleton Gatekeeper: score < tau -> Empty<br/>- Candidate-Subset Invariant Enforcement]
    end

    subgraph Outputs
        D1[output/candidate_pairs.tsv]
        D2[output/matching_results.tsv]
        D3[validate_submission.py Audit: PASS]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2 --> B3 --> B4
    B4 --> D1
    B4 --> C1 --> C2 --> C3 --> D2
    D1 --> D3
    D2 --> D3
```

---

## 4. Detailed Stage-by-Stage Implementation Plan

### Phase 1 — Environment Setup & Local Validation Framework
* **Dependencies:** Install high-performance packages:
  * `polars` / `pyarrow` for ultra-fast, multi-threaded tab-separated file parsing and memory efficiency.
  * `rapidfuzz` for C++ accelerated Levenshtein, Jaro-Winkler, and token sort/set ratios.
  * `scikit-learn` for TF-IDF vectorization and sparse cosine operations.
  * `lightgbm` / `catboost` for fast, lightweight GBDT training and inference.
* **Validation Strategy:**
  * Sample a stratified **$10\%$ holdout ($220,682$ S1 entities)** from `train_source1.tsv` representing both US ($60\%$) and India ($40\%$).
  * Keep all associated S2/S3 records and true ground truth links intact in validation.
  * Build `evaluator.py` to calculate the exact macro-averaged $F_{0.5}$ score including singletons.

---

### Phase 2 — Multilingual Text Preprocessing & Normalization
Design `text_normalizer.py` to be robust, fast, and multilingual (covering English, Hindi transliteration, and French).

1. **Unicode & Diacritic Normalization:**
   * Apply `unicodedata.normalize('NFKD', ...)` to strip accents (`é` $\rightarrow$ `e`, `ô` $\rightarrow$ `o`) while preserving original tokens for France records.
   * Standardize special symbols (`&` $\rightarrow$ `and`, `@` $\rightarrow$ `at`, `/` and `-` $\rightarrow$ spaces).
2. **Business Legal Suffix Standardization:**
   * Normalize legal forms to unified tokens:
     * `Corporation`, `Corp.` $\rightarrow$ `corp`
     * `Incorporated`, `Inc.` $\rightarrow$ `inc`
     * `Private Limited`, `Pvt. Ltd.`, `Pvt Ltd`, `P. Ltd.` $\rightarrow$ `pvt ltd`
     * `Limited Liability Company`, `L.L.C.`, `LLC` $\rightarrow$ `llc`
     * `Société Anonyme`, `S.A.`, `SARL`, `SAS` (French legal forms) $\rightarrow$ `sarl` / `sa`
3. **Address Component Normalization:**
   * Normalize road designations: `Street` / `St` $\rightarrow$ `st`, `Road` / `Rd` $\rightarrow$ `rd`, `Avenue` / `Ave` $\rightarrow$ `ave`, `Boulevard` / `Blvd` $\rightarrow$ `blvd`, `Lane` / `Ln` $\rightarrow$ `ln`.
   * Standardize US state abbreviations (`CA` $\leftrightarrow$ `California`, `TX` $\leftrightarrow$ `Texas`, `NC` $\leftrightarrow$ `North Carolina`).
   * Clean noise tokens: `Door No`, `Flat No`, `Near`, `Opp`, `Floor`, `PMB`, `Apt`, `Unit`.
4. **Handling Missing Addresses:**
   * For the $\sim 3.3\%$ records in S2 and S3 with empty addresses, create an explicit boolean indicator `has_address = 0` and route them through high-precision name matching.

---

### Phase 3 — High-Recall Multi-Index Blocking Engine

Blocking narrows down $\sim 2.2\text{M} \times 10.3\text{M}$ pairs to a compact candidate set of $\le 15$ candidates per S1 entity with $\ge 98.5\%$ candidate recall.

1. **Step 1: Hard Partition by Country:**
   * Only compare `S1[US]` with `S2/S3[US]`, `S1[India]` with `S2/S3[India]`, and `S1[France]` with `S2/S3[France]`.
2. **Step 2: Multi-Key Inverted Index Blocking:**
   Union candidates from 5 complementary blocking keys:
   * **Key A (Exact Normalized Name):** Exact match on normalized full business name.
   * **Key B (Informative Token Inverted Index):** Inverted index on non-stopword, high-IDF name tokens (e.g. `roberts`, `guggenheim`, `mirapyra`).
   * **Key C (Name Prefix / Core Stem):** First 2 significant tokens of the business name.
   * **Key D (Character 3/4-gram MinHash LSH / TF-IDF Top-K):** Cosine/Jaccard candidate retrieval for names with typos or inversions (e.g. `Sai [Intermediates]` vs `Sai Intermediates Ltd`).
   * **Key E (Postal / PIN / City Block + Fuzzy Name):** Group by 5/6 digit postal codes or primary city/district and match partial name stems.
3. **Step 3: Union & Candidate Pruning:**
   * Combine all candidate IDs per S1 entity.
   * Cap maximum candidates per S1 entity to top 15–20 ranked by raw lexical similarity to maintain strict memory efficiency.
   * Export candidate set directly to `output/candidate_pairs.tsv`.

---

### Phase 4 — Pairwise Feature Engineering Engine

For each candidate pair $(S_1, S_{2/3})$, compute rich, highly discriminative features across name, address, and structure:

| Feature Family | Specific Features & Metrics |
| :--- | :--- |
| **Fuzzy Name Similarities** | • `fuzz.ratio` (Levenshtein similarity)<br>• `fuzz.partial_ratio` (substring matching)<br>• `fuzz.token_sort_ratio` (handles inverted word order)<br>• `fuzz.token_set_ratio` (handles added/missing tokens)<br>• `fuzz.WRatio` (weighted composite score)<br>• Jaro-Winkler distance |
| **Vector / N-Gram Similarities** | • Word-level TF-IDF cosine similarity on name<br>• Character 3-gram TF-IDF cosine similarity on name<br>• Word-level TF-IDF cosine similarity on address<br>• Character 3-gram TF-IDF cosine similarity on address |
| **Address Structural Matches** | • Street / Building number exact match $(0/1)$<br>• Postal code / PIN code exact match $(0/1)$<br>• State / Department code match $(0/1)$<br>• Address token Jaccard similarity & Levenshtein ratio<br>• Substring containment flag (is S1 address contained in S2/S3 address?) |
| **Structural & Missingness** | • `has_address` (both have address vs one missing)<br>• Length difference and length ratio (name & address)<br>• Token count difference and token count ratio<br>• Numeric digit overlap count and Jaccard similarity |
| **Cross-Feature Interactions** | • `name_token_set_ratio * address_jaccard`<br>• `name_wratio * (1.0 if postal_match else 0.5)`<br>• Strong name match + missing address composite flag |

---

### Phase 5 — Model Training, Hard Negative Mining & $F_{0.5}$ Calibration

1. **Training Data Construction:**
   * **Positive Pairs:** All true ground truth pairs $(S_1, S_2)$ and $(S_1, S_3)$.
   * **Easy Negatives:** Random non-matching pairs from blocking candidates.
   * **Hard Negatives:** Top candidate pairs from blocking that share identical names but different addresses (different branches), or share identical addresses but different names (different businesses at same building).
   * **Sampling Ratio:** $1 \text{ Positive} : 4 \text{ Negatives}$ to provide a balanced training set while reflecting real candidate distribution.

2. **Model Training (LightGBM / CatBoost):**
   * Objective: Binary cross-entropy with sample weight adjustments.
   * Tree depth: 6–8, Learning rate: 0.05, Boosting rounds: 500–1000 with early stopping on validation macro $F_{0.5}$.
   * High feature importance analysis to eliminate non-informative features.

3. **Macro $F_{0.5}$ Threshold Optimization:**
   * Compute prediction probabilities $\hat{p} \in [0, 1]$ on the validation candidate pairs.
   * Sweep decision threshold $\theta \in [0.50, 0.95]$ with step $0.01$.
   * Select $\theta^*$ that maximizes the exact **macro $F_{0.5}$** score across all S1 entities (including singletons).

4. **Singleton Gatekeeper Logic:**
   * For an entity $S_1$, if all candidate probabilities are below $\theta^*$, output an empty match list `""`.
   * For entities with multiple high-scoring candidates, select all candidate records where $\hat{p} \ge \theta^*$.

---

### Phase 6 — Batch Inference & Low-Memory Test Execution

* The test set contains $1.73\text{M}$ S1 records and $\sim 10\text{M}$ S2/S3 records.
* **Streaming / Chunked Processing:** Process test data country-by-country (`US`, `India`, `France`) and in chunks of $100,000$ S1 entities.
* **Memory Management:**
  * Free temporary arrays after each batch.
  * Use 32-bit float feature representations and compact dictionary IDs.
* **Output Generation:**
  * Stream results into `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
  * Ensure UTF-8 tab-separated formatting with no index columns.

---

### Phase 7 — Verification, Quality Audit & Submission Packaging

1. **Automated Submission Validation:**
   * Execute `utils/validate_submission.py` in strict mode:
     ```bash
     python dataset/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/dataset/test \
       --check-ids
     ```
   * Confirm exit code `0` (`PASS`).
2. **Sanity Checks & Audits:**
   * Verify all $1,732,544$ test S1 entities are present in row order.
   * Verify all predicted IDs are prefixed with `S2-` or `S3-`.
   * Verify every predicted match in `matching_results.tsv` is a subset of `candidate_pairs.tsv`.
   * Verify singleton proportion on test set matches expected distribution ($\sim 5-8\%$).
3. **Documentation & Zip Packaging:**
   * Fill out [Documentation_template.md](file:///c:/Users/DELL/Desktop/Amazon%20ML/Amazon-ML/dataset/Documentation_template.md) with exact methodology, blocking recall, features, model metrics, and error analysis.
   * Package final archive:
     ```bash
     zip -r team_submission.zip output/ code/business_entity_resolution/ Documentation_template.md
     ```

---

## 5. Experimentation & Milestone Roadmap

| Milestone | Key Deliverables & Objective | Target Success Metric |
| :--- | :--- | :--- |
| **M1: Foundation & Baseline** | • Install dependencies (`polars`, `lightgbm`, `rapidfuzz`)<br>• Build stratified 10% validation split & macro $F_{0.5}$ evaluator | Exact local scoring harness operational |
| **M2: High-Recall Blocker** | • Implement country partition + 5-key multi-index blocker<br>• Generate candidate sets for train/validation | Candidate Recall $\ge 98.5\%$, Avg candidates/S1 $\le 12$ |
| **M3: Feature Engineering** | • Implement RapidFuzz, TF-IDF cosine, and structural features<br>• Generate $(X, y)$ pairwise training matrix | $>25$ discriminative features computed |
| **M4: Model Training & Tuning** | • Train LightGBM classifier with hard negative mining<br>• Optimize threshold $\theta^*$ on validation macro $F_{0.5}$ | Validation Macro $F_{0.5} \ge 0.85+$ |
| **M5: Full Test Inference** | • Batch stream test inference across US, India, France<br>• Generate `matching_results.tsv` and `candidate_pairs.tsv` | $1,732,544$ rows produced, zero memory leaks |
| **M6: Verification & Package** | • Run `validate_submission.py`<br>• Fill `Documentation_template.md`<br>• Create final submission zip | `validate_submission.py`: **PASS** |

---

## 6. Summary of Critical Constraints & Fair Play Rules

1. **No External Data / APIs:** Strictly NO Google Places, Google Maps, Bing Maps, OpenStreetMap, geocoding APIs, or external corporate registries. Any external lookup causes instant disqualification.
2. **Model Specs:** Maximum parameter count $\le 8\text{B}$, open-source MIT or Apache-2.0 license.
3. **Reproducibility:** The entire pipeline must be fully runnable end-to-end from `code/business_entity_resolution/` using only local scripts and dependencies in `requirements.txt`.
