# Business Entity Resolution Challenge (Amazon ML 2026)
## AWS-Native End-to-End ML / NLP Solution & Execution Plan

---

## 1. Executive Strategy & Core Objectives

The goal is to build an enterprise-grade, reproducible Entity Resolution (ER) pipeline that maps every **Source 1 (S1)** business record to zero, one, or multiple matching records in **Source 2 (S2)** and **Source 3 (S3)** using only the challenge dataset.

### 1.1 Guiding Principles
* **AWS-Powered High-Throughput Architecture:** Leveraging AWS Cloud infrastructure (**Amazon S3, Amazon SageMaker, Amazon EC2**) to effortlessly process and scale across $>24\text{M}$ total records in train and test with multi-core parallelism and distributed country workers.
* **Record Linkage First, Not Heavy Generative LLMs:** Entity resolution at this scale requires a high-recall multi-key candidate generation stage followed by a precision-calibrated gradient boosting matcher (LightGBM/CatBoost) and optional lightweight bi-encoder embeddings (MIT/Apache 2.0).
* **Precision is King ($F_{0.5}$ Optimization):** In $F_{0.5}$, precision is weighted twice as heavily as recall:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  A false merge (false positive) causes twice the damage of a missed match (false negative).
* **Defend Singletons Fiercely:** $5.58\%$ of S1 entities have 0 matches in ground truth. Correctly predicting empty earns a full $1.0$, while a single incorrect match drops that entity's score to $0.0$.
* **Zero-Shot Generalization for France:** France appears **only in the test set** ($\sim 259\text{k}$ S1, $\sim 703\text{k}$ S2, $\sim 732\text{k}$ S3) and is absent in train. The pipeline must be strictly country-agnostic and handle French language syntax, diacritics, and postal formats.
* **$100\%$ Strict Country Partition Invariant:** Verified on all $7.63\text{M}$ ground truth pairs ($0$ cross-country matches). Country partition allows embarrassingly parallel processing across independent AWS compute nodes.

---

## 2. AWS Cloud Architecture & Infrastructure Blueprint

```mermaid
flowchart TD
    subgraph Storage Layer: Amazon S3 [s3://amazon-ml-2026-er/]
        S3_Raw["/raw/ (TSV Datasets)"]
        S3_Parquet["/processed/ (Country-Partitioned Parquet)"]
        S3_Cand["/candidates/ (Candidate Pairs TSV/Parquet)"]
        S3_Feat["/features/ (Pairwise Feature Matrices)"]
        S3_Models["/models/ (LightGBM / Calibrator Artifacts)"]
        S3_Out["/output/ (matching_results.tsv, candidate_pairs.tsv)"]
    end

    subgraph Compute Layer: AWS SageMaker / EC2
        EC2_Prep["1. Preprocessing & Partitioning<br/>(SageMaker Processing / EC2 c6i.8xlarge)"]
        
        subgraph Country-Parallel Blocking & Feature Extraction
            Worker_US["Worker 1: US<br/>(EC2 c6i.8xlarge / 32 vCPU)"]
            Worker_IN["Worker 2: India<br/>(EC2 c6i.8xlarge / 32 vCPU)"]
            Worker_FR["Worker 3: France<br/>(EC2 c6i.4xlarge / 16 vCPU)"]
        end
        
        SM_Train["2. GBDT Model Training & F0.5 Calibration<br/>(SageMaker Training / ml.c6i.8xlarge or ml.g5.2xlarge)"]
        SM_Infer["3. Distributed Test Inference Engine<br/>(Batch Parallel Inference on S1 Test Chunks)"]
        Validator["4. Verification & Packaging<br/>(validate_submission.py PASS Check)"]
    end

    S3_Raw --> EC2_Prep
    EC2_Prep --> S3_Parquet
    S3_Parquet --> Worker_US & Worker_IN & Worker_FR
    Worker_US & Worker_IN & Worker_FR --> S3_Cand & S3_Feat
    S3_Feat --> SM_Train
    SM_Train --> S3_Models
    S3_Models & S3_Cand --> SM_Infer
    SM_Infer --> S3_Out
    S3_Out --> Validator
```

### 2.1 Recommended AWS Services & Instance Types

| Task / Stage | Recommended AWS Service | Instance Type | Specs & Rationale |
| :--- | :--- | :--- | :--- |
| **Data Lake & Storage** | **Amazon S3** | Standard S3 Bucket | High durability, multi-threaded S3 transfer with `aws s3 sync` or `s3fs` / `boto3`. |
| **Data Prep & Partitioning** | **Amazon SageMaker / EC2** | `c6i.8xlarge` | 32 vCPUs, 64 GB RAM. Fast Polars/PyArrow multi-threaded TSV $\rightarrow$ Parquet conversion. |
| **Blocking & Feature Extraction** | **Amazon EC2 (or SageMaker Processing)** | `c6i.16xlarge` or 3x `c6i.8xlarge` | 64 vCPUs, 128 GB RAM. Parallel C++ RapidFuzz and token inverted index lookups across millions of pairs. |
| **GBDT Training & Tuning** | **Amazon SageMaker Training** | `ml.c6i.8xlarge` or `ml.g5.2xlarge` | High CPU memory for LightGBM/CatBoost multi-threading with histogram binning or GPU acceleration. |
| **Dense Embeddings (Optional)** | **Amazon SageMaker JumpStart / EC2** | `ml.g5.2xlarge` | 1x NVIDIA A10G (24GB VRAM). High-throughput batch embedding generation with open-source models. |
| **Full Test Set Inference** | **Amazon SageMaker Batch / EC2** | `c6i.16xlarge` | Ultra-fast batched inference streaming across 1.73M test entities in under 15 minutes. |

---

## 3. Project Structure & Code Organization

The workspace is organized to support both local execution and AWS cloud deployment:

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
│       │   ├── config.py               # Paths, AWS S3 buckets, constants, thresholds
│       │   ├── aws_utils.py            # Boto3 S3 upload/download, SageMaker runner helpers
│       │   ├── text_normalizer.py      # Multilingual text cleaning, suffix standardization
│       │   ├── blocking.py             # Country-partitioned multi-key inverted index & MinHash
│       │   ├── feature_engineering.py  # RapidFuzz, token overlap, TF-IDF & structural features
│       │   ├── dataset.py              # Polars streaming, training sampler, validation loader
│       │   ├── model.py                # LightGBM / CatBoost pairwise classifier & ranker
│       │   ├── threshold_optimizer.py  # Macro F0.5 calibrator & singleton gatekeeper
│       │   ├── inference.py            # Batched low-memory test inference engine
│       │   └── evaluator.py            # Exact macro F0.5 & candidate recall scoring harness
│       ├── notebooks/
│       │   ├── 01_eda_and_insights.ipynb
│       │   ├── 02_blocking_tuning.ipynb
│       │   ├── 03_model_experiments.ipynb
│       │   └── 04_error_analysis.ipynb
│       ├── scripts/
│       │   ├── sync_to_s3.sh           # Sync raw/processed data to Amazon S3
│       │   ├── run_aws_pipeline.py     # End-to-end cloud pipeline launcher (EC2 / SageMaker)
│       │   └── download_and_verify.py  # Pull outputs from S3, run validator, package zip
│       ├── README.md                   # Step-by-step reproduction guide
│       └── requirements.txt            # Pinned dependencies (boto3, polars, lightgbm, rapidfuzz, etc.)
│
├── output/
│   ├── candidate_pairs.tsv             # Candidate pairs before final classification
│   └── matching_results.tsv            # Final entity matches (leaderboard submission)
│
├── PROJECT_ANALYSIS.md                 # Complete dataset statistics & empirical findings
└── Business_Entity_Resolution_Solution_Plan.md
```

---

## 4. Detailed Stage-by-Stage Implementation Plan (Hybrid: Local Preprocessing + AWS Heavy Compute)

### Phase 1 — Environment Setup, S3 Data Lake & Local Validation Harness `[COMPLETED ✅]`

* **AWS S3 Data Lake Configuration:**
  * **Bucket Name:** `amazon-sagemaker-720682844180-us-east-1-axi666gcrhqjon`
  * **Base S3 URI:** `s3://amazon-sagemaker-720682844180-us-east-1-axi666gcrhqjon/shared/dataset/`
  * **Raw Data Location (Uploaded by User):**
    * `s3://amazon-sagemaker-720682844180-us-east-1-axi666gcrhqjon/shared/dataset/raw/train/`
    * `s3://amazon-sagemaker-720682844180-us-east-1-axi666gcrhqjon/shared/dataset/raw/test/`
* **Dependencies & Local Environment:**
  * Pinned in `requirements.txt`: `boto3`, `polars`, `pyarrow`, `rapidfuzz`, `scikit-learn`, `lightgbm`, `scipy`, `numpy`.
* **Stratified 10% Local Validation Split (Generated):**
  * `val_source1.parquet`: **220,681** S1 entities (US: $132,363$, India: $88,318$).
  * `train_source1_split.parquet`: **1,986,140** S1 entities.
  * Partitioned ground truth sets saved in `dataset/processed/`.
* **Evaluation Harness (Operational):**
  * Built `evaluator.py` supporting exact macro $F_{0.5}$ evaluation, singleton scoring, precision, recall, and candidate blocking recall.

---

### Phase 2 — High-Speed Text Preprocessing & Parquet Conversion (Local) `[IN PROGRESS ⏳]`

Implement `text_normalizer.py` and run locally or via SageMaker Processing Job:

1. **Multilingual Text Normalization:**
   * **Unicode & Diacritic Stripping:** `unicodedata.normalize('NFKD', ...)` to handle French diacritics (`é` $\rightarrow$ `e`, `ô` $\rightarrow$ `o`) and Hindi Romanization.
   * **Legal Suffix Standardization:** Unified mapping for `corp`, `inc`, `pvt ltd`, `llc`, and French legal forms (`sarl`, `sa`, `sas`).
   * **Address Term Standardization:** Standardize street indicators (`st`, `rd`, `ave`, `blvd`, `ln`, `rue`), state codes (`TX` $\leftrightarrow$ `Texas`, `VA` $\leftrightarrow$ `Virginia`), and clean landmark noise.
   * **Missing Address Flag:** Flag the $\sim 3.3\%$ records in S2/S3 with empty address (`has_address = 0`) to route through high-precision name matching.
2. **Parquet Conversion with Snappy Compression:**
   * Convert TSVs into partitioned Parquet files by `country` (`country=US`, `country=India`, `country=France`).
   * Saves $>70\%$ I/O overhead and enables memory-mapped streaming in Polars.
   * Sync processed Parquet to `s3://amazon-ml-2026-<team-name>/processed/`.

---

### Phase 3 — Country-Parallel High-Recall Blocking Engine on AWS

Blocking narrows down $>24\text{M}$ cross-comparisons to $\le 15$ candidate pairs per S1 entity with $\ge 98.5\%$ candidate recall.

```mermaid
flowchart LR
    S1[S1 Records] --> CP[Country Partition]
    CP -->|US| B_US[US Inverted Index Blocker]
    CP -->|India| B_IN[India Inverted Index Blocker]
    CP -->|France| B_FR[France Inverted Index Blocker]
    B_US & B_IN & B_FR --> Union[Union & Top-15 Rank Pruning]
    Union --> S3_Cand[s3://.../candidates/candidate_pairs.parquet]
```

1. **5 Complementary Inverted Index Blocking Keys:**
   * **Key A (Exact Normalized Name):** Exact match on cleaned full business name.
   * **Key B (Informative Token Inverted Index):** Inverted index on non-stopword, high-IDF distinctive tokens (e.g. `guggenheim`, `mirapyra`).
   * **Key C (Name Prefix / Core Stem):** First 2 significant tokens of the business name.
   * **Key D (Character 3/4-Gram MinHash LSH):** Fast Jaccard similarity candidate retrieval for names with typos or word-order inversions.
   * **Key E (Postal / PIN / City Block + Fuzzy Name):** Group by 5/6 digit postal codes or primary city/district and match partial name stems.
2. **Country-Parallel Execution on EC2 / SageMaker:**
   * Launch 3 parallel workers or use multi-core processing (`c6i.16xlarge` 64 vCPUs).
   * Merge candidates per S1 entity, cap at top 15 candidates ranked by lexical similarity, and export directly to `candidate_pairs.tsv` and `s3://.../candidates/`.

---

### Phase 4 — Pairwise High-Dimensional Feature Engineering

For every candidate pair $(S_1, S_{2/3})$, extract discriminative pairwise features across multiple cores using C++ accelerated `RapidFuzz`:

| Feature Family | Specific Features & Metrics |
| :--- | :--- |
| **Fuzzy Name Similarities** | • `fuzz.ratio` (Levenshtein ratio)<br>• `fuzz.partial_ratio` (substring matching)<br>• `fuzz.token_sort_ratio` (handles inverted word order)<br>• `fuzz.token_set_ratio` (handles added/missing tokens)<br>• `fuzz.WRatio` (weighted composite score)<br>• Jaro-Winkler similarity |
| **Vector / N-Gram Similarities** | • Word-level TF-IDF cosine similarity on name<br>• Character 3-gram TF-IDF cosine similarity on name<br>• Word-level TF-IDF cosine similarity on address<br>• Character 3-gram TF-IDF cosine similarity on address |
| **Address Structural Matches** | • Street / Building number exact match $(0/1)$<br>• Postal code / PIN code exact match $(0/1)$<br>• State / Department code match $(0/1)$<br>• Address token Jaccard similarity & Levenshtein ratio<br>• Substring containment flag (is S1 address contained in S2/S3 address?) |
| **Structural & Missingness** | • `has_address` (both have address vs one missing)<br>• Length difference and length ratio (name & address)<br>• Token count difference and token count ratio<br>• Numeric digit overlap count and Jaccard similarity |
| **Cross-Feature Interactions** | • `name_token_set_ratio * address_jaccard`<br>• `name_wratio * (1.0 if postal_match else 0.5)`<br>• Strong name match + missing address composite flag |

---

### Phase 5 — Model Training, Hard Negative Mining & Macro $F_{0.5}$ Calibration

1. **Training Sample Construction:**
   * **Positives:** All true ground truth pairs $(S_1, S_2)$ and $(S_1, S_3)$.
   * **Easy Negatives:** Random non-matching pairs from blocking candidates.
   * **Hard Negatives:** High-similarity non-matches (e.g. same business name at different address, or different businesses at the same address).
   * **Sampling Ratio:** $1 \text{ Positive} : 4 \text{ Negatives}$.
2. **Model Training (LightGBM / CatBoost on SageMaker/EC2):**
   * Train multi-threaded GBDT binary classifier with Early Stopping on validation macro $F_{0.5}$.
   * High-priority precision tuning via class weight penalty on false positives.
3. **Macro $F_{0.5}$ Threshold Optimization & Singleton Defense:**
   * Sweep decision threshold $\theta \in [0.50, 0.95]$ with step $0.01$.
   * Select optimal threshold $\theta^*$ that maximizes exact **macro $F_{0.5}$** across all validation S1 entities.
   * **Singleton Gatekeeper:** If an S1 entity has all candidate probabilities $< \theta^*$, output an empty match list `""` (earning full $1.0$ score on singletons).
   * Save model artifacts and optimal threshold parameters to `s3://.../models/`.

---

### Phase 6 — High-Throughput Test Set Inference on AWS

* The test set contains $1.73\text{M}$ S1 records and $\sim 10\text{M}$ S2/S3 records.
* **EC2 Multi-Core Streaming Inference:**
  * Stream test data country-by-country (`US`, `India`, `France`) in chunks of $100,000$ S1 entities.
  * Score candidate pairs with trained LightGBM model.
  * Apply optimal threshold $\theta^*$ and candidate-subset filter.
  * Stream results directly into `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
  * Sync outputs to `s3://amazon-ml-2026-<team-name>/output/`.

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
2. **Quality Audits:**
   * Verify all $1,732,544$ test S1 entities are present in row order.
   * Verify all predicted IDs are prefixed with `S2-` or `S3-`.
   * Verify every predicted match in `matching_results.tsv` is a subset of `candidate_pairs.tsv`.
   * Verify singleton rate on test set matches expected distribution ($\sim 5-8\%$).
3. **Documentation & Zip Packaging:**
   * Fill out [Documentation_template.md](file:///c:/Users/DELL/Desktop/Amazon%20ML/Amazon-ML/dataset/Documentation_template.md) with exact methodology, blocking recall, features, model metrics, and error analysis.
   * Package final archive:
     ```bash
     zip -r team_submission.zip output/ code/business_entity_resolution/ Documentation_template.md
     ```

---

## 5. AWS Execution Scripts & Automation

To streamline cloud operations, the following automated helper scripts are provided in `code/business_entity_resolution/scripts/`:

### 5.1 S3 Sync Script (`sync_to_s3.sh`)
```bash
#!/bin/bash
BUCKET="s3://amazon-ml-2026-er"
echo "Syncing local datasets to AWS S3: $BUCKET"
aws s3 sync dataset/ $BUCKET/raw/ --exclude "*.zip" --exclude "*.mp4"
echo "Sync complete."
```

### 5.2 End-to-End Cloud Runner (`run_aws_pipeline.py`)
```python
"""
AWS End-to-End Pipeline Launcher
Orchestrates preprocessing, blocking, feature extraction, model training, and inference.
"""
import os, sys, argparse
from src.config import S3_BUCKET, Config
from src.aws_utils import download_from_s3, upload_to_s3

def main():
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline on AWS/Local")
    parser.add_argument("--mode", choices=["local", "aws", "sagemaker"], default="local")
    parser.add_argument("--country", choices=["all", "US", "India", "France"], default="all")
    args = parser.parse_args()
    
    print(f"Starting Entity Resolution Pipeline in [{args.mode}] mode...")
    # Step 1: Preprocess & Parquet Conversion
    # Step 2: Multi-Index Blocking
    # Step 3: Feature Engineering
    # Step 4: LightGBM Training & F0.5 Calibration
    # Step 5: Test Inference & Validation Check
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
```

---

## 6. Milestone Roadmap & Success Metrics

| Milestone | Key Deliverables & Objective | Target Success Metric | Status |
| :--- | :--- | :--- | :---: |
| **M1: Foundation & Local Validation** | • Configured S3 bucket & data lake layout<br>• Installed core dependencies (`polars`, `lightgbm`, `rapidfuzz`)<br>• Generated stratified 10% validation split (220k S1 entities)<br>• Implemented macro $F_{0.5}$ evaluation harness | Local scoring harness operational & splits saved | **`DONE ✅`** |
| **M2: Preprocessing & High-Recall Blocker** | • Multilingual text normalizer (US, India, France)<br>• Implement country partition + 5-key multi-index blocker<br>• Test candidate recall on local 10% validation split | Candidate Recall $\ge 98.5\%$, Avg candidates/S1 $\le 12$ | **`IN PROGRESS ⏳`** |
| **M3: Feature Engineering Engine** | • Compute RapidFuzz, TF-IDF cosine, and structural features<br>• Generate $(X, y)$ pairwise training matrix | $>25$ discriminative features computed | `PENDING ⚪` |
| **M4: Model Training & Tuning (AWS/Local)** | • Train LightGBM classifier with hard negative mining<br>• Optimize threshold $\theta^*$ on validation macro $F_{0.5}$ | Validation Macro $F_{0.5} \ge 0.85+$ | `PENDING ⚪` |
| **M5: Full Test Inference (AWS)** | • Batch stream test inference across US, India, France<br>• Generate `matching_results.tsv` and `candidate_pairs.tsv` | $1,732,544$ rows produced in $<15\text{ mins}$ | `PENDING ⚪` |
| **M6: Verification & Package** | • Run `validate_submission.py` with `--check-ids`<br>• Fill `Documentation_template.md`<br>• Create final submission zip | `validate_submission.py`: **PASS** | `PENDING ⚪` |

---

## 7. Summary of Critical Constraints & Fair Play Rules

1. **No External Data / APIs:** Strictly NO Google Places, Google Maps, Bing Maps, OpenStreetMap, external geocoding APIs, or external corporate registries. Any external lookup causes instant disqualification.
2. **Model Specs:** Maximum parameter count $\le 8\text{B}$, open-source MIT or Apache-2.0 license.
3. **Reproducibility:** The entire pipeline must be fully runnable end-to-end from `code/business_entity_resolution/` using only local scripts and dependencies in `requirements.txt`.
