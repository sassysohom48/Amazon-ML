# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

This repository contains the complete, self-contained, end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task.

The system processes multi-source records across the United States, India, and France, executing:
1. **Streaming Data Ingestion & Normalization**: Converts raw TSV files into partitioned, Snappy-compressed Parquet tables.
2. **Adaptive 5-Channel Inverted Index Candidate Blocker**: Prunes the $1.78 \times 10^{13}$ pairwise space with an Entity Recall ceiling of 91.70% and Oracle $F_{0.5} = 0.8931$ at $K=50$.
3. **SIMD-Accelerated 25-Feature Engineering**: RapidFuzz C++ vectorized name, address, and topological distance metrics.
4. **Dual GBDT Ensemble (LightGBM + CatBoost)**: Trains leaf-wise and oblivious decision tree ensembles with zero entity leakage.
5. **Country-Calibrated Decision Thresholding**: Maximizes Macro $F_{0.5}$ and Singleton Accuracy per country ($\tau_{\text{US}} = 0.75$, $\tau_{\text{INDIA}} = 0.70$, $\tau_{\text{FRANCE}} = 0.71$).
6. **Checkpointed Streaming Inference**: Emits `output/matching_results.tsv` and `output/candidate_pairs.tsv` with automatic country-level checkpoint resumption.

---

## 1. Environment Setup & Requirements

The pipeline requires Python 3.9+ and standard CPU compute (no GPU required).

Install all pinned dependencies:
```bash
pip install -r requirements.txt
```

Core libraries:
* `polars>=0.20.0` & `pyarrow>=15.0.0` (zero-copy vectorized tabular processing)
* `rapidfuzz>=3.8.0` (AVX2/SIMD string distance algorithms)
* `lightgbm>=4.3.0` & `catboost>=1.2.0` (gradient boosted tree modeling)
* `scikit-learn>=1.4.0` & `scipy>=1.12.0`

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── README.md                  # This run and reproduction guide
├── requirements.txt           # Pinned Python package dependencies
├── run_step1.py               # Step 1: TSV Ingestion & Partitioning
├── run_step2.py               # Step 2: Multi-Channel Candidate Blocking
├── run_step3.py               # Step 3: SIMD 25-Feature Extraction
├── run_step4.py               # Step 4: GBDT Training & Country Thresholding
├── run_step5_inference.py     # Step 5: Full Test Streaming Inference
└── src/
    ├── config.py              # Directory paths, hyperparameters, constants
    ├── normalizer.py          # String sanitization & French accent folding
    ├── data_processor.py      # TSV streaming to partitioned Parquet
    ├── dataset.py             # Ground truth loading & verification
    ├── blocking.py            # 5-Channel Inverted Indexing & worker pooling
    ├── feature_builder.py     # 25 dense pairwise SIMD feature vectorizers
    ├── trainer.py             # LightGBM + CatBoost training & threshold sweeps
    ├── inference.py           # Checkpointed test inference & TSV emission
    └── evaluator.py           # Vectorized Macro F0.5 evaluation metrics
```

---

## 3. End-to-End Reproduction Instructions

To reproduce the final submission outputs (`matching_results.tsv` and `candidate_pairs.tsv`) from raw dataset files:

### Step 1: Raw Data Ingestion & Parquet Partitioning
Ingests raw TSVs into partitioned Snappy Parquet files under `data/processed/` and `data/parquet_partitions/`:
```bash
python run_step1.py
```

### Step 2: Multi-Channel Candidate Blocking & Evaluation
Indexes all target records (Source 2 and Source 3) into 5 inverted channels, generates candidate pairs for evaluation entities, and saves `data/processed/val_candidates.parquet`:
```bash
python run_step2.py
```

### Step 3: SIMD Pairwise Feature Engineering
Extracts 25 dense lexical, phonetic, and topological features for all candidate pairs in parallel using RapidFuzz SIMD routines (~50 seconds runtime):
```bash
python run_step3.py
```

### Step 4: GBDT Model Training & Per-Country Threshold Calibration
Trains the LightGBM classifier with early stopping, fits the CatBoost classifier, computes feature importance, performs Macro $F_{0.5}$ threshold sweeps, and saves `models/optimal_thresholds.json`:
```bash
python run_step4.py
```

### Step 5: Checkpointed Streaming Test Inference
Executes streaming multi-process inference across test partitions (US, India, France) in 100k entity batches. Applies country-calibrated thresholds and writes final TSVs into `output/`:
```bash
python run_step5_inference.py --force --workers 4
```

---

## 4. Submission Validation

Verify that generated output files are 100% compliant with competition constraints:
```bash
python ../../dataset/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
A status of `PASS` confirms valid schema, matching headers, single-row constraints, and no duplicate IDs.
