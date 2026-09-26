# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team EntitySync  
**Team Members:** Amazon ML Challenge 2026 Participant  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an enterprise-scale, country-partitioned Entity Resolution (ER) system designed to link 1.73M Source 1 business records across the US, India, and France to target entities in Source 2 and Source 3. Our solution couples a high-recall, 5-signal inverted index blocker with a 27-feature pairwise LightGBM gradient boosting classifier. To maximize the competition's Macro $F_{0.5}$ metric, we introduced a calibrated decision threshold ($\theta^* = 0.65$) and a singleton gatekeeper defense that protects non-matching entities from false positives.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 24.2M records revealed four key challenges:
1. **Zero Cross-Country Match Invariant:** Ground truth analysis across 7.63M pairs confirmed that 100% of matches occur within the same country partition (`US`, `India`, `France`).
2. **Heavy Address Noise & Missing Fields:** Indian addresses frequently suffer from variable spelling, missing PIN codes, and embedded landmarks, while US records exhibit standard street/suite conventions.
3. **Zero-Shot France Generalization:** France records appear exclusively in the test set (~663k S1, ~703k S2, ~732k S3) and require country-agnostic character tokenization and French postal code normalization.
4. **Extreme Metric Asymmetry (Macro $F_{0.5}$):** Precision is weighted 2× over Recall. Crucially, singletons (entities with zero matches) receive a 1.0 score if predicted empty `""`, but drop to 0.0 upon even a single false positive match.

### 2.2 Solution Strategy

**Approach Type:** Country-Partitioned Multi-Index Inverted Index Blocking + Pairwise GBDT Classifier + Singleton Defense Thresholding.  
**Core Innovation:** A 5-signal inverted index candidate generator combined with a precision-biased LightGBM classifier that evaluates RapidFuzz string distances, token set overlaps, and address interactions, gated by an optimal decision threshold ($\theta^* = 0.65$) tailored for Macro $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.96\text{M}$ search space down to $\le 35$ candidates per entity, we designed a country-partitioned multi-index blocker using 5 complementary signals:
- **Exact Normalized Name Match:** High-priority hash lookup for clean name matches.
- **2-Token Name Bigram Prefixes:** Inverted index on consecutive name bigrams.
- **IDF-Weighted Name Tokens:** Postings sorted by Inverse Document Frequency to prioritize rare, informative brand tokens while filtering stopwords.
- **IDF-Weighted Address Tokens:** Address token index with IDF weighting for building numbers and street names.
- **Postal / PIN Code Buckets:** Postal code matches conditioned on token similarity.

**Candidate Pairs Generated:** ~35 candidate pairs per Source 1 entity (~60M pairs total across test set).  
**Recall Preservation:** Empirical validation demonstrated $>98\%$ candidate recall on US records and $>83\%$ on noisy Indian records, ensuring minimal true match attrition before classification.

---

## 4. Matching Model

**Features Used (27 Pairwise Features):**
- **Fuzzy Name Metrics:** RapidFuzz Levenshtein ratio, partial ratio, token sort ratio, token set ratio, WRatio, and Jaro-Winkler similarity.
- **Length & Token Statistics:** Name length absolute difference, length ratio, token Jaccard similarity, token overlap count, token count difference.
- **Address Fuzzy & Structural:** Address Levenshtein ratio, token set ratio, partial ratio, token Jaccard similarity, token overlap count, and presence indicator flags (`has_addr_both`, `has_addr_one_missing`, `has_addr_both_missing`).
- **Postal & Digits:** Exact postal match $(0/1)$, postal presence indicators, and street/building digit overlap $(0/1)$.
- **Non-Linear Interactions:** `name_token_set_ratio * addr_jaccard`, `name_wratio * postal_match`, and `exact_name_missing_addr`.

**Model Type:** LightGBM Gradient Boosted Decision Trees (300 boosting rounds, max depth 8, 63 leaves).  
**Threshold Selection Method:** Systematic grid sweep over $\theta \in [0.50, 0.95]$ on a stratified validation set of 20,000 entities, optimizing the exact Macro $F_{0.5}$ metric. Selected $\theta^* = 0.65$, which penalizes false positives and secures 71.01% precision.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation):** **`0.6112`**
- **Macro Precision:** **`71.01%`**
- **Macro Recall:** **`46.84%`**
- **Common False Positives:** Franchises and chain stores sharing identical brand names but operating in adjacent zip codes or distinct units within the same commercial complex.
- **Common False Negatives:** Entities with extreme phonetic spelling deviations in Indian names (transliteration discrepancies) or severely truncated address strings.

---

## 6. Conclusion

Our solution achieves high-throughput, memory-efficient business entity resolution by combining country-partitioned multi-index candidate blocking with a 27-feature pairwise LightGBM model. By aligning our decision threshold ($\theta^* = 0.65$) directly with the Macro $F_{0.5}$ metric and defending singletons, the pipeline produces high-precision, competition-compliant submissions.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/config.py`: Centralized configuration and paths.
- `code/business_entity_resolution/src/blocking.py`: Multi-signal inverted index blocker.
- `code/business_entity_resolution/src/feature_engineering.py`: Vectorized RapidFuzz feature extractor.
- `code/business_entity_resolution/src/run_train_eval.py`: LightGBM training and threshold calibration runner.
- `code/business_entity_resolution/src/inference.py`: High-throughput test set inference engine.
- `code/business_entity_resolution/src/package_submission.py`: Final verification and ZIP packager.
- `dataset/utils/validate_submission.py`: Official competition validation harness.
