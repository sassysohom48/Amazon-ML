# Business Entity Resolution — Implementation Instructions

**Audience:** Antigravity (coding agent). Follow this file phase by phase, in order.
Do not skip the "Flaws to check" sub-section of a step — write the test it describes
*before* moving to the next step. If a test fails, apply the listed fix, re-test, then proceed.

---

## 0. Objective

Given `train_source1/2/3.tsv` (~12M rows combined) and `train_ground_truth.tsv`, build a
pipeline that, for every Source 1 entity in the test set, predicts the list of Source 2 /
Source 3 entity_ids that refer to the same real-world business. Output two files:
`matching_results.tsv` (scored) and `candidate_pairs.tsv` (diagnostic — must be the exact
input your model scored, not an earlier looser pass).

Optimized metric: macro-averaged **F_0.5** (precision weighted 2x over recall) per
Source 1 entity, including singletons (empty predicted list scores 1.0 if truly a
singleton, 0.0 if you predict anything for it).

## 1. Hard constraints (violating any of these invalidates the submission)

1. **No external lookups.** No translation APIs, no geocoding APIs, no business-registry
   lookups, no internet access from the pipeline at inference or training time. Write an
   automated test that runs the pipeline with network access disabled and asserts it still
   completes — this is your proof of compliance, not just a promise.
2. **Final model constraint:** any pretrained model used (e.g., an embedding model) must be
   MIT or Apache-2.0 licensed and ≤8B parameters. Record the exact model name + license in
   the README.
3. **Country is an open string set.** Never write code that special-cases or filters against
   a literal `{"US", "India"}` list. Test this explicitly by injecting a synthetic
   `"France"` (or any made-up) country value into a unit test and confirming the pipeline
   produces output for it with no code path failure.
4. **Output format is exact** — see `Documentation` from the problem statement. Run
   `utils/validate_submission.py` after every pipeline run, not just before final upload.
5. **Scale:** ~12M rows total. Any step written as a per-row Python loop over the full
   dataset is a bug, not a style choice — flag it in code review and vectorize/batch it.

## 2. Repository structure to create

```
code/business_entity_resolution/
├── src/
│   ├── data/            # TSV -> Parquet ingestion, profiling
│   ├── normalize/       # text cleaning, representation building
│   ├── blocking/        # MinHash, ANN index, candidate generation
│   ├── features/        # pairwise feature computation
│   ├── model/           # training, calibration, threshold tuning
│   ├── inference/        # end-to-end scoring pipeline (shared by train/val/test)
│   └── eval/            # F_0.5 scorer, blocking-recall scorer
├── tests/
│   ├── unit/            # one test file per module above
│   └── integration/     # tiny synthetic end-to-end dataset (see Phase J)
├── README.md
└── requirements.txt
```

**Rule:** `src/inference/` must be the *only* code path that both the training pipeline
(to build features for the classifier) and the final test-set scoring pipeline call. Never
let two separately-written scripts implement "the same" normalization/blocking/features —
that divergence is a top cause of silent train/test skew.

---

## PHASE A — Data Engineering Foundation

### Step A1: Convert TSV → Parquet, partitioned by `country`

- **What:** Read each source TSV once with an explicit tab separator, write to Parquet
  partitioned by the (raw, unnormalized) `country` value.
- **Why:** At 12M rows, re-parsing TSV text on every access is slow and memory-heavy.
  Parquet is columnar/compressed and lets later steps read only the columns they need.
- **Example:** `train_source2.tsv` → `source2/country=India/part.parquet`,
  `source2/country=US/part.parquet`.
- **Flaws to check / tests:**
  - The sample data showed Windows line endings (`\r\n`). A naive parse can leave a
    trailing `\r` stuck on the last column's value (e.g., `country` becomes `"US\r"`).
    **Test:** assert no cell in any column ends with `\r` or contains a raw `\n`.
    **Fix:** strip `\r`/`\n` explicitly during ingestion, don't rely on the TSV parser
    to catch it.
  - **Test:** row count after ingestion == `wc -l` of the source file minus 1 (header).
    A mismatch means silently dropped or split rows.
  - **Test:** `entity_id` is unique within each source file, and every `entity_id` in
    `source2` starts with `S2-`, in `source3` with `S3-`, etc. Catches upstream data bugs
    early instead of surfacing as confusing downstream match failures.

### Step A2: Profile the full dataset before assuming anything

- **What:** Count rows per source/country, null/empty rate of `business_address`,
  script composition of `business_name` (fraction containing Devanagari, Kannada, or
  other non-Latin Unicode ranges, and fraction containing accented Latin characters).
- **Why:** Earlier estimates were based on a 50-row sample. Real proportions at 12M rows
  could be very different and should drive resourcing (e.g., how much the cross-script
  embedding feature needs to matter).
- **Flaws to check / tests:**
  - **Test:** check whether any Source 2 or Source 3 `entity_id` appears under **more
    than one** `source1_entity_id` in `train_ground_truth.tsv`. The spec only states a
    Source 1 entity may have zero/one/many matches — it does not say a Source 2/3 record
    can't legitimately match multiple Source 1 entities. Verify empirically; if it
    happens, your train/val split and negative-sampling logic (Phase F) must account for
    it, or you'll mislabel legitimate positives as leaked negatives.
  - **Test:** check `country` string hygiene — stray whitespace, inconsistent casing
    (`" US"` vs `"US"` vs `"us"`). If found, normalize casing/whitespace on this field
    specifically (this is cleanup, not hardcoding literal country names) before using it
    as a partition/blocking key, or true matches will silently land in different
    partitions and never be compared.

---

## PHASE B — Normalization

### Step B1: Vectorized text cleaning

- **What:** Unicode NFKC normalization, lowercasing, stripping junk artifacts (leading
  `--`, trailing `| www.site.com`), punctuation harmonization (`&` → `and`), whitespace
  collapse — applied as batch/vectorized string operations (Polars expressions, or
  Dask/Spark at this scale), never a Python `for` loop over rows.
- **Why:** A per-row Python cleaning function called ~36M times (12M rows × 3 sources)
  is one of the easiest ways to make this pipeline take days instead of minutes.
- **Flaws to check / tests:**
  - **Test:** regression-test a hand-picked list of "tricky" names (e.g., a business
    legitimately named with `&`, or containing digits that look like junk) before/after
    cleaning, and manually confirm meaning wasn't destroyed by an over-aggressive regex.
  - **Test:** round-trip a sample of known multilingual strings through NFKC
    normalization and confirm character count / encoding stays sane (no mojibake
    introduced).

### Step B2: Do NOT hardcode legal-suffix stripping lists

- **What:** Instead of a literal dictionary of `{Inc, LLC, Pvt, Ltd, SARL, SAS, ...}`,
  rely on TF-IDF's IDF weighting (Phase E) to naturally down-weight these tokens since
  they recur very frequently across the corpus.
- **Why:** The spec explicitly warns against hardcoding to known countries. A suffix
  list built from US/India patterns does nothing for French `SARL`/`EURL` and would need
  constant maintenance for any other unseen country.
- **Flaws to check / tests:**
  - **Test:** inspect the IDF value of known common tokens (`ltd`, `inc`, `pvt`) within
    each country partition. If a partition is small and these tokens aren't sufficiently
    down-weighted, **fix** with a data-driven (not hardcoded) rule instead: down-weight
    any token appearing in more than X% of names *within that country's own data* — a
    statistical threshold, not a literal word list, so it still generalizes to unseen
    countries.

---

## PHASE C — Representation Building

### Step C1: Build and cache 4 representations per record, computed once

- **What:**
  (a) normalized text string, (b) character n-gram shingles for MinHash,
  (c) numeric tokens extracted from address via regex (street numbers, digit groups —
  no assumption about a fixed PIN/ZIP length), (d) a dense vector from a small
  multilingual sentence-embedding model.
- **Why:** No single technique covers every noise type: typos need character-level
  similarity, cross-script name matching needs embeddings, address precision needs
  exact numeric extraction. Compute once, reuse everywhere downstream — recomputing per
  comparison is not viable at this scale.
- **Example:** `"राम मार्केटिंग प्राइवेट लिमिटेड"` and `"Ram Marketing Private Limited"`
  share zero character n-grams (different scripts) but should land close together in the
  embedding model's vector space.
- **Flaws to check / tests:**
  - **Test:** version/hash the normalization config into the cache file names; if
    normalization logic changes later, stale cached representations must be invalidated
    automatically, not silently reused.
  - **Test — embedding model quality on real data, not assumed:** before trusting the
    embedding model for blocking, sample known true-positive Hindi/English name pairs
    from `train_ground_truth.tsv` and confirm their cosine similarity is meaningfully
    higher than random pairs. Do the same for any other non-Latin script present in
    training data. If the model performs poorly on a specific script, note it as a known
    limitation rather than silently trusting it.
  - **Test — numeric token noise:** check whether short/common numeric tokens (single
    digits, 4-digit years) are being extracted and causing false-positive blocking
    matches between unrelated businesses that just happen to share a common number.
    **Fix:** filter out numeric tokens below a minimum length/rarity, or require a
    numeric-token match to co-occur with a city/locality-token match before counting it
    as a strong signal, rather than treating any shared digit as meaningful alone.

---

## PHASE D — Candidate Generation / Blocking

This phase sets your **recall ceiling** — nothing downstream can recover a true match
this phase fails to surface. Treat it as the highest-priority phase to get right and test.

### Step D1: Hard-partition by (normalized) country

- **Why:** Cuts an otherwise combinatorially impossible comparison space. But partition
  key must be the *cleaned* country string from Step A2, applied generically — not a
  fixed list — so an unseen value like `"France"` still forms its own valid partition.
- **Flaws to check / tests:**
  - **Test:** confirm every distinct country value in the data produces its own
    partition and that partition sizes sum to the total row count (no rows dropped due
    to unexpected/null country values).
  - **Flaw:** a mislabeled `country` field (data entry error) would silently prevent a
    true match from ever being compared, with no visible symptom besides depressed
    recall. **Fix:** for any Source 1 entity that ends up with **zero** candidates after
    Step D3, add one fallback pass that searches across all countries with a stricter
    similarity threshold, rather than assuming cross-country matches are structurally
    impossible.

### Step D2: MinHash-LSH **and** ANN embedding search, unioned

- **What:** MinHash-LSH over character shingles (catches typo/abbreviation-level
  near-duplicates); ANN search (e.g., FAISS) over embedding vectors, top-k per entity
  (catches cross-script/semantic matches sharing no character overlap at all).
- **Why both:** MinHash alone misses `"Ram Marketing"` vs its Devanagari equivalent
  entirely (zero shared n-grams). Embeddings alone can miss a same-script typo that a
  cheap character-level method catches trivially. Take the union, not either alone.
- **Flaws to check / tests:**
  - **Test — recall vs. candidate-count curve:** on the validation split (see Phase G),
    sweep LSH band/row parameters and ANN's `k`, plot blocking recall (fraction of true
    matches surviving blocking) against average candidate-set size. Pick the smallest
    candidate set that still clears a high recall bar (e.g., ≥98%) — don't guess `k=20`
    and move on.
  - **Flaw — approximate index precision:** if using a quantized/IVF-PQ FAISS index for
    memory reasons, the *distance returned during blocking is approximate*, not true
    cosine similarity. Do not carry this value forward as a final classifier feature
    without re-checking it (see Step E1 fix).
  - **Memory flaw at 12M-row scale:** holding fp32 embeddings for all rows in memory
    simultaneously may be infeasible. **Fix:** store embeddings as fp16, and/or use an
    IVF/PQ quantized index instead of a flat index.

### Step D3: Exact-token booster, unioned in before capping

- **What:** Add as a candidate any pair sharing an exact numeric address token together
  with an exact city/locality token (combination, not a lone digit — see Step C1 flaw).
- **Why:** Cheap, high-precision safety net for cases where the name is badly mangled
  but address anchors survived intact.
- **Flaws to check / tests:** confirm this union happens **before** any per-entity
  candidate cap (Step D4) is applied — if capping happens first, this high-precision
  signal can get crowded out by noisier fuzzy candidates ranked above it.

### Step D4: Cap candidates per Source 1 entity (e.g., top ~20–30)

- **Why:** Without a cap, generic names (e.g., "Summit Inc") can pull in hundreds of
  low-value candidates, inflating feature-computation cost with no recall benefit.
- **Flaws to check / tests:**
  - **Test:** bucket Source 1 entities by name frequency/commonness and measure
    blocking recall per bucket separately. If common-name entities show lower recall
    because their true match got crowded out of a fixed cap, **fix** with an adaptive
    cap (raise it selectively for high-collision buckets) rather than one global number.

### Step D5: Write this exact surviving set to `candidate_pairs.tsv`

- **Why:** The spec requires this file to be the literal input your model scores, not
  an earlier looser pass. Getting this wrong triggers a validator warning.
- **Flaws to check / tests:**
  - **Test:** run `utils/validate_submission.py` on `candidate_pairs.tsv` immediately
    after generating it — fail fast on format bugs (duplicate IDs, wrong ID prefixes,
    missing rows) before wasting time on the modeling phases downstream.

---

## PHASE E — Feature Engineering

### Step E1: Reuse blocking-time computation, but recompute where precision matters

- **What:** The candidate set is now small (tens per entity, not millions). Recompute an
  **exact** cosine similarity on full-precision embedding vectors for this final feature
  table, rather than reusing the approximate distance from a quantized ANN index.
- **Why (this corrects an earlier assumption):** blind reuse is fine for *finding*
  candidates cheaply, but an approximate/quantized distance is not precise enough to be
  a reliable input to the classifier. Since the candidate set is now tiny, exact
  recomputation is cheap here — recompute for accuracy at this stage even though you
  used the fast approximate version to survive Phase D.

### Step E2: Batch-compute the remaining pairwise features

- **What:** `rapidfuzz` (C++-backed) for Jaro-Winkler / token-sort ratio, computed in
  batch; sparse TF-IDF cosine similarity via vectorized sparse dot products. Feature
  list: name TF-IDF cosine, name Jaro-Winkler, name token-set ratio, name embedding
  cosine (exact, from E1), address numeric-token overlap, address token Jaccard,
  address embedding cosine, country-match flag, missing-address flag.
- **Why:** Pure-Python string comparison run tens of millions of times is the other
  classic way to make this pipeline impractically slow.
- **Flaws to check / tests:**
  - **Flaw — vocabulary explosion:** fitting a standard TF-IDF vectorizer on the full
    12M-row corpus (many scripts, many unique word forms) may blow up memory with an
    enormous vocabulary. **Fix:** use a `HashingVectorizer` (fixed dimensionality, no
    stored vocabulary) or fit IDF weights on a sampled subset and apply hashing for the
    actual vectors.
  - **Test:** confirm feature computation is called via a batched/vectorized API
    (`rapidfuzz.process.cdist`, not a Python loop calling `fuzz.ratio` per pair), and
    profile actual throughput on a 1% sample before assuming it'll scale.

---

## PHASE F — Training Data Construction

### Step F1: Explode ground truth into positive pairs

- **Example:** `S1-00001 → S2-00047,S3-00812` becomes two labeled rows:
  `(S1-00001, S2-00047, label=1)` and `(S1-00001, S3-00812, label=1)`.

### Step F2: Source negatives from your own blocking output — not randomly

- **Why (critical):** if negatives are randomly sampled unrelated businesses, the
  classifier learns an easy task and becomes overconfident. At inference time, every
  candidate it sees is a *hard* near-miss that survived blocking. Training on the same
  hard negatives it will face live is essential for the precision-heavy F_0.5 metric.
- **Flaws to check / tests:**
  - **Test — unrecoverable positives:** measure the fraction of ground-truth positive
    pairs whose Source 2/3 ID never appeared in that entity's blocked candidate set at
    all. This is your hard recall ceiling. If it's high (e.g., >5–10%), fix Phase D
    before spending any time tuning the classifier — no model can recover a match that
    was never presented to it.
  - **Flaw — leakage via the many-to-many relationship confirmed in Phase A2:** if a
    Source 2/3 entity legitimately matches multiple Source 1 entities, make sure it
    isn't mislabeled as a negative for one of them just because it's listed as positive
    for another. Cross-check every generated negative against the *full* ground truth
    table, not just the current Source 1 entity's row.

### Step F3: Handle class imbalance

- **What:** each Source 1 entity typically yields 1–3 positives against ~20–30 hard
  negatives. Use class weighting (e.g., `scale_pos_weight`) rather than discarding
  negatives — these hard negatives are exactly what the model needs to see.

---

## PHASE G — Model Training

### Step G1: Train a gradient-boosted tree classifier (LightGBM/XGBoost/CatBoost)

- **Why this model family:** scales to tens of millions of feature rows, natively
  tolerates missing features (important since address is sometimes empty), and is
  trained entirely from scratch on provided data — no external-lookup or licensing
  concern. The embedding model from Phase C is the only externally-sourced component and
  must satisfy the MIT/Apache-2.0, ≤8B constraint (record this explicitly in the README).
- **Flaws to check / tests:**
  - **Flaw:** training on the full negative set derived from 12M rows may be too large
    to fit comfortably. **Fix:** cap negatives per Source 1 entity (keep the hardest —
    highest-scoring — negatives, not a random subsample) while retaining full positive
    coverage.
  - **Test:** use early stopping on a held-out validation fold and reasonable
    regularization (max depth, min samples per leaf) to guard against overfitting to
    dataset-specific artifacts in the hard-negative set.

### Step G2: Split by entity, not by row

- **Why:** if pairs from the same Source 1 entity land in both train and validation
  folds, you leak information and get an optimistic score that won't hold on test data.
- **Flaws to check / tests:**
  - **Flaw — chain-store leakage:** near-identical franchise-style businesses across
    *different* real entities (same name, similar address pattern in different
    locations) could still leak signal across an entity-based split. **Test:** manually
    inspect the highest-similarity cross-fold pairs for this pattern; if present,
    consider grouping the split by normalized-name+city cluster instead of raw
    `entity_id`.

---

## PHASE H — Threshold Tuning (this is where F_0.5 is won or lost)

### Step H1: Sweep the probability threshold to maximize macro-averaged F_0.5

- **Why:** F_0.5 weights precision 2x over recall, so the optimal cutoff is typically
  higher than a naive 0.5. Since a correctly-predicted-empty singleton scores 1.0 and a
  false merge on it scores 0.0, erring conservative near the decision boundary usually
  wins even at some recall cost.
- **Example:** threshold 0.50 → precision 0.70 / recall 0.85 → F_0.5 ≈ 0.73.
  Threshold 0.75 → precision 0.88 / recall 0.70 → F_0.5 ≈ 0.83. The higher threshold
  wins despite lower recall, because of the 2x precision weighting.
- **Flaws to check / tests:**
  - **Flaw:** a single global threshold may be systematically miscalibrated for one
    country's noisier data distribution. **Test:** compute F_0.5 per country on the
    validation split; if there's a large gap, calibrate per-country thresholds — but
    **only for country groups with sufficient validation support**.
  - **Flaw (important for generalization):** France has zero labeled training data, so
    it cannot get its own calibrated threshold. **Fix:** any country group not seen in
    validation must fall back to the global-average threshold — do not let a per-group
    calibration mechanism silently produce an undefined/default threshold for unseen
    groups. Write a unit test that feeds a synthetic unseen-country group through the
    threshold-selection code and confirms it falls back correctly rather than erroring
    or defaulting to 0.

---

## PHASE I — Inference and Submission

### Step I1: Run test data through the exact same code path as training

- **Why:** any divergence between train-time and test-time preprocessing (training/
  serving skew) degrades performance in ways that are very hard to debug later.
- **Test:** feed one fixed sample row through both the training-feature-generation call
  and the test-inference call, and assert the resulting feature vectors are identical.

### Step I2: Aggregate predictions into `matching_results.tsv`

- **Important — do not over-correct:** since Source 1 entities may share the same
  matched Source 2/3 record (many-to-many is allowed per the spec, pending the Phase A2
  check), do **not** globally deduplicate a Source 2/3 ID across different Source 1
  rows. The "no duplicate IDs" rule applies *within a single row's list*, not across
  rows.
- Leave `matched_entity_ids` empty for entities with nothing above threshold.

### Step I3: Validate before every submission

- **What:** run `utils/validate_submission.py` on both output files.
- **Flaw to check:** `candidate_pairs.tsv` and `matching_results.tsv` must come from the
  **same pipeline run** — regenerating one after re-tuning parameters without
  regenerating the other creates a silent mismatch. **Fix:** generate both files in a
  single script invocation, and add a test asserting every ID in
  `matching_results.tsv` also appears in that entity's row in `candidate_pairs.tsv`.

---

## PHASE J — Integration Test Suite (build this alongside every phase above, not after)

Build one small synthetic dataset (a few dozen hand-crafted rows) that deliberately
includes every edge case discussed above, and run the full pipeline against it as a fast
end-to-end smoke test after every change:

- A same-script exact match, a same-script typo match, a Devanagari-vs-Latin match, an
  address with reordered components, an address with a missing field, a business with a
  legal suffix as a prefix instead of a suffix, a chain-store name collision (same name,
  different city — must NOT match), a true singleton (no match anywhere), a synthetic
  unseen country value (e.g., `"France"` or a made-up one), and — if confirmed present
  in Phase A2 — a Source 2/3 record legitimately matching two different Source 1 rows.
- Assert: correct matches found, chain-store collision correctly rejected, singleton
  correctly produces an empty list, unseen-country row still produces a valid (possibly
  empty) output row with no crash, and `validate_submission.py` passes on the output.

## Definition of done (per phase)

A phase is not complete until: (1) its steps are implemented, (2) every "Flaws to
check" test in that phase is written and passing, (3) the Phase J synthetic dataset
still passes end-to-end, and (4) blocking-recall / F_0.5 metrics for that phase are
logged and reviewed before moving to the next phase.
