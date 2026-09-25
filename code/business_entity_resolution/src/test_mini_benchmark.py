import sys
import time
import math
from collections import defaultdict
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR
from src.evaluator import evaluate_blocking_recall

def run_mini_benchmark(sample_n: int = 500, max_candidates: int = 35):
    print("=" * 70)
    print(f"FAST MINI-BENCHMARK: {sample_n} S1 ENTITIES WITH FULL TARGET MATCHES")
    print("=" * 70)

    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")
    
    # 1. Pick sample S1 entities
    s1_sample = val_s1.slice(0, sample_n)
    s1_sample_ids = set(s1_sample["entity_id"].to_list())

    # 2. Extract their Ground Truth
    gt_map = {}
    all_true_target_ids = set()
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if s1_id in s1_sample_ids and matches and str(matches).strip():
            m_set = set(str(matches).strip().split(","))
            gt_map[s1_id] = m_set
            all_true_target_ids.update(m_set)

    total_true_pairs = sum(len(v) for v in gt_map.values())
    print(f"Entities: {len(s1_sample):,} | True Match Pairs: {total_true_pairs:,} | Target Match IDs: {len(all_true_target_ids):,}")

    # 3. Load Targets: All true targets + 30,000 distractor targets
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet")

    s2_true = train_s2.filter(pl.col("entity_id").is_in(list(all_true_target_ids)))
    s3_true = train_s3.filter(pl.col("entity_id").is_in(list(all_true_target_ids)))
    s2_distractors = train_s2.slice(0, 15000)
    s3_distractors = train_s3.slice(0, 15000)

    targets = pl.concat([s2_true, s3_true, s2_distractors, s3_distractors]).unique(subset=["entity_id"])
    print(f"Target Pool Size: {len(targets):,} records (contains 100% of true matches + 30k distractors).")

    # 4. Build Multi-Signal Inverted Index
    t_eids = targets["entity_id"].to_list()
    t_names = targets["name_clean"].to_list()
    t_tokens = targets["name_tokens"].to_list()
    t_addrs = targets["addr_clean"].to_list()
    t_postals = targets["postal_digits"].to_list()

    exact_name_idx = defaultdict(list)
    token_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    postal_idx = defaultdict(list)

    for idx, (name, tok_str, addr_str, post_str) in enumerate(zip(t_names, t_tokens, t_addrs, t_postals)):
        if name:
            exact_name_idx[name].append(idx)
        for t in tok_str.split() if tok_str else []:
            token_idx[t].append(idx)
        for a in addr_str.split() if addr_str else []:
            if len(a) >= 3:
                addr_idx[a].append(idx)
        for p in post_str.split() if post_str else []:
            if len(p) >= 4:
                postal_idx[p].append(idx)

    # Weights
    n_targets = len(targets)
    token_weights = {t: math.log(n_targets / len(p)) for t, p in token_idx.items()}
    addr_weights = {a: 0.7 * math.log(n_targets / len(p)) for a, p in addr_idx.items() if len(p) >= 2}

    # 5. Query Entities
    s1_eids = s1_sample["entity_id"].to_list()
    s1_names = s1_sample["name_clean"].to_list()
    s1_tokens = s1_sample["name_tokens"].to_list()
    s1_addrs = s1_sample["addr_clean"].to_list()
    s1_postals = s1_sample["postal_digits"].to_list()

    t_start = time.time()
    cand_map = {}

    for eid, name, tok_str, addr_str, post_str in zip(s1_eids, s1_names, s1_tokens, s1_addrs, s1_postals):
        scores = defaultdict(float)

        # Exact Name (+100.0)
        if name and name in exact_name_idx:
            for idx in exact_name_idx[name]:
                scores[idx] += 100.0

        # Name Tokens (rarest tokens get higher IDF)
        toks = tok_str.split() if tok_str else []
        for t in toks:
            if t in token_weights:
                scores_delta = token_weights[t]
                for idx in token_idx[t]:
                    scores[idx] += scores_delta

        # Address Tokens
        addrs = [a for a in addr_str.split() if len(a) >= 3] if addr_str else []
        for a in addrs:
            if a in addr_weights:
                scores_delta = addr_weights[a]
                for idx in addr_idx[a]:
                    scores[idx] += scores_delta

        # Postal
        postals = post_str.split() if post_str else []
        for p in postals:
            if p in postal_idx and len(postal_idx[p]) <= 500:
                for idx in postal_idx[p]:
                    scores[idx] += 2.0

        if not scores:
            cand_map[eid] = set()
            continue

        if len(scores) <= max_candidates:
            cand_map[eid] = {t_eids[idx] for idx in scores.keys()}
        else:
            top_k = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)[:max_candidates]
            cand_map[eid] = {t_eids[idx] for idx in top_k}

    elapsed = time.time() - t_start
    print(f"Queried {len(s1_sample):,} entities in {elapsed:.3f}s ({len(s1_sample)/elapsed:,.0f} entities/s).")

    # Evaluate
    metrics = evaluate_blocking_recall(gt_map, cand_map)
    print("\n" + "=" * 70)
    print(f"MINI-BENCHMARK RESULTS (K={max_candidates}):")
    print(f"  • Candidate Recall:        {metrics['candidate_recall']*100:.2f}%")
    print(f"  • Captured True Matches:   {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"  • Missed True Matches:     {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
    print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']:.2f}")
    print("=" * 70)

    return metrics

if __name__ == "__main__":
    run_mini_benchmark(500, max_candidates=35)
