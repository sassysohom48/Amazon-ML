import sys
import time
import math
from collections import defaultdict
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR
from src.evaluator import evaluate_blocking_recall

def test_opt(max_candidates: int = 35):
    print("Testing optimized multi-signal scoring blocker on India...")
    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet").filter(pl.col("country") == "India")
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet").filter(pl.col("country") == "India")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet").filter(pl.col("country") == "India")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")

    # Sample 5,000 S1 entities for rapid iteration
    sample_s1 = val_s1.slice(0, 5000)
    sample_s1_ids = set(sample_s1["entity_id"].to_list())

    val_gt_map = {}
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if s1_id in sample_s1_ids and matches and str(matches).strip():
            val_gt_map[s1_id] = set(str(matches).strip().split(","))

    print(f"Sample: {len(sample_s1):,} S1 entities, {sum(len(v) for v in val_gt_map.values()):,} true matches.")

    targets = pl.concat([train_s2, train_s3])
    t_eids = targets["entity_id"].to_list()
    t_names = targets["name_clean"].to_list()
    t_tokens = targets["name_tokens"].to_list()
    t_addrs = targets["addr_clean"].to_list()
    t_postals = targets["postal_digits"].to_list()

    n_targets = len(targets)
    print(f"Building indexes over {n_targets:,} targets...")
    t0 = time.time()

    exact_name_idx = defaultdict(list)
    token_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    postal_idx = defaultdict(list)

    for idx, (name, tok_str, addr_str, post_str) in enumerate(zip(t_names, t_tokens, t_addrs, t_postals)):
        if name:
            exact_name_idx[name].append(idx)
        toks = tok_str.split() if tok_str else []
        for t in toks:
            token_idx[t].append(idx)
        addrs = addr_str.split() if addr_str else []
        for a in addrs:
            if len(a) >= 3:
                addr_idx[a].append(idx)
        postals = post_str.split() if post_str else []
        for p in postals:
            if len(p) >= 4:
                postal_idx[p].append(idx)

    # Precompute weights
    token_weights = {}
    for t, post in token_idx.items():
        p_len = len(post)
        if p_len <= 35000:
            token_weights[t] = math.log(n_targets / p_len)

    addr_weights = {}
    for a, post in addr_idx.items():
        p_len = len(post)
        if 2 <= p_len <= 10000:
            addr_weights[a] = 0.7 * math.log(n_targets / p_len)

    print(f"Indexes built in {time.time() - t0:.2f}s.")

    # Query
    t1 = time.time()
    cand_map = {}

    s1_eids = sample_s1["entity_id"].to_list()
    s1_names = sample_s1["name_clean"].to_list()
    s1_tokens = sample_s1["name_tokens"].to_list()
    s1_addrs = sample_s1["addr_clean"].to_list()
    s1_postals = sample_s1["postal_digits"].to_list()

    for eid, name, tok_str, addr_str, post_str in zip(s1_eids, s1_names, s1_tokens, s1_addrs, s1_postals):
        scores = defaultdict(float)

        # 1. Exact Name
        if name and name in exact_name_idx:
            for idx in exact_name_idx[name]:
                scores[idx] += 100.0

        toks = tok_str.split() if tok_str else []
        # 2. Name Tokens (weighted)
        for t in toks:
            if t in token_weights:
                w = token_weights[t]
                for idx in token_idx[t]:
                    scores[idx] += w

        # 3. Address Tokens (weighted)
        addrs = [a for a in addr_str.split() if len(a) >= 3] if addr_str else []
        for a in addrs:
            if a in addr_weights:
                w = addr_weights[a]
                for idx in addr_idx[a]:
                    scores[idx] += w

        # 4. Postal
        postals = post_str.split() if post_str else []
        for p in postals:
            if p in postal_idx and len(postal_idx[p]) <= 1000:
                for idx in postal_idx[p]:
                    scores[idx] += 2.0

        if not scores:
            cand_map[eid] = set()
            continue

        # Top K
        if len(scores) <= max_candidates:
            cand_map[eid] = {t_eids[idx] for idx in scores.keys()}
        else:
            # Sort top K
            top_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)[:max_candidates]
            cand_map[eid] = {t_eids[idx] for idx in top_indices}

    t_query = time.time() - t1
    print(f"Queried {len(sample_s1):,} S1 entities in {t_query:.2f}s ({len(sample_s1)/t_query:,.0f} entities/s).")

    metrics = evaluate_blocking_recall(val_gt_map, cand_map)
    print("\n" + "=" * 70)
    print(f"METRICS (5,000 Sample India, K={max_candidates}):")
    print(f"  • Candidate Recall:        {metrics['candidate_recall']*100:.2f}%")
    print(f"  • Captured True Matches:   {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"  • Missed True Matches:     {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
    print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    test_opt(35)
