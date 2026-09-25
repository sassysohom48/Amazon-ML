"""
Ultra-Fast Set-Intersection Blocker.
Uses C-level Python set intersections for (tok1 & tok2), (tok1 & addr_tok), and exact names.
"""

import time
from collections import defaultdict
from typing import Dict, List, Set
import polars as pl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR
from src.evaluator import evaluate_blocking_recall


def test_set_intersection_blocker(max_candidates: int = 25):
    print(f"Testing Set-Intersection Blocker on India Validation Split...")
    
    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet").filter(pl.col("country") == "India")
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet").filter(pl.col("country") == "India")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet").filter(pl.col("country") == "India")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")

    india_s1_ids = set(val_s1["entity_id"].to_list())
    val_gt_map = {}
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if s1_id in india_s1_ids and matches and str(matches).strip():
            val_gt_map[s1_id] = set(str(matches).strip().split(","))

    targets = pl.concat([train_s2, train_s3])
    t_eids = targets["entity_id"].to_list()
    t_names = targets["name_clean"].to_list()
    t_tokens = targets["name_tokens"].to_list()
    t_addrs = targets["addr_clean"].to_list()
    t_postals = targets["postal_digits"].to_list()

    print(f"Building set indexes over {len(targets):,} targets...")
    t0 = time.time()

    exact_name_idx = defaultdict(set)
    token_idx = defaultdict(set)
    addr_tok_idx = defaultdict(set)
    postal_idx = defaultdict(set)

    for eid, name, tok_str, addr_str, post_str in zip(t_eids, t_names, t_tokens, t_addrs, t_postals):
        if name:
            exact_name_idx[name].add(eid)

        toks = tok_str.split() if tok_str else []
        for t in toks:
            token_idx[t].add(eid)

        addr_toks = addr_str.split() if addr_str else []
        for at in addr_toks:
            if len(at) >= 3:
                addr_tok_idx[at].add(eid)

        postals = post_str.split() if post_str else []
        for p in postals:
            if len(p) >= 4:
                postal_idx[p].add(eid)

    # Filter out ultra-frequent address tokens (e.g. road, street, city)
    filtered_addr_idx = {k: v for k, v in addr_tok_idx.items() if len(v) <= 10000}
    filtered_token_idx = {k: v for k, v in token_idx.items() if len(v) <= 50000}

    print(f"Indexes built in {time.time() - t0:.2f}s:")
    print(f"  • Exact Names:  {len(exact_name_idx):,}")
    print(f"  • Name Tokens:  {len(filtered_token_idx):,}")
    print(f"  • Addr Tokens:  {len(filtered_addr_idx):,}")
    print(f"  • Postal Codes: {len(postal_idx):,}")

    # Query entities
    s1_eids = val_s1["entity_id"].to_list()
    s1_names = val_s1["name_clean"].to_list()
    s1_tokens = val_s1["name_tokens"].to_list()
    s1_addrs = val_s1["addr_clean"].to_list()
    s1_postals = val_s1["postal_digits"].to_list()

    print(f"\nQuerying {len(s1_eids):,} validation S1 entities...")
    t1 = time.time()
    cand_map = {}

    for eid, name, tok_str, addr_str, post_str in zip(s1_eids, s1_names, s1_tokens, s1_addrs, s1_postals):
        candidates = set()

        # 1. Exact name match
        if name and name in exact_name_idx:
            candidates.update(exact_name_idx[name])

        toks = tok_str.split() if tok_str else []
        addr_toks = [at for at in addr_str.split() if len(at) >= 3] if addr_str else []

        # 2. 2-Token Intersections (t1 & t2)
        if len(toks) >= 2:
            # Sort tokens by frequency so we intersect smallest sets first
            sorted_toks = sorted([t for t in toks if t in filtered_token_idx], key=lambda t: len(filtered_token_idx[t]))
            for i in range(min(len(sorted_toks), 3)):
                for j in range(i + 1, min(len(sorted_toks), 4)):
                    inter = filtered_token_idx[sorted_toks[i]] & filtered_token_idx[sorted_toks[j]]
                    candidates.update(inter)
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

        # 3. 1-Token + 1-Addr Token Intersection (t1 & a1)
        if len(candidates) < max_candidates and toks and addr_toks:
            sorted_toks = sorted([t for t in toks if t in filtered_token_idx], key=lambda t: len(filtered_token_idx[t]))
            sorted_addrs = sorted([a for a in addr_toks if a in filtered_addr_idx], key=lambda a: len(filtered_addr_idx[a]))

            for t in sorted_toks[:2]:
                for a in sorted_addrs[:2]:
                    inter = filtered_token_idx[t] & filtered_addr_idx[a]
                    candidates.update(inter)
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break

        # 4. Rarest single token if it's ultra-distinct (< 1000 records)
        if len(candidates) < max_candidates and toks:
            sorted_toks = sorted([t for t in toks if t in filtered_token_idx], key=lambda t: len(filtered_token_idx[t]))
            for t in sorted_toks[:2]:
                if len(filtered_token_idx[t]) <= 500:
                    candidates.update(filtered_token_idx[t])
                    if len(candidates) >= max_candidates:
                        break

        # 5. Postal code + 1 token intersection
        if len(candidates) < 5:
            postals = post_str.split() if post_str else []
            for p in postals:
                if p in postal_idx and len(postal_idx[p]) <= 500:
                    for t in toks:
                        if t in filtered_token_idx:
                            inter = postal_idx[p] & filtered_token_idx[t]
                            candidates.update(inter)

        # Cap candidates per S1
        if len(candidates) > max_candidates:
            cand_map[eid] = set(list(candidates)[:max_candidates])
        else:
            cand_map[eid] = candidates

    t_query = time.time() - t1
    print(f"Candidate retrieval completed in {t_query:.2f}s ({len(s1_eids)/t_query:,.0f} entities/s).")

    metrics = evaluate_blocking_recall(val_gt_map, cand_map)
    print("\n" + "=" * 70)
    print(f"CANDIDATE BLOCKING METRICS (India Validation, K={max_candidates}):")
    print(f"  • Candidate Recall:        {metrics['candidate_recall']*100:.2f}%")
    print(f"  • Captured True Matches:   {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"  • Missed True Matches:     {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
    print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    test_set_intersection_blocker(25)
