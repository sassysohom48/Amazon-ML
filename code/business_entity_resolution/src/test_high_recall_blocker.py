"""
High-Recall Blocker Proof of Concept.
Combines Name Tokens + Distinctive Address Tokens + Postal Codes to break 95% Candidate Recall.
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


def test_high_recall_blocker(max_candidates: int = 40):
    print("=" * 70)
    print(f"HIGH-RECALL BLOCKER EVALUATION (K={max_candidates})")
    print("=" * 70)

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

    print(f"Indexing {len(targets):,} target records (Name + Address + Postal)...")
    t0 = time.time()

    exact_name_idx = defaultdict(list)
    token_idx = defaultdict(list)
    addr_tok_idx = defaultdict(list)
    postal_idx = defaultdict(list)

    for eid, name, tok_str, addr_str, post_str in zip(t_eids, t_names, t_tokens, t_addrs, t_postals):
        if name:
            exact_name_idx[name].append(eid)

        toks = tok_str.split() if tok_str else []
        for t in toks:
            token_idx[t].append(eid)

        addr_toks = addr_str.split() if addr_str else []
        for at in addr_toks:
            if len(at) >= 3:
                addr_tok_idx[at].append(eid)

        postals = post_str.split() if post_str else []
        for p in postals:
            if len(p) >= 4:
                postal_idx[p].append(eid)

    # Filter out ultra-frequent noise tokens
    filtered_token_idx = {k: v for k, v in token_idx.items() if len(v) <= 35000}
    filtered_addr_idx = {k: v for k, v in addr_tok_idx.items() if len(v) <= 8000}
    filtered_postal_idx = {k: v for k, v in postal_idx.items() if len(v) <= 800}

    print(f"Indexes built in {time.time() - t0:.2f}s:")
    print(f"  • Exact Names:     {len(exact_name_idx):,}")
    print(f"  • Name Tokens:     {len(filtered_token_idx):,}")
    print(f"  • Address Tokens:  {len(filtered_addr_idx):,}")
    print(f"  • Postal Codes:    {len(filtered_postal_idx):,}")

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

        # 1. Exact Name Match
        if name and name in exact_name_idx:
            candidates.update(exact_name_idx[name][:20])

        toks = tok_str.split() if tok_str else []
        addr_toks = [at for at in addr_str.split() if len(at) >= 3] if addr_str else []

        # 2. Rarest Name Tokens (up to 3 rarest tokens)
        if toks:
            token_postings = []
            for t in toks:
                if t in filtered_token_idx:
                    token_postings.append((len(filtered_token_idx[t]), t))
            token_postings.sort(key=lambda x: x[0])

            for _, t in token_postings[:3]:
                candidates.update(filtered_token_idx[t][:15])
                if len(candidates) >= max_candidates:
                    break

        # 3. Distinctive Address Tokens (if still under max_candidates)
        if len(candidates) < max_candidates and addr_toks:
            addr_postings = []
            for at in addr_toks:
                if at in filtered_addr_idx:
                    addr_postings.append((len(filtered_addr_idx[at]), at))
            addr_postings.sort(key=lambda x: x[0])

            for _, at in addr_postings[:2]:
                candidates.update(filtered_addr_idx[at][:10])
                if len(candidates) >= max_candidates:
                    break

        # 4. Postal Code Matches
        if len(candidates) < max_candidates:
            postals = post_str.split() if post_str else []
            for p in postals:
                if p in filtered_postal_idx:
                    candidates.update(filtered_postal_idx[p][:10])
                    if len(candidates) >= max_candidates:
                        break

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
    test_high_recall_blocker(40)
