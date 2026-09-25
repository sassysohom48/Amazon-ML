import sys
from pathlib import Path
import polars as pl
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRAIN_GT
from src.text_normalizer import normalize_record
from src.evaluator import evaluate_blocking_recall

def test_top50():
    print("=" * 70)
    print("RUNNING CANDIDATE BLOCKING BENCHMARK ON TOP 50 SAMPLE DATASET")
    print("=" * 70)

    sample_dir = Path("sample dataset")
    df_s1 = pl.read_csv(sample_dir / "top50_train_source1.tsv", separator="\t")
    df_s2 = pl.read_csv(sample_dir / "top50_train_source2.tsv", separator="\t")
    df_s3 = pl.read_csv(sample_dir / "top50_train_source3.tsv", separator="\t")

    # Load train ground truth
    print(f"Loading ground truth from {TRAIN_GT}...")
    gt_df = pl.read_csv(TRAIN_GT, separator="\t")
    
    s1_ids = set(df_s1["entity_id"].to_list())
    gt_map = {}
    for row in gt_df.iter_rows():
        s1_id, matches = row[0], row[1]
        if s1_id in s1_ids and matches and str(matches).strip():
            gt_map[s1_id] = set(str(matches).strip().split(","))

    print(f"Loaded {len(df_s1)} S1 entities, {len(df_s2)} S2 records, {len(df_s3)} S3 records.")
    print(f"Ground Truth contains {sum(len(v) for v in gt_map.values())} true match pairs for these 50 entities.")

    # Clean records
    s1_clean = []
    for r in df_s1.iter_rows(named=True):
        res = normalize_record(r["entity_id"], r.get("business_name"), r.get("business_address"), r.get("country"))
        s1_clean.append({
            "entity_id": res[0], "country": res[1], "name_clean": res[4],
            "addr_clean": res[5], "name_tokens": res[6], "postal_digits": res[8]
        })

    targets_clean = []
    for df in [df_s2, df_s3]:
        for r in df.iter_rows(named=True):
            res = normalize_record(r["entity_id"], r.get("business_name"), r.get("business_address"), r.get("country"))
            targets_clean.append({
                "entity_id": res[0], "country": res[1], "name_clean": res[4],
                "addr_clean": res[5], "name_tokens": res[6], "postal_digits": res[8]
            })

    # Build Index
    exact_name_idx = defaultdict(list)
    token_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    postal_idx = defaultdict(list)

    for r in targets_clean:
        eid = r["entity_id"]
        name = r["name_clean"]
        if name:
            exact_name_idx[name].append(eid)
        for t in r["name_tokens"].split():
            token_idx[t].append(eid)
        for a in r["addr_clean"].split():
            if len(a) >= 3:
                addr_idx[a].append(eid)
        for p in r["postal_digits"].split():
            if len(p) >= 4:
                postal_idx[p].append(eid)

    # Query
    cand_map = {}
    for r in s1_clean:
        eid = r["entity_id"]
        cands = set()
        
        # 1. Exact name
        if r["name_clean"] in exact_name_idx:
            cands.update(exact_name_idx[r["name_clean"]])

        # 2. Name tokens
        for t in r["name_tokens"].split():
            if t in token_idx:
                cands.update(token_idx[t])

        # 3. Address tokens
        for a in r["addr_clean"].split():
            if len(a) >= 3 and a in addr_idx:
                cands.update(addr_idx[a])

        # 4. Postal
        for p in r["postal_digits"].split():
            if len(p) >= 4 and p in postal_idx:
                cands.update(postal_idx[p])

        cand_map[eid] = cands

    # Check matches within the top 50 target pool
    top50_target_ids = {r["entity_id"] for r in targets_clean}
    filtered_gt_map = {k: v & top50_target_ids for k, v in gt_map.items() if (v & top50_target_ids)}

    print(f"\nGround truth matches that exist inside the top50 S2/S3 sample target pool: {sum(len(v) for v in filtered_gt_map.values())}")
    
    if filtered_gt_map:
        metrics = evaluate_blocking_recall(filtered_gt_map, cand_map)
        print("\n" + "=" * 70)
        print("TOP 50 SAMPLE BLOCKING METRICS:")
        print(f"  • Candidate Recall:        {metrics['candidate_recall']*100:.2f}%")
        print(f"  • Captured True Matches:   {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
        print(f"  • Missed True Matches:     {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
        print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']:.2f}")
        print("=" * 70)

    # Print first 10 sample matches
    print("\nSample S1 Candidate Output (First 10 entities):")
    for r in s1_clean[:10]:
        eid = r["entity_id"]
        true_m = gt_map.get(eid, set())
        in_target_pool = true_m & top50_target_ids
        found_m = cand_map.get(eid, set())
        print(f"\nS1: {eid} | Name: '{r['name_clean']}' | Addr: '{r['addr_clean']}'")
        print(f"  Full GT Matches:          {true_m}")
        print(f"  GT in Top50 Target Pool:  {in_target_pool}")
        print(f"  Candidates Retrieved:     {found_m}")
        if in_target_pool:
            print(f"  Captured: {in_target_pool.issubset(found_m)}")

if __name__ == "__main__":
    test_top50()
