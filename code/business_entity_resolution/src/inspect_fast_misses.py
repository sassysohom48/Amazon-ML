import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR

def inspect():
    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet").filter(pl.col("country") == "India")
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")

    india_s1_ids = set(val_s1["entity_id"].to_list())
    gt_pairs = []
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if s1_id in india_s1_ids and matches and str(matches).strip():
            for m in str(matches).strip().split(","):
                gt_pairs.append((s1_id, m.strip()))
                if len(gt_pairs) >= 100:
                    break
        if len(gt_pairs) >= 100:
            break

    print(f"Total India GT pairs sample: {len(gt_pairs)}")
    s1_dict = {r["entity_id"]: r for r in val_s1.filter(pl.col("entity_id").is_in([p[0] for p in gt_pairs])).iter_rows(named=True)}
    
    tgt_ids = [p[1] for p in gt_pairs]
    s2_matches = train_s2.filter(pl.col("entity_id").is_in(tgt_ids))
    s3_matches = train_s3.filter(pl.col("entity_id").is_in(tgt_ids))
    
    tgt_dict = {}
    for r in s2_matches.iter_rows(named=True):
        tgt_dict[r["entity_id"]] = ("S2", r)
    for r in s3_matches.iter_rows(named=True):
        tgt_dict[r["entity_id"]] = ("S3", r)

    print(f"Found {len(tgt_dict)} / {len(tgt_ids)} target entities in S2/S3 tables.")
    
    for i, (s1_id, tgt_id) in enumerate(gt_pairs[:10]):
        s1 = s1_dict.get(s1_id, {})
        source, tgt = tgt_dict.get(tgt_id, ("None", {}))
        print(f"\n--- Pair {i+1} ---")
        print(f"S1 ({s1_id}): Name='{s1.get('name_clean')}' | Tokens='{s1.get('name_tokens')}' | Addr='{s1.get('addr_clean')}' | Country='{s1.get('country')}'")
        print(f"{source} ({tgt_id}): Name='{tgt.get('name_clean')}' | Tokens='{tgt.get('name_tokens')}' | Addr='{tgt.get('addr_clean')}' | Country='{tgt.get('country')}'")

if __name__ == "__main__":
    inspect()
