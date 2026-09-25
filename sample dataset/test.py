# import pandas as pd
# df = pd.read_csv(r"dataset\dataset\train\train_source1.tsv", sep="\t", nrows=10)
# print(df)

import pandas as pd
from pathlib import Path

files = [r"dataset\dataset\train\train_source1.tsv", r"dataset\dataset\train\train_source2.tsv",r"dataset\dataset\train\train_source3.tsv"]

for file_str in files:
    in_path = Path(file_str)
    
    # Creates "top50_file1.tsv" in the EXACT same folder as the original file
    out_path = in_path.with_name(f"top50_{in_path.name}")
    
    # Automatically creates the folder structure if it doesn't exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read first 50 rows and save
    df = pd.read_csv(in_path, sep="\t", nrows=50)
    df.to_csv(out_path, sep="\t", index=False)
    
    print(f"Saved: {out_path.resolve()}")