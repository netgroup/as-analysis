import os
import pandas as pd

RANK_DIR = "rank"
FILE_IN = "rank_unordered.tsv"
FILE_OUT = "rank_ordered.csv"

file_path_in = os.path.join(RANK_DIR, FILE_IN)
df = pd.read_csv(file_path_in, sep="\t", names=["rank"]).rename_axis("node").sort_values(by='rank', ascending=False)
file_path_out = os.path.join(RANK_DIR, FILE_OUT)
df.to_csv(file_path_out)