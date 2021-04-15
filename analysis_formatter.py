import os
import pandas as pd
import json

ANALYSIS_DIR = "analysis_2020_08"
COLUMNS = "c_names.json"
OUT_FILE = "analysis.tsv"

with open(COLUMNS) as json_file:
  column_names = json.load(json_file)
files_list = os.listdir(ANALYSIS_DIR)
for elem in files_list:
    if elem[0] == '.' or elem[0] == 'a':
        continue
    file_path = os.path.join(ANALYSIS_DIR, elem)
    df = pd.read_csv(file_path, delimiter="\t", names=column_names)
    if 'analysis_df' in locals():
        analysis_df = analysis_df.append(df)
    else:
        analysis_df = df
analysis_df = analysis_df.sort_values(by="ds_nodes").reset_index().drop("index", axis=1)
file_path = os.path.join(ANALYSIS_DIR, OUT_FILE)
analysis_df.to_csv(file_path, sep="\t", index=False)
