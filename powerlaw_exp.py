import os
import pandas as pd
import powerlaw

RANK_DIR = "rank"
STATS_DIR = "stats"
FILE_IN = "rank_ordered.csv"

file_path_in = os.path.join(RANK_DIR, FILE_IN)
df = pd.read_csv(file_path_in)
rank_list = df["rank"].to_list()
results = powerlaw.Fit(rank_list)
print(results.power_law.alpha)
print(results.power_law.xmin)
R, p = results.distribution_compare('power_law', 'lognormal')
print("R: {}, p: {}".format(R,p))