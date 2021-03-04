import matplotlib.pyplot as plt
import json
import pandas as pd

with open("c_names.json") as json_file:
  column_names = json.load(json_file)
analysis_df = pd.read_csv("analysis-1.tsv", delimiter="\t", names=column_names)
analysis_df = pd.read_csv("analysis-1.tsv", delimiter="\t", names=column_names, index_col=None)

analysis_df = analysis_df.reset_index().drop("index", axis=1).sort_values(by="ds_nodes")
# df = analysis_df[["ri_avg_tot_neighs", "ds_links"]]
df = analysis_df["ds_links"].reset_index().drop("index", axis=1)
print(df)

# df.plot(ls="", marker = ".", x="ri_avg_tot_neighs", markersize=7, alpha=.3)
df.plot(ls="", marker = ".")
plt.yscale("log")
# plt.xscale("log")
plt.savefig('foo.pdf')