import pandas as pd
import matplotlib.pyplot as plt
analysis_df = pd.read_csv("analysis_2020_08/analysis.tsv", delimiter="\t").drop(columns=['avg_shortest_path_len']).dropna()
pd.plotting.scatter_matrix(analysis_df, alpha=0.5, s=0.1, diagonal='kde')
plt.savefig("scatter_matrix.png", dpi=5000)