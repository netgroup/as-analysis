import matplotlib.pyplot as plt
import json
import pandas as pd

with open("c_names.json") as json_file:
  column_names = json.load(json_file)
analysis_df = pd.read_csv("analysis-1.tsv", delimiter="\t", names=column_names)
rows = analysis_df.loc[analysis_df['as_number'] == 21508]
for row in rows["dis_leaf_aggr_leaf1_num"]:
    obj = json.loads(row)
print(obj)
df = pd.DataFrame.from_dict(obj, orient="index")
new_index = list(map(str, list(range(1,int(df.last_valid_index())+1))))
df = df.reindex(new_index, fill_value=0)
print(df)


# for row in analysis_df["ri_tot_neighs_dis"]:
#     obj = json.loads(row)
#     print(obj.keys())
#     print(obj.values())

plt.yscale("log")
# plt.xscale("log")
# plt.xticks([])
# plt.bar(obj.keys(), obj.values())
df.plot(ls="", marker = ".")
plt.yscale("log")
# plt.xscale("log")
plt.xlim(left = 0.9)
plt.ylim(bottom = 0.9)
# plt.plot(obj.keys(), obj.values(), ls="", marker = ".")
plt.savefig('foo.pdf')
