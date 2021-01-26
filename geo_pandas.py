import pandas as pd
import json
with open("stats/geo_data.json", 'r') as in_file:
  json_data = json.load(in_file)
  df = pd.DataFrame.from_dict(json_data, orient='index').sort_values(by="tot_nodes_count", ascending=False)
  print(df)
  df.to_csv("stats/geography.csv")
