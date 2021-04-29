import pandas as pd
import json
from itertools import islice

RATIO = 0.7
MIN_SIZE = 1000
BATCH_SIZE = 100



def chunks(data, SIZE=BATCH_SIZE):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}


# prepare dictionary with geo data

with open("stats/geo_continent_data.json", "r", encoding="utf8") as file_in:
    geo_all_dict = json.load(file_in)
geo_dict = dict()
continent_count = {
    "NA": 0,
    "SA": 0,
    "EU": 0,
    "AF": 0,
    "AS": 0,
    "OC": 0,
    "AN": 0,
    "MANY": 0
}
for as_num, data in geo_all_dict.items():
    # if data["tot_nodes_count"] < MIN_SIZE:
    #     pass
    if data["NA_nodes_count"] >= RATIO * data["tot_nodes_count"]:
        geo_dict[as_num] = "NA"
        continent_count["NA"] += 1
    elif data["SA_nodes_count"] >= RATIO * data["tot_nodes_count"]:
        geo_dict[as_num] = "SA"
        continent_count["SA"] += 1
    elif data["EU_nodes_count"] >= RATIO * data["tot_nodes_count"]:
        geo_dict[as_num] = "EU"
        continent_count["EU"] += 1
    elif data["AF_nodes_count"] >= RATIO * data["tot_nodes_count"]:
        geo_dict[as_num] = "AF"
        continent_count["AF"] += 1
    elif data["AS_nodes_count"] >= RATIO * data["tot_nodes_count"]:
        geo_dict[as_num] = "AS"
        continent_count["AS"] += 1
    elif data["OC_nodes_count"] >= RATIO * data["tot_nodes_count"]:
        geo_dict[as_num] = "OC"
        continent_count["OC"] += 1
    elif data["AN_nodes_count"] >= RATIO * data["tot_nodes_count"]:
        geo_dict[as_num] = "AN"
        continent_count["AN"] += 1
    else:
        continent_count["MANY"] += 1


# import dataframe with analysis data

analysis_df = pd.read_csv("analysis_2020_08/analysis.tsv", delimiter="\t", index_col=0).dropna(how="all", subset=["avg_coreness", "graph_coreness", "core_order", "density_lcc", "assortativity_lcc", "transitivity_lcc", "avg_shortest_path_len", "approx_avg_shortest_path_len"]).drop(columns=["ri_pp_ifs_dis", "ri_tot_neighs_dis", "dis_leaf1_aggr_type", "dis_leaf_aggr_leaf1_num", "re_pp_ifs_dis", "re_tot_neighs_dis"])


# per-continent stats

continents = dict()
for continent, count in continent_count.items():
    continent_dict = dict()
    for column in analysis_df.columns:
        continent_dict[column] = 0
    continent_dict["count"] = 0
    continents[continent] = continent_dict
for index, row in analysis_df.iterrows():
    as_number = str(index)
    if as_number not in geo_dict:
        continue
    continent = geo_dict[as_number]
    for column in analysis_df.columns:
        continents[continent][column] = continents[continent][column] + row[column]
    continents[continent]["count"] = continents[continent]["count"] + 1
# average
for continent, row in continents.items():
    for column, data in row.items():
        if row["count"] != 0 and column != "count":
            continents[continent][column] = data / row["count"]

continent_df = pd.DataFrame.from_dict(continents, orient="index").drop(["AN", "MANY"]).drop(columns=["avg_shortest_path_len"])
# print(continent_df)

# continent stats ordered by number of nodes

sorted_geo_all_dict = {k: v for k, v in sorted(geo_all_dict.items(), key=lambda item: item[1]["tot_nodes_count"]) if v["tot_nodes_count"] >= MIN_SIZE}


ratio_dict = dict()
for as_number, data in sorted_geo_all_dict.items():
    tot = data["tot_nodes_count"]
    ratio_dict[as_number] = max(data["NA_nodes_count"], data["SA_nodes_count"], data["EU_nodes_count"], data["AF_nodes_count"], data["AS_nodes_count"], data["OC_nodes_count"], data["AN_nodes_count"]) / tot
# print(ratio_dict)

chunks_data = dict()
i = 0
for chunk in chunks(ratio_dict, 30):
    count = 0
    for as_number, value in chunk.items():
        if value < RATIO:
            count += 1
    chunks_data[i] = {
        "multi_continent_AS_count": count,
        "largest_AS_nodes_count": sorted_geo_all_dict[as_number]["tot_nodes_count"]
    }
    i += 1

chunks_df = pd.DataFrame.from_dict(chunks_data, orient="index")
with open("stats/geo_chunks.csv", "w", encoding="utf8") as out_file:
    chunks_df.to_csv(out_file)
print(chunks_df)
