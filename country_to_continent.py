import csv
import json
import pandas as pd

INPUT_FILE = 'itdk_2020_08_clean/nodes_geo.csv'
POP_GDP_FILE = "stats/lcc_pop_gdp_country.json"

ctr_to_cnt_dict = dict()
with open(INPUT_FILE, 'r', encoding="utf8") as geo_file:
    reader_geo = csv.reader(geo_file)
    # skip headers
    next(reader_geo, None)
    for row in reader_geo:
        country = row[2]
        continent = row[1]
        if continent != '' and country !='':
            if country not in ctr_to_cnt_dict:
                ctr_to_cnt_dict[country] = continent

with open(POP_GDP_FILE, 'r', encoding="utf8") as pop_gdp_file:
    pop_gdp = json.load(pop_gdp_file)
for country, continent in ctr_to_cnt_dict.items():
    if country in pop_gdp:
        pop_gdp[country]["continent"] = continent
df = pd.DataFrame.from_dict(pop_gdp, orient="index")
df.dropna(inplace=True, subset=["population", "gdp"])
df = df.query("ds_nodes != 0")
df.to_csv("stats/lcc_table_full.csv")