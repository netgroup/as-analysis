import os
import json
import csv
INPUT_AS_FILE = 'nodes_as.csv'
INPUT_GEO_FILE = 'nodes_geo.csv'
INPUT_DIR = 'itdk_2020_08_clean/'

as_geo = dict()
file_path_as = os.path.join(INPUT_DIR, INPUT_AS_FILE)
file_path_geo = os.path.join(INPUT_DIR, INPUT_GEO_FILE)
no_continent = 0
geo_lost = 0
matches = 0
geo_mag_as = 0

def parse_as(as_line):
    as_node, as_number = as_line
    if as_number in as_geo:
        as_geo[as_number]["tot_nodes_count"] = as_geo[as_number]["tot_nodes_count"] + 1
    else:
        as_geo[as_number] = {
            "tot_nodes_count": 1,
            "NA_nodes_count": 0,
            "SA_nodes_count": 0,
            "EU_nodes_count": 0,
            "AF_nodes_count": 0,
            "AS_nodes_count": 0,
            "OC_nodes_count": 0,
            "AN_nodes_count": 0
        }
    return as_number, as_node

with open(file_path_as, 'r', encoding="utf8") as as_file:
    with open(file_path_geo, 'r', encoding="utf8") as geo_file:
        reader_as = csv.reader(as_file)
        reader_geo = csv.reader(geo_file)
        # skip headers
        next(reader_as, None)
        next(reader_geo, None)
        # read first lines
        # read geo
        geo_line = next(reader_geo, None)
        geo_node, continent = geo_line[:2]
        # read as
        as_line = next(reader_as, None)
        as_number, as_node = parse_as(as_line)

        geo_file_ended = False
        while True:
            if geo_file_ended:
                as_line = next(reader_as, None)
                if as_line == None:
                    print("as file ended")
                    break
                as_number, as_node = parse_as(as_line)
            elif int(geo_node.strip('N')) < int(as_node.strip('N')):
                geo_lost = geo_lost + 1
                geo_line = next(reader_geo, None)
                if geo_line == None:
                    print("geo file ended")
                    geo_file_ended = True
                    continue
                geo_node, continent = geo_line[:2]
            elif int(geo_node.strip('N')) > int(as_node.strip('N')):
                geo_mag_as = geo_mag_as + 1
                as_line = next(reader_as, None)
                if as_line == None:
                    print("as file ended")
                    break
                as_number, as_node = parse_as(as_line)
            elif geo_node == as_node:
                matches = matches + 1
                if continent+"_nodes_count" in as_geo[as_number]:
                    as_geo[as_number][continent+"_nodes_count"] = as_geo[as_number][continent+"_nodes_count"] + 1
                else:
                    # print(geo_line+"no continent specified")
                    if continent != "":
                        print("unknown continent: "+continent)
                        print(geo_node)
                    no_continent = no_continent + 1
                geo_line = next(reader_geo, None)
                if geo_line == None:
                    print("geo file ended")
                    geo_file_ended = True
                    continue
                geo_node, continent = geo_line[:2]

print("no continent: "+str(no_continent))
print("geo lost: "+str(geo_lost))
print("matches: "+str(matches))
print("geo > as: "+str(geo_mag_as))

json_data = json.dumps(as_geo)
with open('stats/geo_data.json', 'w', encoding='utf8') as out_file:
    out_file.write(json_data)