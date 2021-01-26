'''
Run this script to extract statistics from the original itdk files
'''
import argparse
import os
from functools import cmp_to_key
import pandas as pd

AS_NODES_FILE = "midar-iff.nodes.as"
LINKS_FILE = "midar-iff.links"
GEO_FILE = "midar-iff.nodes.geo"
GEO_FILE_SORTED = "midar-iff.nodes.geo_sorted"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_dir", default="itdk_2020_08", help="input dir containing itdk dataset")
    parser.add_argument("-o", metavar="output_dir", default="stats", help="output dir to write statistics")
    args = parser.parse_args()
    return args.i, args.o

def nodes_count(input_dir, output_dir):
    as_count = dict()
    file_path = os.path.join(input_dir, AS_NODES_FILE)
    with open(file_path, 'r', encoding="utf8") as myfile:
        for myline in myfile:
            tokens = myline.split()
            as_number = tokens[2]
            if as_number in as_count:
                as_count[as_number] = as_count[as_number] + 1
            else:
                as_count[as_number] = 1
    df = pd.DataFrame(data=as_count.items(), columns=['AS_number', 'nodes_count']).set_index('AS_number')
    df = df.sort_values(by='nodes_count', ascending=False)
    as_count_csv = df.to_csv()
    file_path = os.path.join(output_dir, "as_count.csv")
    with open(file_path, 'w', encoding="utf8") as out_file:
        out_file.write(as_count_csv)
    return as_count



if __name__ == "__main__":
    input_dir, output_dir = parse()
    # count nodes in every AS
    print("counting nodes for every AS...")
    nodes_count(input_dir, output_dir)
    print("nodes counted!")
    # count links in every AS
    print("counting links for every AS...")