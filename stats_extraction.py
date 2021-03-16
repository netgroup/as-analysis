'''
Run this script to extract statistics from the original itdk files
'''
import argparse
import os
from functools import cmp_to_key
import pandas as pd
import csv

AS_NODES_FILE = "midar-iff.nodes.as"
LINKS_FILE = "midar-iff.links"
GEO_FILE = "midar-iff.nodes.geo"
GEO_FILE_SORTED = "midar-iff.nodes.geo_sorted"

INPUT_DIR = "itdk_2020_08_clean"
OUTPUT_DIR = "stats"
AS_NODES_FILE = "nodes_as.csv"
LINKS_FILE_FM = "links_fm.csv"
LINKS_FILE_MA = "links_ma.txt"
GEO_FILE = "nodes_geo.csv"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_dir", default=INPUT_DIR, help="input dir containing itdk dataset")
    parser.add_argument("-o", metavar="output_dir", default=OUTPUT_DIR, help="output dir to write statistics")
    args = parser.parse_args()
    return args.i, args.o

def nodes_count(input_dir, output_dir):
    as_count = dict()
    file_path = os.path.join(input_dir, AS_NODES_FILE)
    with open(file_path, 'r', encoding="utf8") as myfile:
        reader = csv.reader(myfile)
        next(reader, None)  # skip the headers
        for line in reader:
            as_number = line[1]
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

def nodes_as_to_dict(input_dir):
    nodes_as = dict()
    file_path = os.path.join(input_dir, AS_NODES_FILE)
    with open(file_path, 'r', encoding="utf8") as myfile:
        reader = csv.reader(myfile)
        next(reader, None)  # skip the headers
        for line in reader:
            nodes_as[line[0]] = line[1]
    return nodes_as

def count_intra_links_ma(input_dir): # TODO
    link_nodes = dict()
    as_links = dict()
    file_path = os.path.join(input_dir, LINKS_FILE_MA)
    a_s = ''
    with open(file_path, 'r', encoding="utf8") as myfile:
        for myline in myfile:
            if not myline.startswith('#'):
            tokens = myline.split(':  ')
            if not len(tokens) == 2:
                print('ERROR LEN TOKENS')
                break
            newtokens = tokens[1].split()
            link_in_as = True
            for node in newtokens:
                onlynode = node.split(':')[0]
                if a_s == '':
                if onlynode not in nodes_as:
                    link_in_as = False
                    break
                a_s = nodes_as[onlynode]
                else:
                if onlynode not in nodes_as:
                    link_in_as = False
                    break
                if nodes_as[onlynode] != a_s:
                    link_in_as = False
                    break
            if link_in_as:
                if a_s in as_links:
                as_links[a_s] = as_links[a_s] + 1
                else:
                as_links[a_s] = 1
            a_s = ''



if __name__ == "__main__":
    input_dir, output_dir = parse()
    # count nodes in every AS
    print("counting nodes for every AS...")
    # nodes_count(input_dir, output_dir)
    print("nodes counted!")
    # generate nodes_as dictionary
    nodes_as = nodes_as_to_dict(input_dir)
    # count links in every AS
    print("counting links for every AS...")