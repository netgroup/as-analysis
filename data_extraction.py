'''
Run this script to extract nodes and links files for every AS
'''
import pandas as pd
import argparse
import os
import csv

AS_NODES_FILE = "nodes_as.csv"
LINKS_FILE_MA = "links_ma.txt"
LINKS_FILE_FM = "links_fm.csv"
NODES_FILE_OUT = '_nodes.txt'
LINKS_FILE_OUT_MA = '_links_ma.txt'
LINKS_FILE_OUT_FM = '_links_fm.csv'
LINKS_FILE_OUT_INTRA_MA = '_links_intra_ma.txt'
LINKS_FILE_OUT_INTRA_FM = '_links_intra_fm.csv'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_dir", default="itdk_2020_08_clean", help="input dir containing itdk dataset cleaned")
    parser.add_argument("-o", metavar="output_dir", default="data_2020_08", help="output dir to write extracted data")
    parser.add_argument("-s", metavar="stats_dir", default="stats", help="input dir to read statistics")
    parser.add_argument("--size", metavar="as_size", default="1000", help="minimum size of ASes to be considered")
    parser.add_argument("--singleas", metavar="single_as", help="AS number of single as to be considered")
    args = parser.parse_args()
    return args.i, args.o, args.s, args.size, args.singleas

def read_as_list(stats_dir, min_size):
    min_size = int(min_size)
    # select all as with the desired size in terms of number of nodes
    as_count_file = os.path.join(stats_dir, "as_count.csv")
    as_df = pd.read_csv(as_count_file).applymap(int)
    for i, row in as_df.iterrows():
        if row["nodes_count"] < min_size:
            as_df = as_df[:i]
            break
    # smallest as first in list
    as_list = as_df["AS_number"].iloc[::-1].tolist()
    as_list[:] = list(map(str, as_list))
    return as_list

def extract_as_nodes(as_number, input_dir, output_dir, as_nodes_file=AS_NODES_FILE):
    """
    selects the nodes that belong to a given AS
    as_number is a string like '34577'
    writes the nodes into the file AS_NUMBER_nodes.txt
    returns a set object with the nodes IDs
    """
    as_nodes = set(())
    file_path = os.path.join(input_dir, as_nodes_file)
    with open(file_path, 'r', encoding="utf8") as myfile:
        file_path_out = os.path.join(output_dir, as_number+NODES_FILE_OUT)
        with open(file_path_out, 'w', encoding="utf8") as out_file:
            reader = csv.reader(myfile)
            next(reader, None)  # skip the headers
            for line in reader:
                if line[1] == as_number:
                    node = line[0]
                    out_file.write(node+'\n')
                    as_nodes.add(node)
    return as_nodes

def extract_as_links_ma(set_of_as_nodes, as_number, input_dir, output_dir, links_file=LINKS_FILE_MA):
    """
    selects links of an autonomous sytems
    selects all links that insist on a node that belongs to set_of_as_nodes
    it assumes that the set_of_as_nodes contains the nodes of the AS
    set_of_as_nodes is a set of node IDs e.g. 'N23548457'
    writes the output into AS_NUMBER_links_ma.txt
    as_number is a string like '34577'
    """
    file_path = os.path.join(input_dir, links_file)
    with open(file_path, 'r', encoding="utf8") as myfile:
        file_path_out = os.path.join(output_dir, as_number + LINKS_FILE_OUT_MA)
        with open(file_path_out, 'w', encoding="utf8") as out_file:
            for line in myfile:
                tokens = line.split()[1:]
                for node in tokens:
                    if node in set_of_as_nodes:
                        out_file.write(line)
                        break

def extract_as_links_fm(set_of_as_nodes, as_number, input_dir, output_dir, links_file=LINKS_FILE_FM):
    file_path = os.path.join(input_dir, links_file)
    with open(file_path, 'r', encoding="utf8") as myfile:
        file_path_out = os.path.join(output_dir, as_number + LINKS_FILE_OUT_FM)
        with open(file_path_out, 'w', encoding="utf8") as out_file:
            reader = csv.reader(myfile)
            writer = csv.writer(out_file)
            header = next(reader, None)
            writer.writerow(header)
            for line in reader:
                if line[1] in set_of_as_nodes or line[2] in set_of_as_nodes:
                    writer.writerow(line)

# TODO links intra
def extract_as_links_intra_ma(set_of_as_nodes, as_number, input_dir, output_dir, links_file=LINKS_FILE_MA):
    file_path = os.path.join(input_dir, links_file)
    with open(file_path, 'r', encoding="utf8") as myfile:
        file_path_out = os.path.join(output_dir, as_number + LINKS_FILE_OUT_INTRA_MA)
        with open(file_path_out, 'w', encoding="utf8") as out_file:
            for line in myfile:
                tokens = line.split()[1:]
                link_in_as = True
                for node in tokens:
                    if node not in set_of_as_nodes: # entire link is not intra AS
                        link_in_as = False
                        break
                if link_in_as:
                    out_file.write(line)

def extract_as_links_intra_fm(set_of_as_nodes, as_number, input_dir, output_dir, links_file=LINKS_FILE_FM):
    file_path = os.path.join(input_dir, links_file)
    with open(file_path, 'r', encoding="utf8") as myfile:
        file_path_out = os.path.join(output_dir, as_number + LINKS_FILE_OUT_INTRA_FM)
        with open(file_path_out, 'w', encoding="utf8") as out_file:
            reader = csv.reader(myfile)
            writer = csv.writer(out_file)
            header = next(reader, None)
            writer.writerow(header)
            for line in reader:
                if line[1] in set_of_as_nodes and line[2] in set_of_as_nodes:
                    writer.writerow(line)

def run_all(input_dir, output_dir, stats_dir, min_size, single_as):
    # get list of ASes to be extracted
    if single_as is None:
        as_list = read_as_list(stats_dir, min_size)
    else:
        as_list = [single_as]
    # extract nodes and links for every AS
    for as_number in as_list:
        print("extracting nodes from AS {}...".format(as_number))
        as_nodes = extract_as_nodes(as_number, input_dir, output_dir)
        print("extracting ma links...")
        extract_as_links_ma(as_nodes, as_number, input_dir, output_dir)
        print("extracting fm links...")
        extract_as_links_fm(as_nodes, as_number, input_dir, output_dir)
        print("extracting ma intra links...")
        extract_as_links_intra_ma(as_nodes, as_number, input_dir, output_dir)
        print("extracting fm intra links...")
        extract_as_links_intra_fm(as_nodes, as_number, input_dir, output_dir)
    print("extraction complete!")


if __name__ == "__main__":
    input_dir, output_dir, stats_dir, min_size, single_as = parse()
    run_all(input_dir, output_dir, stats_dir, min_size, single_as)
    