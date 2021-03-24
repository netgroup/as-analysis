'''
Run this script to extract statistics from the original itdk files
'''
import argparse
import os
import pandas as pd
import csv
import logging

LOG_PATH = 'logs'
LOG_F_NAME = 'stats_extraction_log'

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("{0}/{1}.log".format(LOG_PATH, LOG_F_NAME))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

INPUT_DIR = "itdk_2020_08_clean"
OUTPUT_DIR = "stats"
AS_NODES_FILE = "nodes_as.csv"
LINKS_FILE_FM = "links_fm.csv"
LINKS_FILE_MA = "links_ma.txt"
OUT_FILE = "as_stats.csv"

START = 'start'
ALL_AS = 'all_as'
NO_AS = 'no_as'
AS_NO_AS = 'as_no_as'
INTER_AS = 'inter_as'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_dir", default=INPUT_DIR, help="input dir containing clean itdk dataset")
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

def links_fm_count(input_dir, nodes_as):
    # every AS has a list with [intra links, links to no-AS nodes, links to other ASs]
    as_links = dict()
    file_path = os.path.join(input_dir, LINKS_FILE_FM)
    with open(file_path, 'r', encoding="utf8") as myfile:
        reader = csv.reader(myfile)
        next(reader, None)  # skip the headers
        for line in reader:
            node1, node2 = line[1:3]
            if node1 in nodes_as:
                as1 = nodes_as[node1]
                if node2 in nodes_as:
                    as2 = nodes_as[node2]
                    if as2 == as1:  # insert intra-AS link in the AS
                        if as1 in as_links:
                            as_links[as1][0] = as_links[as1][0] + 1
                        else:
                            as_links[as1] = [1, 0, 0]
                    else:       # insert inter-AS link in both ASes
                        if as1 in as_links:
                            as_links[as1][2] = as_links[as1][2] + 1
                        else:
                            as_links[as1] = [0, 0, 2]
                        if as2 in as_links:
                            as_links[as2][2] = as_links[as2][2] + 1
                        else:
                            as_links[as2] = [0, 0, 2]
                else:       # insert link to no AS in the AS
                    if as1 in as_links:
                        as_links[as1][1] = as_links[as1][1] + 1
                    else:
                        as_links[as1] = [0, 1, 0]
            elif node2 in nodes_as:     # insert link to no AS in the AS
                as2 = nodes_as[node2]
                if as2 in as_links:
                    as_links[as2][1] = as_links[as2][1] + 1
                else:
                    as_links[as2] = [0, 1, 0]
    return as_links


def links_ma_count(input_dir, nodes_as):
    # every AS has a list with [intra links, links to no-AS nodes, links to other ASs]
    as_links = dict()
    file_path = os.path.join(input_dir, LINKS_FILE_MA)
    with open(file_path, 'r', encoding="utf8") as file_in:
        for line in file_in:
            state = START
            as_list = list()
            tokens = line.split()
            nodes = tokens[1:]
            for node in nodes:
                if state == START:
                    if node in nodes_as:
                        as_list.append(nodes_as[node])
                        state = ALL_AS
                    else:
                        state = NO_AS
                elif state == ALL_AS:
                    if node in nodes_as:
                        if nodes_as[node] in as_list:
                            state = ALL_AS
                        else:
                            as_list.append(nodes_as[node])
                            state = INTER_AS
                    else:
                        state = AS_NO_AS
                elif state == NO_AS:
                    if node in nodes_as:
                        as_list.append(nodes_as[node])
                        state = AS_NO_AS
                elif state == AS_NO_AS:
                    if node in nodes_as:
                        if nodes_as[node] not in as_list:
                            as_list.append(nodes_as[node])
                            state = INTER_AS
                elif state == INTER_AS:
                    if node in nodes_as:
                        if nodes_as[node] not in as_list:
                            as_list.append(nodes_as[node])
                else:
                    logging.info('state = {}, something went wrong while parsing nodes in link: {}'.format(state, line))
            if state == ALL_AS:
                as_number = as_list[0]
                if as_number in as_links:
                    as_links[as_number][0] = as_links[as_number][0] + 1
                else:
                    as_links[as_number] = [1, 0, 0]
            elif state == AS_NO_AS:
                as_number = as_list[0]
                if as_number in as_links:
                    as_links[as_number][1] = as_links[as_number][1] + 1
                else:
                    as_links[as_number] = [0, 1, 0]
            elif state == INTER_AS:
                for as_number in as_list:
                    if as_number in as_links:
                        as_links[as_number][2] = as_links[as_number][2] + 1
                    else:
                        as_links[as_number] = [0, 0, 1]
    return as_links

def merge_and_write_csv(output_dir, nodes, links_ma, links_fm):
    df = pd.DataFrame(data=nodes.items(), columns=['AS_number', 'nodes_count']).set_index('AS_number')
    df = df.sort_values(by='nodes_count', ascending=False)
    links_ma_df = pd.DataFrame.from_dict(data=links_ma, orient="index", columns=["intra_links_ma", "links_to_no_AS_ma", "links_to_other_AS_ma"]).rename_axis("AS_number")
    links_fm_df = pd.DataFrame.from_dict(data=links_fm, orient="index", columns=["intra_links_fm", "links_to_no_AS_fm", "links_to_other_AS_fm"]).rename_axis("AS_number")
    df = df.merge(right=links_ma_df, how='outer', left_index=True, right_index=True)
    df = df.merge(right=links_fm_df, how='outer', left_index=True, right_index=True).sort_values(by='nodes_count', ascending=False).fillna(0).applymap(int)
    print(df)
    as_stats_csv = df.to_csv()
    file_path = os.path.join(output_dir, OUT_FILE)
    with open(file_path, 'w', encoding="utf8") as out_file:
        out_file.write(as_stats_csv)


if __name__ == "__main__":
    input_dir, output_dir = parse()
    logging.info("generating nodes AS dictionary...")
    nodes_as = nodes_as_to_dict(input_dir)
    logging.info("counting nodes for every AS...")
    nodes = nodes_count(input_dir, output_dir)
    logging.info("counting MA links for every AS...")
    links_ma = links_ma_count(input_dir, nodes_as)
    logging.info("counting FM links for every AS...")
    links_fm = links_fm_count(input_dir, nodes_as)
    logging.info("merging and writing to file...")
    merge_and_write_csv(output_dir, nodes, links_ma, links_fm)
    logging.info("done!")