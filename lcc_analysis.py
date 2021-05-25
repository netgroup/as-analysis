'''
build networkx graph and analyze
'''
import pandas as pd
import argparse
import os
import networkx as nx
import json
from itertools import islice
import numpy as np
import csv
import math
import time
import logging
from multiprocessing import Pool

LOG_PATH = 'logs'
LOG_F_NAME = 'graph_analysis_log'

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("{0}/{1}.log".format(LOG_PATH, LOG_F_NAME))
fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.addHandler(fileHandler)

NODES_FILE_OUT = '_nodes.txt'
LINKS_FILE_OUT_MA = '_links_ma.txt'
LINKS_FILE_OUT_FM = '_links_fm.csv'
LINKS_FILE_OUT_INTRA_MA = '_links_intra_ma.txt'
LINKS_FILE_OUT_INTRA_FM = '_links_intra_fm.csv'

column_names = []
data_tsv = ''
columns_details = []


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_dir", default="data_2020_08",
                        help="input dir containing per AS extracted data")
    parser.add_argument("-o", metavar="output_dir", default="analysis_2020_08",
                        help="output dir to write analysis output")
    parser.add_argument("-s", metavar="stats_dir",
                        default="stats", help="input dir to read statistics")
    parser.add_argument("--size", metavar="as_size", default="20",
                        help="minimum size of ASes to be considered")
    parser.add_argument("--max-size", metavar="max_size", default="10000000",
                        help="maximum size of ASes to be considered")
    parser.add_argument("--singleas", metavar="single_as",
                        help="AS number of single as to be considered")
    parser.add_argument("--links", metavar="links_type", default="fm",
                        help="link types: ma fm intra_ma intra_fm", choices=["ma", "fm", "intra_ma", "intra_fm"])
    parser.add_argument("--minlcc", metavar="min_lcc_coverage", default="0.9",
                        help="minimum LCC coverage of the AS for being analyzed")
    parser.add_argument("--whole", action='store_false', help="analyze not only LCC")
    parser.add_argument("--noapprox", action="store_false",
                        help="do not approximate results for average shortest path length (use for large topologies)")
    parser.add_argument("--parallelize", action='store_true', help="parallelize analysis over multiple CPUs")
    args = parser.parse_args()
    return args.i, args.o, args.s, args.size, args.singleas, args.links, args.whole, args.minlcc, args.noapprox, args.parallelize, args.max_size


def compute_approx_function(min_size, max_size, max_cut):
    max_cut = max_cut * max_size        # from percentage to actual number
    slope = max_cut / (max_size - min_size)
    const = min_size * slope
    return slope, const


def step_calculator(min_size, max_size, max_cut, nodes_number):
    if nodes_number > max_size:
        keep = max_size * (1 - max_cut)
    else:
        slope, const = compute_approx_function(min_size, max_size, max_cut)
        cut = slope * nodes_number - const
        keep = nodes_number - cut
        logging.debug("nodes: {}, cut: {}, keep: {}".format(nodes_number, cut, keep))
    step = int(round(nodes_number / keep))
    logging.debug("step: {}".format(step))
    return step


def approx_aspl(G, min_size=1e04, max_size=3e06, max_cut=0.99):
    nodes_number = G.graph["largest_cc_size"]
    if nodes_number < min_size:
        # return nx.average_shortest_path_length(G)
        use_all = True
    else:
        use_all = False
    step = step_calculator(min_size, max_size, max_cut, nodes_number)
    state = "source"
    source_nodes_number = 0
    tot_paths_len = 0
    nodes = iter(G.nodes)
    for node in nodes:
        if state == "source" or use_all:
            source_nodes_number = source_nodes_number + 1
            paths = nx.single_source_shortest_path_length(G, node)
            avg_path_len = sum(paths.values()) / len(paths)
            tot_paths_len = tot_paths_len + avg_path_len
            state = "step"
        elif state == "step":
            for i in range(2, step):
                next(nodes, None)
            state = "source"
    return tot_paths_len / source_nodes_number


def read_as_list(stats_dir, min_size, max_size):
    min_size = int(min_size)
    max_size = int(max_size)
    # select all as with the desired min size in terms of number of nodes
    as_count_file = os.path.join(stats_dir, "as_count.csv")
    as_df = pd.read_csv(as_count_file).applymap(int)
    for i, row in as_df.iterrows():
        if row["nodes_count"] < min_size:
            as_df = as_df[:i]
            break
    # select all as with the desired max size in terms of number of nodes
    for i, row in as_df.iterrows():
        if row["nodes_count"] < max_size:
            as_df = as_df[i:]
            break
    # smallest as first in list
    as_list = as_df["AS_number"].iloc[::-1].tolist()
    as_list[:] = list(map(str, as_list))
    return as_list


def read_as_nodes_from_file(path):
    """Reads nodes from file, returns a set of nodes
    as_number is a string like '34577'
    the file name is AS_NUMBER_nodes.txt or AS_NUMBER_nodes.txt
    """
    as_nodes = []
    with open(path, 'r', encoding="utf8") as myfile:
        for myline in myfile:
            node = myline.strip()
            as_nodes.append(node)
    return as_nodes


def read_links_from_file(path, links_type):
    """
    Reads links from file, returns a dictionary
    as_number is a string like '34577'
    the file name is AS_NUMBER_links_ma.txt or  AS_NUMBER_links_intra_fm.csv
    the dictionary contains the link id as key, and 
    the list of nodes insisting on the link as value
    """
    links = dict()
    with open(path, 'r', encoding="utf8") as myfile:
        if links_type == "ma":
            for myline in myfile:
                tokens = myline.split()
                node_list = tokens[1:]
                links[tokens[0]] = node_list
        elif links_type == "fm":
            reader = csv.reader(myfile)
            next(reader, None)  # skip the headers
            for line in reader:
                links[line[0]] = [line[1], line[2]]
    return links


def split_dict(data, size):
    result = []
    it = iter(data)
    for i in range(0, len(data), size):
        result.append({k: data[k] for k in islice(it, size)})
    return result


def update_graph_from_links(links, G):
    """Updates the graph G with the information in the links dictionary"""
    for link_id, node_list in links.items():
        if len(node_list) > 2:
            G.add_node(link_id, type='sw')
            for node in node_list:
                # TODO check if belonging to another AS?
                if node not in G:
                    G.add_node(node, type='re')
                G.add_edge(node, link_id, type='ma', weight=1)
                if nx.is_directed(G):
                    G.add_edge(link_id, node,  type='ma', weight=0)
        else:
            if not node_list[0] in G:
                G.add_node(node_list[0], type='re')
            if not node_list[1] in G:
                G.add_node(node_list[1], type='re')
            G.add_edge(node_list[0], node_list[1],
                       id=link_id, type='pp', weight=1)
            if nx.is_directed(G):
                G.add_edge(node_list[1], node_list[0],
                           id=link_id, type='pp', weight=1)


def create_graph_from_files(as_number, G, input_dir, links_type="ma"):
    """
    create graph from nodes and links files
    as_number is a string like '34577'
    """
    nodes_file_path = os.path.join(input_dir, as_number + NODES_FILE_OUT)
    as_nodes = read_as_nodes_from_file(nodes_file_path)
    logging.debug("AS: {}".format(as_number))
    logging.debug("Number of nodes (AS routers): {}".format(len(as_nodes)))
    if links_type == "ma":
        links_file_suffix = LINKS_FILE_OUT_MA
    elif links_type == "fm":
        links_file_suffix = LINKS_FILE_OUT_FM
    elif links_type == "intra_ma":
        links_file_suffix = LINKS_FILE_OUT_INTRA_MA
        links_type = "ma"
    elif links_type == "intra_fm":
        links_file_suffix = LINKS_FILE_OUT_INTRA_FM
        links_type = "fm"
    else:
        pass
    links_file_path = os.path.join(input_dir, as_number + links_file_suffix)
    links = read_links_from_file(links_file_path, links_type)
    logging.debug("Number of links: {}".format(len(links)))
    G.graph['dataset_nodes'] = len(as_nodes)
    array_num_split = 600 if len(as_nodes) >= 600 else 1
    as_nodes_splitted = np.array_split(np.array(as_nodes), array_num_split)
    del as_nodes
    for as_nodes_s in as_nodes_splitted:
        G.add_nodes_from(as_nodes_s, type='ri')
    del as_nodes_splitted

    G.graph['dataset_links'] = len(links)
    link_split_size = 500 if len(links) >= 500 else 1
    links_splitted = split_dict(links, link_split_size)
    del links
    for item in links_splitted:
        update_graph_from_links(item, G)

    G.graph['as_number'] = as_number


class Mydistribution:
    def __init__(self):
        self.distrib = dict()

    def __contains__(self, key):
        return key in self.distrib

    def __str__(self):
        return json.dumps(self.distrib, sort_keys=True)

    def to_json(self):
        return json.dumps(self.distrib, sort_keys=True)

    def add(self, value):
        previous = self.distrib.setdefault(value, 0)
        self.distrib[value] = previous+1

    def mean(self):
        num = 0
        tot_value = 0
        for key, value in self.distrib.items():
            num = num + value
            tot_value = tot_value + value*key
        if num != 0:
            return float(tot_value)/num
        else:
            return "N/A"

    def maxval(self):
        maxvalue = 0
        for key in self.distrib.keys():
            if key > maxvalue:
                maxvalue = key
        return maxvalue

    def len(self):
        """ number of different values"""
        return len(self.distrib)

    def count(self):
        """ number of elements in the distribution"""
        num = 0
        for key, value in self.distrib.items():
            num = num + value
        return num

    def get(self, key):
        return self.distrib[key]


def reset_tsv():
    global column_names
    global data_tsv
    global columns_details

    column_names = []
    data_tsv = ''
    columns_details = []


def add_to_tsv(col_name, value, details):
    global column_names
    global data_tsv
    global columns_details

    column_names.append(col_name)
    data_tsv = data_tsv + '\t' + str(value)
    columns_details.append((col_name, details))


def m_add_to_tsv(couple_list):
    for couple in couple_list:
        add_to_tsv(couple[0], couple[1], couple[2])


def clusterize_leaf_nodes(G):
    """identify leaf_aggregator nodes by clustering leaf1 node

    a leaf_aggregator node is a node that is connected to a leaf1 node
    a node is a leaf_aggregator node if it has the attribute
    G.nodes[node]['leaf1_neigh'] : number of leaf1 node connected
    """

    clusters = Mydistribution()

    # iterates of all nodes and properties
    for n, attr in G.nodes.data():

        if attr['type'] == 'ri':
            if attr['class'] == 'leaf1':

                # only leaf1 ri nodes are selected
                # iterates over all neighbors
                for node in G.adj[n].keys():
                    clusters.add(node)
                    G.nodes[node]['leaf1_neigh'] = clusters.get(node)

    # print('len di clusters (leaf_aggregator routers): ', clusters.len())
    # print(clusters)

    leaf_aggregator_count = 0

    distr_leaf_aggregator_type = Mydistribution()
    distr_leaf_aggr_leaf1_num = Mydistribution()
    for n, attr in G.nodes.data():
        if attr['type'] == 'ri':
            if 'leaf1_neigh' in attr:
                leaf_aggregator_count = leaf_aggregator_count + 1
                distr_leaf_aggregator_type.add(attr['class'])
                distr_leaf_aggr_leaf1_num.add(attr['leaf1_neigh'])

    # print ('internal leaf_aggregator routers : ', leaf_aggregator_count)

    # print (distr_leaf_aggregator_type)
    safety_check(clusters.len(), leaf_aggregator_count,
                 "len di clusters (leaf_aggregator routers)", "internal leaf_aggregator routers")

    # print (distr_leaf_aggr_leaf1_num)
    safety_check(distr_leaf_aggr_leaf1_num.count(), distr_leaf_aggregator_type.count(),
                 "distr_leaf_aggr_leaf1_num.count()", "distr_leaf_aggregator_type.count()")

    return distr_leaf_aggregator_type, distr_leaf_aggr_leaf1_num


def safety_check(left, right, left_description, right_description, relative_tol=1e-4):
    if not (math.isclose(left, right, rel_tol=relative_tol)):
        logging.error("ERROR {} : {} IS DIFFERENT FROM {} : {}".format(
            left_description, left, right_description, right))
        #print ("ERROR avg_ri_pp_ifs : {} different from distribution average : {}".format(float(ri_pp_ifs)/ri_count,ri_pp_ifs_distribution.mean()))
        raise Exception("Inconsistent {} vs {}".format(
            left_description, right_description))


def f3(myfloat):
    return '{:.3f}'.format(myfloat)


def graph_statistics(G, lcc_only, links_type, min_lcc_cov, approx, no_output=False):
    """
    Analyse graph, adds attributes to nodes, classify nodes and returns TSV formatted data
    it adds attributes to the nodes of G as follows
    if a node is a switch
    G.nodes[sw]['ri_count'] : internal routers connected to sw
    G.nodes[sw]['re_count'] : external routers connected to sw

    G.nodes[n]['class'] : 'leaf1'/'leaf2'/'default'/'border'
    G.nodes[n]['ext_neighs'] : external neighbors
    """
    start_time = time.time()

    global column_names
    global data_tsv
    global columns_details

    i = 0

    # count of AS routers
    ri_count = 0

    # count of non-AS routers, non-AS routers are added because
    # the links in the dataset include non-AS routers
    re_count = 0

    # iteraters over the adjacencies to evaluate
    # graph properties
    # n is source node, nbrs is a dict of (neighbor:link attr)
    for n, nbrs in G.adj.items():
        i = i + 1
        node_type = G.nodes[n]['type']
        if node_type == 'ri':
            ri_count = ri_count + 1
        elif node_type == 're':
            re_count = re_count + 1
        elif node_type == 'sw':
            sw_count = sw_count + 1

    logging.info("adjacencies analysis completed at {} seconds".format(
        time.time()-start_time))

    lcc_cov = G.graph['largest_cc_size'] / (ri_count + re_count)

    if no_output:
        return

    ############################
    # returns 1 TSV formatted and 2 JSON formatted strings
    #
    # first string is column names
    # second string is TSV data
    # third string is column descritpion

    reset_tsv()

    add_to_tsv('as_number', G.graph['as_number'], 'AS number')
    add_to_tsv('ds_nodes', G.graph['dataset_nodes'],
               'AS nodes in the full dataset')

    if not 'largest_cc_size' in G.graph:
        G.graph['largest_cc_size'] = 0

    add_to_tsv('largest_cc_size', G.graph['largest_cc_size'],
               'size of largest connected component (number of AS nodes, non-AS nodes and switches')
    if links_type == "ma" or links_type == "intra_ma":
        add_to_tsv('largest_cc_coverage', G.graph['largest_cc_size'] / (ri_count + re_count + sw_count),
                   'fraction of largest connected component vs size of graph (number of AS nodes, non-AS nodes and switches')
    else:
        add_to_tsv('largest_cc_coverage', G.graph['largest_cc_size'] / (ri_count + re_count),
                   'fraction of largest connected component vs size of graph (number of AS nodes and non-AS nodes')


    return [column_names, data_tsv, columns_details]


def batch_create_graph_from_file_and_analyze(as_number_list, output_tsv_filename, file_col_names, file_col_desc,
                                             input_dir, links_type, output_dir, lcc_only, min_lcc_cov, approx, eval_conn_comp=True, Directed=False):
    """
    create the networkx graph, analyze it, write TSV and readable text
    output_tsv_filename is the name of the TSV output file 
    (the data is appended, the file is not overwritten)
    it contains the information of all AS
    as_number_list is a list of strings like '34577'
    file_col_names is the name of the JSON file with column names
    file_col_desc is a JSON file with columns description
    the readable text is written on standard out, so it can
    be redirected on a file with '>' redirection
    or better with '| tee output_file.txt'

    if eval_conn_comp is true (default), the attribute 
    G.graph['largest_cc_size'] is added to G
    """
    output_tsv_filepath = os.path.join(output_dir, output_tsv_filename)
    col_names = col_details = []
    with open(output_tsv_filepath, 'a', encoding="utf8") as out_file:
        for as_number in as_number_list:
            start_time = time.time()
            if Directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            logging.info('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            logging.info('AS NUMBER: {}'.format(as_number))
            create_graph_from_files(as_number, G, input_dir, links_type)
            logging.info("Graph built in {} seconds".format(time.time()-start_time))
            start_time = time.time()
            if eval_conn_comp:
                G.graph['largest_cc_size'] = len(
                    max(nx.connected_components(G), key=len))
            returned_stats = graph_statistics(
                G, lcc_only, links_type, min_lcc_cov, approx)
            if len(returned_stats) == 3:
                col_names, dat_tsv, col_details = returned_stats
                logging.info("Graph analyzed in {} seconds\nwriting to file...".format(
                    time.time()-start_time))
                out_file.write(dat_tsv + '\n')
            else:
                logging.info("Graph not analyzed, {} seconds passed".format(
                    time.time()-start_time))
    with open(file_col_names, 'w', encoding="utf8") as out_file:
        out_file.write(json.dumps(col_names) + '\n')
    with open(file_col_desc, 'w', encoding="utf8") as out_file:
        out_file.write(json.dumps(col_details) + '\n')

def batch_create_graph_from_file_and_analyze_parall(as_number):
    fileHandler = logging.FileHandler("{0}/{1}.log".format(LOG_PATH, str(os.getpid()) + "_" + LOG_F_NAME))
    fileHandler.setFormatter(logFormatter)
    rootLogger.handlers.pop()
    rootLogger.addHandler(fileHandler)

    eval_conn_comp=True
    Directed=False
    output_tsv_filename = "analysis.tsv"
    file_col_names = "c_names.json"
    file_col_desc = "c_desc.json"
    input_dir, output_dir, stats_dir, min_size, single_as, links_type, lcc_only, min_lcc_cov, approx, parall, max_size = parse()
    output_tsv_filepath = os.path.join(output_dir, str(os.getpid()) + "_" + output_tsv_filename)
    col_names = col_details = []
    with open(output_tsv_filepath, 'a', encoding="utf8") as out_file:
        start_time = time.time()
        if Directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        logging.info('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        logging.info('AS NUMBER: {}'.format(as_number))
        create_graph_from_files(as_number, G, input_dir, links_type)
        logging.info("Graph built in {} seconds".format(time.time()-start_time))
        start_time = time.time()
        if eval_conn_comp:
            G.graph['largest_cc_size'] = len(
                max(nx.connected_components(G), key=len))
        returned_stats = graph_statistics(
            G, lcc_only, links_type, min_lcc_cov, approx)
        if len(returned_stats) == 3:
            col_names, dat_tsv, col_details = returned_stats
            logging.info("Graph analyzed in {} seconds\nwriting to file...".format(
                time.time()-start_time))
            out_file.write(dat_tsv + '\n')
        else:
            logging.info("Graph not analyzed, {} seconds passed".format(
                time.time()-start_time))
    with open(file_col_names, 'w', encoding="utf8") as out_file:
        out_file.write(json.dumps(col_names) + '\n')
    with open(file_col_desc, 'w', encoding="utf8") as out_file:
        out_file.write(json.dumps(col_details) + '\n')


def run_all(input_dir, output_dir, stats_dir, min_size, single_as, links_type, lcc_only, min_lcc_cov, approx, max_size):
    # get list of ASes to be analyzed
    if single_as is None:
        as_list = read_as_list(stats_dir, min_size, max_size)
    else:
        as_list = [single_as]
    start_time = time.time()
    batch_create_graph_from_file_and_analyze(as_list, "analysis.tsv", "c_names.json",
                                             "c_desc.json", input_dir, links_type, output_dir, lcc_only, min_lcc_cov, approx)
    execution_time = time.time() - start_time
    logging.info("Total execution time: {}".format(execution_time))

def run_all_parall(stats_dir, min_size, max_size):
    as_list = read_as_list(stats_dir, min_size, max_size)
    start_time = time.time()
    with Pool() as p:
        p.map(batch_create_graph_from_file_and_analyze_parall, as_list)
    execution_time = time.time() - start_time
    logging.info("Total execution time: {}".format(execution_time))

if __name__ == "__main__":
    input_dir, output_dir, stats_dir, min_size, single_as, links_type, lcc_only, min_lcc_cov, approx, parall, max_size = parse()
    if parall and not single_as:
        run_all_parall(stats_dir, min_size, max_size)
    else:
        run_all(input_dir, output_dir, stats_dir, min_size,
                single_as, links_type, lcc_only, min_lcc_cov, approx, max_size)
