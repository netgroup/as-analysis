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
    parser.add_argument("-i", metavar="input_dir", default="data_2020_08", help="input dir containing per AS extracted data")
    parser.add_argument("-o", metavar="output_dir", default="analysis_2020_08", help="output dir to write analysis output")
    parser.add_argument("-s", metavar="stats_dir", default="stats", help="input dir to read statistics")
    parser.add_argument("--size", metavar="as_size", default="1000", help="minimum size of ASes to be considered")
    parser.add_argument("--singleas", metavar="single_as", help="AS number of single as to be considered")
    parser.add_argument("--links", metavar="links_type", default="fm", help="link types: ma fm intra_ma intra_fm", choices=["ma", "fm", "intra_ma", "intra_fm"])
    parser.add_argument("--minlcc", metavar="min_lcc_coverage", default="0.9", help="minimum LCC coverage of the AS for being analyzed")
    # TODO ############
    parser.add_argument("--lcc", action='store_true', help="only analyze LCC")
    parser.add_argument("--approx", metavar="approximate_results", default="auto", help="approximate results for average shortest path length (use for large topologies)", choices=["auto", "fixed", "false"])
    parser.add_argument("--approxsize", metavar="approximation_size", default="1000", help="minimum size of ASes to be considered")
    ###################
    args = parser.parse_args()
    return args.i, args.o, args.s, args.size, args.singleas, args.links, args.lcc, args.minlcc

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
                links[line[0]] = [line[1],line[2]]
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
    print("AS: {}".format(as_number))
    print("Number of nodes (AS routers): {}".format(len(as_nodes)))
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
    print("Number of links: {}".format(len(links)))
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
        print("ERROR {} : {} IS DIFFERENT FROM {} : {}".format(
            left_description, left, right_description, right))
        #print ("ERROR avg_ri_pp_ifs : {} different from distribution average : {}".format(float(ri_pp_ifs)/ri_count,ri_pp_ifs_distribution.mean()))
        raise Exception("Inconsistent {} vs {}".format(
            left_description, right_description))

def f3(myfloat):
    return '{:.3f}'.format(myfloat)


def graph_statistics(G, lcc_only, links_type, min_lcc_cov, no_output=False):
    """Analyse graph, adds attributes to nodes, classify nodes and returns TSV formatted data

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

    # count AS routers with only 1 internal (ri) neigh
    leaf1_count = 0

    # count AS routers with only 2 internal (ri) neighs
    leaf2_count = 0

    # count AS routers with at least one ext neigh
    border_count = 0

    # as routers : number of point-to-point interfaces
    ri_pp_ifs = 0
    ri_pp_ifs_distribution = Mydistribution()

    if links_type == "ma" or links_type == "intra_ma":
        # as routers : number of Multiple Access interfaces
        ri_ma_ifs = 0
        ri_ma_ifs_distribution = Mydistribution()

        # as_routers : number of total interfaces
        ri_tot_ifs = 0
        ri_tot_ifs_distribution = Mydistribution()

        # as routers : number of routers reachable over MA interfaces
        ri_ma_neighs = 0
        ri_ma_neighs_distribution = Mydistribution()

    # as_routers : number of total neighbors
    ri_tot_neighs = 0
    ri_tot_neighs_distribution = Mydistribution()

    # count of non-AS routers, non-AS routers are added because
    # the links in the dataset include non-AS routers
    re_count = 0

    # non-as routers : number of point-to-point interfaces
    re_pp_ifs = 0
    re_pp_ifs_distribution = Mydistribution()

    if links_type == "ma" or links_type == "intra_ma":
        # non-as routers : number of Multiple Access interfaces
        re_ma_ifs = 0
        re_ma_ifs_distribution = Mydistribution()

        # non-as_routers : number of total interfaces
        re_tot_ifs = 0
        re_tot_ifs_distribution = Mydistribution()

        # non-as routers : number of routers reachable over MA interfaces
        re_ma_neighs = 0
        re_ma_neighs_distribution = Mydistribution()

    # non-as_routers : number of total neighbors
    re_tot_neighs = 0
    re_tot_neighs_distribution = Mydistribution()

    if links_type == "ma" or links_type == "intra_ma":
        # count of switches
        sw_count = 0

        # switches: number of interfaces
        sw_ifs = 0
        sw_ifs_distribution = Mydistribution()

        # for each switch, count the number of ri and re nodes
        # adding the attributes 'ri_count' and 're_count'
        for n, attr in G.nodes.data():
            if attr['type'] == 'sw':
                sw_ri_count = 0
                sw_re_count = 0
                # print (n,attr)
                # print (G.adj[n])
                for node in G.adj[n].keys():
                    # print (node)
                    # print (G.nodes[node]['type'])
                    if G.nodes[node]['type'] == 'ri':
                        sw_ri_count = sw_ri_count + 1
                    else:
                        sw_re_count = sw_re_count + 1
                attr['ri_count'] = sw_ri_count
                attr['re_count'] = sw_re_count
                attr['fanout'] = sw_ri_count + sw_ri_count
            if attr['type'] == 'ri' or attr['type'] == 're':
                attr['fanout'] = len(G.adj[n])
        
        print("switch analysis completed at {} seconds".format(time.time()-start_time))

    # iteraters over the adjacencies to evaluate
    # graph properties
    # n is source node, nbrs is a dict of (neighbor:link attr)
    for n, nbrs in G.adj.items():
        i = i + 1
        # print (n, nbrs)
        # print (nbrs)
        # print (G[n])
        node_type = G.nodes[n]['type']
        # print(node_type)
        if node_type == 'ri':
            ri_count = ri_count + 1
            pp_ifs = 0
            ma_ifs = 0
            ma_neighs = 0

            # used to count neighs that are external ('re')
            ext_neighs = 0

            for node_id, link_attrs in nbrs.items():
                if link_attrs['type'] == 'pp':
                    pp_ifs = pp_ifs + 1
                    if G.nodes[node_id]['type'] == 're':
                        ext_neighs = ext_neighs + 1

                if link_attrs['type'] == 'ma':
                    ma_ifs = ma_ifs + 1
                    # print (G.degree[node_id])
                    # print (G.adj[node_id])
                    ma_neighs = ma_neighs + G.degree[node_id] - 1
                    ri_ma_neighs_distribution.add(G.degree[node_id] - 1)
                    ext_neighs = ext_neighs + G.nodes[node_id]['re_count']

            ri_pp_ifs = ri_pp_ifs + pp_ifs
            ri_pp_ifs_distribution.add(pp_ifs)

            if links_type == "ma" or links_type == "intra_ma":
                ri_ma_ifs = ri_ma_ifs + ma_ifs
                ri_ma_ifs_distribution.add(ma_ifs)

                ri_tot_ifs = ri_tot_ifs + pp_ifs + ma_ifs
                ri_tot_ifs_distribution.add(pp_ifs + ma_ifs)

                ri_ma_neighs = ri_ma_neighs + ma_neighs

            ri_tot_neighs = ri_tot_neighs + pp_ifs + ma_neighs
            ri_tot_neighs_distribution.add(pp_ifs + ma_neighs)

            G.nodes[n]['ext_neighs'] = ext_neighs
            G.nodes[n]['class'] = 'default'
            # print ('ext_neighs', ext_neighs)
            # print (pp_ifs, ma_neighs)
            if ext_neighs == 0:
                if (pp_ifs + ma_neighs) == 1:
                    G.nodes[n]['class'] = 'leaf1'
                    leaf1_count = leaf1_count + 1
                if (pp_ifs + ma_neighs) == 2:
                    G.nodes[n]['class'] = 'leaf2'
                    leaf2_count = leaf2_count + 1
            else:
                G.nodes[n]['class'] = 'border'
                border_count = border_count + 1

        elif node_type == 're':
            re_count = re_count + 1
            pp_ifs = 0
            ma_ifs = 0
            ma_neighs = 0
            for node_id, link_attrs in nbrs.items():
                if link_attrs['type'] == 'pp':
                    pp_ifs = pp_ifs + 1
                if link_attrs['type'] == 'ma':
                    ma_ifs = ma_ifs + 1
                    ma_neighs = ma_neighs + G.degree[node_id] - 1
                    re_ma_neighs_distribution.add(G.degree[node_id] - 1)

            re_pp_ifs = re_pp_ifs + pp_ifs
            re_pp_ifs_distribution.add(pp_ifs)

            if links_type == "ma" or links_type == "intra_ma":
                re_ma_ifs = re_ma_ifs + ma_ifs
                re_ma_ifs_distribution.add(ma_ifs)

                re_tot_ifs = re_tot_ifs + pp_ifs + ma_ifs
                re_tot_ifs_distribution.add(pp_ifs + ma_ifs)

                re_ma_neighs = re_ma_neighs + ma_neighs

            re_tot_neighs = re_tot_neighs + pp_ifs + ma_neighs
            re_tot_neighs_distribution.add(pp_ifs + ma_neighs)

        elif node_type == 'sw':
            # print (n, nbrs)
            sw_count = sw_count + 1
            sw = 0
            for node_id, link_attrs in nbrs.items():
                sw = sw + 1
            sw_ifs = sw_ifs + sw
            sw_ifs_distribution.add(sw)

    print("adjacencies analysis completed at {} seconds".format(time.time()-start_time))

    lcc_cov = G.graph['largest_cc_size'] / (ri_count + re_count)
    if lcc_cov < float(min_lcc_cov):
        print("LCC coverage is: {}\nlower than minimum desired: {}\nAS not analyzed".format(lcc_cov, min_lcc_cov))
        return []

    # clusterize leaf nodes

    dis_leaf_aggr_type, dis_leaf_aggr_leaf1_num = clusterize_leaf_nodes(G)
    leaf_agg_count = dis_leaf_aggr_type.count()

    print("leaf nodes clusterizing completed at {} seconds".format(time.time()-start_time))
    
    # more graph stats with LCC
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc)

    print("LCC extracted at {} seconds".format(time.time()-start_time))

    if not lcc_only:
        density = nx.density(G)
        print("density calculated at {} seconds".format(time.time()-start_time))
        assortativity = nx.degree_pearson_correlation_coefficient(G)    # faster assortativity algorithm
        print("assortativity calculated at {} seconds".format(time.time()-start_time))
        transitivity = nx.transitivity(G)   # global clustering coeff
        print("transitivity calculated at {} seconds".format(time.time()-start_time))

    # largest connected component
    core_number = nx.core_number(G_lcc)
    avg_coreness = sum(core_number.values()) / len(core_number)
    graph_coreness = 0
    graph_coreness = max([c for c in core_number.values()])
    core_order = 0
    for coreness in core_number.values():
        if coreness == graph_coreness:
            core_order = core_order + 1
    print("coreness stats calculated at {} seconds".format(time.time()-start_time))
    density_lcc = nx.density(G_lcc)
    print("density of LCC calculated at {} seconds".format(time.time()-start_time))
    assortativity_lcc = nx.degree_pearson_correlation_coefficient(G_lcc)
    print("assortativity of LCC calculated at {} seconds".format(time.time()-start_time))
    transitivity_lcc = nx.transitivity(G_lcc)
    print("transitivity of LCC calculated at {} seconds".format(time.time()-start_time))
    avg_shortest_path_len = nx.average_shortest_path_length(G_lcc)
    print("avg shortest path length of LCC calculated at {} seconds".format(time.time()-start_time))
    # max_degree = max([d for n, d in G.degree()])

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
    add_to_tsv('ds_links', G.graph['dataset_links'],
               'links (PP or multi-access) in the full dataset')

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

    add_to_tsv('as_routers', ri_count, 'AS routers')
    add_to_tsv('non_as_routers', re_count, 'non-AS routers')
    if links_type == "ma" or links_type == "intra_ma":
        add_to_tsv('switches', sw_count, 'switches')

    m_add_to_tsv([('leaf1', leaf1_count, 'AS routers with only 1 internal neighbor'),
                  ('leaf2', leaf2_count, 'AS routers with only 2 internal neighbor'),
                  ('default', ri_count - leaf1_count - leaf2_count - border_count,
                   'AS routers with no external neighbors and more than 2 int neighbors'),
                  ('border', border_count, 'AS routers with external neighbors')])

    if ri_count != 0:
        m_add_to_tsv([('ri_avg_pp_ifs', f3(ri_pp_ifs / ri_count), 'Average of PP interfaces for AS routers'),
                    ('ri_max_pp_ifs', ri_pp_ifs_distribution.maxval(),
                    'Max of PP interfaces for AS routers'),
                    ('ri_tot_pp_ifs', ri_pp_ifs, 'Total PP interfaces of AS routers')])

        if links_type == "ma" or links_type == "intra_ma":
            m_add_to_tsv([('ri_avg_ma_ifs', f3(ri_ma_ifs / ri_count), 'Average of MA interfaces for AS routers'),
                        ('ri_max_ma_ifs', ri_ma_ifs_distribution.maxval(),
                        'Max of MA interfaces for AS routers'),
                        ('ri_tot_ma_ifs', ri_ma_ifs, 'Total MA interfaces of AS routers')])

            m_add_to_tsv([('ri_avg_tot_ifs', f3(ri_tot_ifs / ri_count), 'Average of total interfaces for AS routers'),
                        ('ri_max_tot_ifs', ri_tot_ifs_distribution.maxval(),
                        'Max of total interfaces for AS routers'),
                        ('ri_tot_tot_ifs', ri_tot_ifs, 'Total interfaces of AS routers')])
    else:
        m_add_to_tsv([('ri_avg_pp_ifs', 'N/A', 'Average of PP interfaces for AS routers'),
                    ('ri_max_pp_ifs', ri_pp_ifs_distribution.maxval(),
                    'Max of PP interfaces for AS routers'),
                    ('ri_tot_pp_ifs', ri_pp_ifs, 'Total PP interfaces of AS routers')])

        if links_type == "ma" or links_type == "intra_ma":
            m_add_to_tsv([('ri_avg_ma_ifs', 'N/A', 'Average of MA interfaces for AS routers'),
                        ('ri_max_ma_ifs', ri_ma_ifs_distribution.maxval(),
                        'Max of MA interfaces for AS routers'),
                        ('ri_tot_ma_ifs', ri_ma_ifs, 'Total MA interfaces of AS routers')])

            m_add_to_tsv([('ri_avg_tot_ifs', 'N/A', 'Average of total interfaces for AS routers'),
                        ('ri_max_tot_ifs', ri_tot_ifs_distribution.maxval(),
                        'Max of total interfaces for AS routers'),
                        ('ri_tot_tot_ifs', ri_tot_ifs, 'Total interfaces of AS routers')])

    if links_type == "ma" or links_type == "intra_ma":
        if ri_ma_ifs != 0:
            m_add_to_tsv(
                [('ri_avg_ma_neighs', f3(ri_ma_neighs / ri_ma_ifs), 'Average of neighbors on each MA interface for AS routers'),
                ('ri_max_ma_neighs', ri_ma_neighs_distribution.maxval(),
                'Max of neighbors on a MA interface for AS routers'),
                ('ri_tot_ma_neighs', ri_ma_neighs, 'Total neighbors on MA interfaces for AS routers')])
        else:
            m_add_to_tsv(
                [('ri_avg_ma_neighs', 'N/A', 'Average of neighbors on each MA interface for AS routers'),
                ('ri_max_ma_neighs', ri_ma_neighs_distribution.maxval(),
                'Max of neighbors on a MA interface for AS routers'),
                ('ri_tot_ma_neighs', ri_ma_neighs, 'Total neighbors on MA interfaces for AS routers')])

    if ri_count != 0:
        m_add_to_tsv([('ri_avg_tot_neighs', f3(ri_tot_neighs / ri_count), 'Average of neighbors for AS routers'),
                    ('ri_max_tot_neighs', ri_tot_neighs_distribution.maxval(),
                    'Max of neighbors for AS routers'),
                    ('ri_tot_tot_neighs', ri_tot_neighs, 'Total neighbors for AS routers')])
    else:
        m_add_to_tsv([('ri_avg_tot_neighs', 'N/A', 'Average of neighbors for AS routers'),
                    ('ri_max_tot_neighs', ri_tot_neighs_distribution.maxval(),
                    'Max of neighbors for AS routers'),
                    ('ri_tot_tot_neighs', ri_tot_neighs, 'Total neighbors for AS routers')])

    m_add_to_tsv([('ri_leaf1_aggr', leaf_agg_count, 'Leaf1 aggregator internal routers'),
                  ('avg_leaf1_per_leaf1_aggr', dis_leaf_aggr_leaf1_num.mean(),
                   'Mean of leaf1 node per leaf1 aggregator AS router'),
                  ('max_leaf1_per_leaf1_aggr', dis_leaf_aggr_leaf1_num.maxval(),
                   'Max of leaf1 node per leaf1 aggregator AS router')])

    if re_count != 0:
        m_add_to_tsv([('re_avg_pp_ifs', f3(re_pp_ifs / re_count), 'Average of PP interfaces for non-AS routers'),
                    ('re_max_pp_ifs', re_pp_ifs_distribution.maxval(),
                    'Max of PP interfaces for non-AS routers'),
                    ('re_tot_pp_ifs', re_pp_ifs, 'Total PP interfaces of non-AS routers')])

        if links_type == "ma" or links_type == "intra_ma":
            m_add_to_tsv([('re_avg_ma_ifs', f3(re_ma_ifs / re_count), 'Average of MA interfaces for non-AS routers'),
                        ('re_max_ma_ifs', re_ma_ifs_distribution.maxval(),
                        'Max of MA interfaces for non-AS routers'),
                        ('re_tot_ma_ifs', re_ma_ifs, 'Total MA interfaces of non-AS routers')])

            m_add_to_tsv([('re_avg_tot_ifs', f3(re_tot_ifs / re_count), 'Average of total interfaces for non-AS routers'),
                        ('re_max_tot_ifs', re_tot_ifs_distribution.maxval(),
                        'Max of total interfaces for non-AS routers'),
                        ('re_tot_tot_ifs', re_tot_ifs, 'Total interfaces of non-AS routers')])
    else:
        m_add_to_tsv([('re_avg_pp_ifs', 'N/A', 'Average of PP interfaces for non-AS routers'),
                    ('re_max_pp_ifs', re_pp_ifs_distribution.maxval(),
                    'Max of PP interfaces for non-AS routers'),
                    ('re_tot_pp_ifs', re_pp_ifs, 'Total PP interfaces of non-AS routers')])

        if links_type == "ma" or links_type == "intra_ma":
            m_add_to_tsv([('re_avg_ma_ifs', 'N/A', 'Average of MA interfaces for non-AS routers'),
                        ('re_max_ma_ifs', re_ma_ifs_distribution.maxval(),
                        'Max of MA interfaces for non-AS routers'),
                        ('re_tot_ma_ifs', re_ma_ifs, 'Total MA interfaces of non-AS routers')])

            m_add_to_tsv([('re_avg_tot_ifs', 'N/A', 'Average of total interfaces for non-AS routers'),
                        ('re_max_tot_ifs', re_tot_ifs_distribution.maxval(),
                        'Max of total interfaces for non-AS routers'),
                        ('re_tot_tot_ifs', re_tot_ifs, 'Total interfaces of non-AS routers')])

    if links_type == "ma" or links_type == "intra_ma":
        if re_ma_ifs != 0:
            m_add_to_tsv([('re_avg_ma_neighs', f3(re_ma_neighs / re_ma_ifs),
                        'Average of neighbors on each MA interface for non-AS routers'),
                        ('re_max_ma_neighs', re_ma_neighs_distribution.maxval(),
                        'Max of neighbors on a MA interface for non-AS routers'),
                        ('re_tot_ma_neighs', re_ma_neighs, 'Total neighbors on MA interfaces for non-AS routers')])
        else:
            m_add_to_tsv([('re_avg_ma_neighs', 'N/A',
                        'Average of neighbors on each MA interface for non-AS routers'),
                        ('re_max_ma_neighs', re_ma_neighs_distribution.maxval(),
                        'Max of neighbors on a MA interface for non-AS routers'),
                        ('re_tot_ma_neighs', re_ma_neighs, 'Total neighbors on MA interfaces for non-AS routers')])

    if re_count != 0:
        m_add_to_tsv([('re_avg_tot_neighs', f3(re_tot_neighs / re_count), 'Average of neighbors for non-AS routers'),
                    ('re_max_tot_neighs', re_tot_neighs_distribution.maxval(),
                    'Max of neighbors for non-AS routers'),
                    ('re_tot_tot_neighs', re_tot_neighs, 'Total neighbors for non-AS routers')])
    else:
        m_add_to_tsv([('re_avg_tot_neighs', 'N/A', 'Average of neighbors for non-AS routers'),
                    ('re_max_tot_neighs', re_tot_neighs_distribution.maxval(),
                    'Max of neighbors for non-AS routers'),
                    ('re_tot_tot_neighs', re_tot_neighs, 'Total neighbors for non-AS routers')])

    if links_type == "ma" or links_type == "intra_ma":
        if sw_count != 0:
            m_add_to_tsv([('sw_avg_ifs', f3(sw_ifs / sw_count), 'Average of interfaces for switches'),
                        ('sw_max_ifs', sw_ifs_distribution.maxval(),
                        'Max of interfaces for switches'),
                        ('sw_tot_ifs', sw_ifs, 'Total interfaces of switches')])
        else:
            m_add_to_tsv([('sw_avg_ifs', 'N/A', 'Average of interfaces for switches'),
                        ('sw_max_ifs', sw_ifs_distribution.maxval(),
                        'Max of interfaces for switches'),
                        ('sw_tot_ifs', sw_ifs, 'Total interfaces of switches')])

    add_to_tsv('ri_pp_ifs_dis', ri_pp_ifs_distribution.to_json(),
               'Distribution of PP interfaces for AS routers')
    if links_type == "ma" or links_type == "intra_ma":
        add_to_tsv('ri_ma_ifs_dis', ri_ma_ifs_distribution.to_json(),
                'Distribution of MA interfaces for AS routers')
        add_to_tsv('ri_tot_ifs_dis', ri_tot_ifs_distribution.to_json(),
                'Distribution of all interfaces for AS routers')
        add_to_tsv('ri_ma_neighs_dis', ri_ma_neighs_distribution.to_json(),
                'Distribution of neighbors for MA interface for AS routers')
    add_to_tsv('ri_tot_neighs_dis', ri_tot_neighs_distribution.to_json(),
               'Distribution of neighbors for AS routers')

    add_to_tsv('dis_leaf1_aggr_type', dis_leaf_aggr_type.to_json(),
               'Distribution of types for leaf1 aggregator AS routers')
    add_to_tsv('dis_leaf_aggr_leaf1_num', dis_leaf_aggr_leaf1_num.to_json(
    ), 'Distribution of leaf1 number for leaf1 aggretator AS routers')

    add_to_tsv('re_pp_ifs_dis', re_pp_ifs_distribution.to_json(),
               'Distribution of PP interfaces for non-AS routers')
    if links_type == "ma" or links_type == "intra_ma":
        add_to_tsv('re_ma_ifs_dis', re_ma_ifs_distribution.to_json(),
                'Distribution of MA interfaces for non-AS routers')
        add_to_tsv('re_tot_ifs_dis', re_tot_ifs_distribution.to_json(),
                'Distribution of all interfaces for non-AS routers')
        add_to_tsv('re_ma_neighs_dis', re_ma_neighs_distribution.to_json(),
                'Distribution of neighbors for MA interface for non-AS routers')
    add_to_tsv('re_tot_neighs_dis', re_tot_neighs_distribution.to_json(),
               'Distribution of neighbors for non-AS routers')
    if links_type == "ma" or links_type == "intra_ma":
        add_to_tsv('sw_ifs_dis', sw_ifs_distribution.to_json(),
                'Distribution of interfaces for switches')
    
    ############################

    if not lcc_only:
        add_to_tsv('density', density, 'Graph density')
        add_to_tsv('assortativity', assortativity, 'Graph assortativity')
        add_to_tsv('transitivity', transitivity, 'Graph transitivity, or global clustering coefficient')
    add_to_tsv('avg_coreness', avg_coreness, 'Average nodes coreness')
    add_to_tsv('graph_coreness', graph_coreness, 'Graph coreness (max node coreness)')
    add_to_tsv('core_order', core_order, 'Number of nodes in graph core (subgraph containing nodes with maximum coreness)')
    add_to_tsv('density_lcc', density_lcc, 'Graph density of largest connected component')
    add_to_tsv('assortativity_lcc', assortativity_lcc, 'Graph assortativity of largest connected component')
    add_to_tsv('transitivity_lcc', transitivity_lcc, 'Graph transitivity, or global clustering coefficient of largest connected component')
    add_to_tsv('avg_shortest_path_len', avg_shortest_path_len, 'Graph average shortest path length')

    # print (column_names)
    # print (data_tsv)
    # print (columns_details)

    return [column_names, data_tsv, columns_details]


def batch_create_graph_from_file_and_analyze(as_number_list, output_tsv_filename, file_col_names, file_col_desc,
                                             input_dir, links_type, output_dir, lcc_only, min_lcc_cov, eval_conn_comp=True, Directed=False):
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
    output_tsv_filepath = os.path.join(output_dir,output_tsv_filename)
    col_names = col_details = []
    with open(output_tsv_filepath, 'a', encoding="utf8") as out_file:
        for as_number in as_number_list:
            start_time = time.time()
            if Directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('AS NUMBER: {}'.format(as_number))
            create_graph_from_files(as_number, G, input_dir, links_type)
            print("Graph built in {} seconds".format(time.time()-start_time))
            start_time = time.time()
            if eval_conn_comp:
                G.graph['largest_cc_size'] = len(
                    max(nx.connected_components(G), key=len))
            returned_stats = graph_statistics(G, lcc_only, links_type, min_lcc_cov)
            if len(returned_stats) == 3:
                col_names, dat_tsv, col_details = returned_stats
                print("Graph analyzed in {} seconds\nwriting to file...".format(time.time()-start_time))
                out_file.write(dat_tsv + '\n')
            else:
                print("Graph not analyzed, {} seconds passed".format(time.time()-start_time))
    with open(file_col_names, 'w', encoding="utf8") as out_file:
        out_file.write(json.dumps(col_names) + '\n')
    with open(file_col_desc, 'w', encoding="utf8") as out_file:
        out_file.write(json.dumps(col_details) + '\n')

def run_all(input_dir, output_dir, stats_dir, min_size, single_as, links_type, lcc_only, min_lcc_cov):
    # get list of ASes to be analyzed
    if single_as is None:
        as_list = read_as_list(stats_dir, min_size)
    else:
        as_list = [single_as]
    start_time = time.time()
    batch_create_graph_from_file_and_analyze(as_list, "analysis.tsv", "c_names.json", "c_desc.json", input_dir, links_type, output_dir, lcc_only, min_lcc_cov)
    execution_time = time.time() - start_time
    print("Total execution time: {}".format(execution_time))

if __name__ == "__main__":
    input_dir, output_dir, stats_dir, min_size, single_as, links_type, lcc_only, min_lcc_cov = parse()
    run_all(input_dir, output_dir, stats_dir, min_size, single_as, links_type, lcc_only, min_lcc_cov)