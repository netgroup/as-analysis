'''
clean itdk dataset for processing
'''
import argparse
import os
from functools import cmp_to_key

AS_NODES_FILE = "midar-iff.nodes.as"
LINKS_FILE = "midar-iff.links"
GEO_FILE = "midar-iff.nodes.geo"
AS_FILE_CLEAN = "nodes_as.csv"
LINKS_FILE_MA_CLEAN = "links_ma.txt"
LINKS_FILE_FM_CLEAN = "links_fm.csv"
GEO_FILE_CLEAN = "nodes_geo.csv"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_dir", default="itdk_2020_08", help="input dir containing itdk dataset")
    parser.add_argument("-o", metavar="output_dir", default="itdk_2020_08_clean", help="output dir to write statistics")
    args = parser.parse_args()
    return args.i, args.o

def clean_nodes(input_dir, output_dir, input_file=AS_NODES_FILE, ouput_file=AS_FILE_CLEAN):
    in_path = os.path.join(input_dir, input_file)
    out_path = os.path.join(output_dir, ouput_file)
    with open(in_path, 'r', encoding="utf8") as file_r:
        with open(out_path, 'w', encoding="utf8") as file_w:
            file_w.write("node,AS\n")
            for line in file_r:
                tokens = line.split()
                node = tokens[1].strip(':')
                as_number = tokens[2]
                file_w.write("{},{}\n".format(node,as_number))

def compare(a, b):
    if a[0] == '#' and b[0] == '#':
        return 0
    elif a[0] != '#' and b[0] == '#':
        return 1
    elif a[0] == '#' and b[0] != '#':
        return -1
    a = int(a.split()[1][1:-1])
    b = int(b.split()[1][1:-1])
    if a>b:
        return 1
    elif a<b:
        return -1
    elif a==b:
        return 0
    else:
        print("error")

def geo_sort(input_dir, input_file):
    file_path_geo = os.path.join(input_dir, input_file)
    with open(file_path_geo, 'r', encoding="utf8") as geo_file:
        lines = geo_file.readlines()
        lines.sort(key=cmp_to_key(compare))
        return lines

def clean_geo(input_dir, output_dir, input_file=GEO_FILE, ouput_file=GEO_FILE_CLEAN):
    lines = geo_sort(input_dir, input_file)
    file_path_geo_clean = os.path.join(output_dir, ouput_file)
    with open(file_path_geo_clean, 'w', encoding="utf8") as out_file:
        out_file.write("node,continent,country\n")
        for line in lines:
            if line[0] != '#':
                tokens = line.split('\t')
                node = tokens[0].split()[1].strip(':')
                continent = tokens[1]
                country = tokens[2]
                out_file.write("{},{},{}\n".format(node,continent,country))

def clean_links(input_dir, output_dir, input_file=LINKS_FILE, ouput_file=LINKS_FILE_MA_CLEAN):
    in_path = os.path.join(input_dir, input_file)
    out_path = os.path.join(output_dir, ouput_file)
    with open(in_path, 'r', encoding="utf8") as file_r:
        with open(out_path, 'w', encoding="utf8") as file_w:
            for line in file_r:
                if line[0] != '#':
                    tokens = line.split(":  ")
                    link = tokens[0].split()[1]
                    nodes = tokens[1].split()
                    for node in nodes:
                        node = node.split(':')[0]
                        link = link+' '+node
                    file_w.write(link+'\n')

def ma_links_to_full_mesh(input_dir, input_file=LINKS_FILE_MA_CLEAN, ouput_file=LINKS_FILE_FM_CLEAN):
    """
    reads links from file and creates a new one where multiple access
    links become a full mesh of point to point links
    """
    in_path = os.path.join(input_dir, input_file)
    out_path = os.path.join(input_dir, ouput_file)
    with open(in_path, 'r', encoding="utf8") as file_r:
        with open(out_path, 'w', encoding="utf8") as file_w:
            file_w.write("link,node1,node2\n")
            for line in file_r:
                tokens = line.split()
                if len(tokens) > 3:
                    nodes = tokens[1:]
                    nodes_left = nodes
                    link = tokens[0]
                    i = 1
                    for first_node in nodes:
                        nodes_left = nodes_left[1:]
                        if len(nodes_left) >= 1:
                            for second_node in nodes_left:
                                file_w.write("{}.{},{},{}\n".format(link,str(i),first_node,second_node))
                                i = i+1
                elif len(tokens) == 3:
                    file_w.write(','.join(tokens)+'\n')
                else:
                    print("Wrong line format: "+str(line))

def run_all(input_dir, output_dir):
    print("nodes as file...")
    clean_nodes(input_dir, output_dir)
    print("nodes geo file...")
    clean_geo(input_dir, output_dir)
    print("links file...")
    clean_links(input_dir, output_dir)
    print("doing links full mesh...")
    ma_links_to_full_mesh(output_dir)

if __name__ == "__main__":
    input_dir, output_dir = parse()
    run_all(input_dir, output_dir)