import os
import pandas as pd
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

RANK_DIR = "rank"
FILE_IN = "rank_unordered.tsv"
FILE_OUT = "rank_ordered.csv"

file_path_in = os.path.join(RANK_DIR, FILE_IN)
df = pd.read_csv(file_path_in, sep="\t", names=["rank"]).rename_axis("node").sort_values(by='rank', ascending=False)
file_path_out = os.path.join(RANK_DIR, FILE_OUT)
df.to_csv(file_path_out)