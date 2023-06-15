"""
Obtain the client utilization of the individual cohorts and sum them up.
Also obtain statistics for the FL run.
This script requires a base directory name.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('root_models_dir')
args = parser.parse_args()

total_peers = int(args.root_models_dir.split("_")[1])

with open("../../data/client_utilizations.csv", "w") as out_file:
    out_file.write("group,time,total_peers,peers_training\n")

    for group in ["cohorts", "fl"]:
        cohort_utilizations = {
            "clients_training": defaultdict(lambda: 0),
            "bytes_up": defaultdict(lambda: 0),
            "bytes_down": defaultdict(lambda: 0),
            "train_time": defaultdict(lambda: 0),
            "network_time": defaultdict(lambda: 0),
        }
        filepaths = glob.glob("../../data/%s_c*" % args.root_models_dir) if group == "cohorts" else ["../../data/%s_dfl" % args.root_models_dir]
        for full_filepath in filepaths:
            if group == "cohorts":
                dir_name = os.path.basename(full_filepath)
                cohort_index = int(dir_name.split("_")[-2][1:])
                print("Parsing client utilizations of cohort %d..." % cohort_index)

            with open(os.path.join(full_filepath, "activities.csv")) as accuracies_file:
                csvreader = csv.reader(accuracies_file)
                next(csvreader)
                for row in csvreader:
                    time = int(row[0])
                    clients_training = int(row[3])
                    bytes_up = int(row[4])
                    bytes_down = int(row[5])
                    train_time = float(row[6])
                    network_time = float(row[6])

                    cohort_utilizations[time]["clients_training"] += clients_training
                    cohort_utilizations[time]["bytes_up"] += bytes_up
                    cohort_utilizations[time]["bytes_down"] += bytes_down
                    cohort_utilizations[time]["train_time"] += train_time
                    cohort_utilizations[time]["network_time"] += network_time

        for time, utilization_info in cohort_utilizations.items():
            out_file.write("%s,%d,%d,%d,%d,%d,%f,%f\n" % (group, time, total_peers,
                                                          utilization_info["clients_training"],
                                                          utilization_info["bytes_up"],
                                                          utilization_info["bytes_down"],
                                                          utilization_info["train_time"],
                                                          utilization_info["network_time"]))
