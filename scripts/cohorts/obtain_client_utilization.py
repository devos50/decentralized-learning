"""
Obtain the client utilization of the individual cohorts and sum them up.
Also obtain statistics for the FL run.
This script requires a base directory name.
"""
import argparse
import csv
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('root_models_dir')
args = parser.parse_args()

total_peers = int(args.root_models_dir.split("_")[0][1:])

with open("../../data/client_utilizations.csv", "w") as out_file:
    out_file.write("group,time,total_peers,peers_training\n")

    cohort_utilizations = {}
    for group in ["cohorts", "fl"]:
        pattern = "../../data/%s_c*" % args.root_models_dir if group == "cohorts" else "../../data/%s" % args.root_models_dir
        for full_filepath in glob.glob(pattern):
            dir_name = os.path.basename(full_filepath)

            if group == "cohorts":
                cohort_index = int(dir_name.split("_")[-2][1:])
                print("Parsing client utilizations of cohort %d..." % cohort_index)

            with open(os.path.join(full_filepath, "activities.csv")) as accuracies_file:
                csvreader = csv.reader(accuracies_file)
                next(csvreader)
                for row in csvreader:
                    time = int(row[0])
                    clients_training = int(row[3])

                    if time not in cohort_utilizations:
                        cohort_utilizations[time] = 0
                    cohort_utilizations[time] += clients_training

        for time, clients_training in cohort_utilizations.items():
            out_file.write("%s,%d,%d\n" % (group, time, clients_training))
