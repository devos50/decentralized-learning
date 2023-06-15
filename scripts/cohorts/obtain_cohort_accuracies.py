"""
Obtain the accuracies of the individual cohorts, given a base directory name.
"""
import argparse
import csv
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('root_models_dir')  # The root directory containing the directories with data of individual (cohort) sessions
args = parser.parse_args()

with open("../../data/cohort_accuracies.csv", "w") as out_file:
    out_file.write("cohort,time,accuracy,loss\n")

    for full_filepath in glob.glob("../../data/%s_c*" % args.root_models_dir):
        dir_name = os.path.basename(full_filepath)
        cohort_index = int(dir_name.split("_")[-2][1:])
        print("Parsing accuracies of cohort %d..." % cohort_index)

        # Parse accuracies
        with open(os.path.join(full_filepath, "accuracies.csv")) as accuracies_file:
            csvreader = csv.reader(accuracies_file)
            next(csvreader)
            for row in csvreader:
                time = float(row[2])
                accuracy = float(row[5])
                loss = float(row[6])
                out_file.write("%d,%f,%f,%f\n" % (cohort_index, time, accuracy, loss))
