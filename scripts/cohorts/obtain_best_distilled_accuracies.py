"""
This script reads the distill_accuracies_*.csv files and obtains the best accuracies.
"""
import glob

for full_filepath in glob.glob("../../data/distill_accuracies_*.csv"):
    with open(full_filepath) as in_file:
        last_line = list(in_file.readlines())[-1].strip()
        parts = last_line.split(",")
        distill_time = int(parts[0])
        best_accuracy = float(parts[4])

        print("cifar10,cohorts,%d,0,0,%f,0" % (distill_time, best_accuracy))
