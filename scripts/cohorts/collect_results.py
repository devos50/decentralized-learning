"""
Collect and print all results, for different combinations of alpha, cohorts and seeds.
"""
import os

results = []

COMBINATIONS = []
for cohorts in [1, 2, 4, 8, 16, 32, 64, 128, 200]:
    for seeds in range(90, 95):
        for alpha in ["1.0", "0.3", "0.1"]:
            COMBINATIONS.append((alpha, cohorts, seeds))


for alpha, cohorts, seed in COMBINATIONS:
    print("Collecting results for a=%s, c=%d, s=%d" % (alpha, cohorts, seed))

    # Obtain the final accuracy
    best_accuracy = 0
    best_loss = 0
    distill_file_name = "output_distill_c%d_s%d_a%s.log" % (cohorts, seed, alpha)
    if not os.path.exists(distill_file_name):
        print("Distill file %s does not exist!" % distill_file_name)
        continue

    with open(distill_file_name, "r") as distill_file:
        lines = list(distill_file.readlines())
        if cohorts > 1:
            for line in lines:
                if line.startswith("INFO:distiller:Accuracy of student model after"):
                    parts = line.strip().split(" ")
                    accuracy = float(parts[-4][:-1])
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_loss = float(parts[-3])
        else:
            # We should be looking for the accuracy of the single student model
            for line in lines:
                if line.startswith("INFO:distiller:Accuracy of teacher model 0"):
                    parts = line.strip().split(" ")
                    best_accuracy = float(parts[-2][:-1])
                    best_loss = float(parts[-1])

    # Obtain the experiment times + resource usages
    stats_file_name = "data/n_200_cifar10_dirichlet%s00000_sd%d_ct%d_dfl/activities.csv" % (alpha, seed, cohorts)
    if not os.path.exists(stats_file_name):
        print("Stats file %s does not exist!" % stats_file_name)
        continue

    with open(stats_file_name, "r") as stats_file:
        lines = list(stats_file.readlines())
        parts = lines[-1].split(",")
        exp_time = int(parts[0])
        comm_cost = int(parts[4])
        train_time = int(float(parts[6]))
        results.append((cohorts, seed, alpha, best_accuracy, best_loss, exp_time, train_time, comm_cost))

# Write all the results away
with open("results.csv", "w") as results_file:
    results_file.write("dataset,cohorts,seed,alpha,accuracy,loss,tta,rta,bytes\n")
    for result in results:
        results_file.write("cifar10,%d,%d,%s,%f,%f,%d,%d,%d\n" % result)
