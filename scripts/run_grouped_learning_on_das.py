import logging
import os
import stat
import subprocess

NUM_PEERS = 200
NUM_GROUPS = 20
NUM_GROUPS_PER_DAS_NODE = 4

logger = logging.getLogger("grouped-learning")

groups_queue = list(range(NUM_GROUPS))
job_nr = 0
processes = []

while groups_queue:
    logger.info("Scheduling new job on DAS - %d groups left", len(groups_queue))

    groups_on_this_node = []
    while groups_queue and len(groups_on_this_node) < NUM_GROUPS_PER_DAS_NODE:
        client = groups_queue.pop(0)
        groups_on_this_node.append(client)

    peers_per_group = NUM_PEERS // NUM_GROUPS
    train_cmds = ""
    for group in groups_on_this_node:
        active_participants = "%d-%d" % (group * peers_per_group, (group + 1) * peers_per_group)
        data_dir = "data_%d" % group
        train_cmd = "python3 -u simulations/dfl/cifar10.py --num-aggregators 1 --fix-aggregator --sample-size 10 --peers 200 \
--model resnet8 --batch-size 16 --learning-rate 0.0025 --weight-decay 0.0003 --accuracy-logging-interval 1 --train-device-name \"cuda:0\" \
--accuracy-device-name \"cuda:0\" --bypass-model-transfers --store-best-model --partitioner dirichlet --active-participants %s --alpha 0.1 \
--duration 600000 --datadir %s > g%d.log 2>&1 &" % (active_participants, data_dir, group)
        train_cmds += train_cmd + "\n"

    bash_file_name = "run_%d.sh" % job_nr
    with open(bash_file_name, "w") as bash_file:
        bash_file.write("""#!/bin/bash
module load cuda11.7/toolkit/11.7
source /home/spandey/venv3/bin/activate
cd %s
export PYTHONPATH=%s:%s/pyipv8

%s
wait""" % (os.getcwd(), os.getcwd(), os.getcwd(), train_cmds))
    st = os.stat(bash_file_name)
    os.chmod(bash_file_name, st.st_mode | stat.S_IEXEC)

    out_file_path = os.path.join(os.getcwd(), "out_%d.log" % job_nr)
    cmd = "prun -t 24:00:00 -np 1 -o %s %s" % (out_file_path, os.path.join(os.getcwd(), bash_file_name))
    logger.debug("Command: %s", cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes.append((p, cmd))
    job_nr += 1


for p, cmd in processes:
    p.wait()
    logger.info("Command %s completed!", cmd)
    if p.returncode != 0:
        raise RuntimeError("Training subprocess exited with non-zero code %d" % p.returncode)


# Combine the accuracies
with open("accuracies.csv", "w") as out_file:
    out_file.write("dataset,group,time,peer,round,accuracy,loss\n")
    for job_idx in range(NUM_GROUPS):
        with open(os.path.join("data_%d" % job_idx, "accuracies.csv")) as acc_file:
            parsed_header = False
            for line in acc_file.readlines():
                if not parsed_header:
                    parsed_header = True
                    continue

                out_file.write(line)
