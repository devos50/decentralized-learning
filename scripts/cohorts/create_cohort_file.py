import argparse
import logging
import os
import sys
import time

from accdfl.core.datasets import create_dataset
from accdfl.core.session_settings import SessionSettings, LearningSettings

from sklearn.cluster import KMeans
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cohort_creator")

parser = argparse.ArgumentParser()
parser.add_argument('num_peers', type=int)
parser.add_argument('cohorts', type=int)
parser.add_argument('--method', type=str, default="uniform", choices=["uniform", "data"])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--partitioner', type=str, default="iid")
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args(sys.argv[1:])

if not args.output:
    args.output = "cohorts_n%d_c%d_s%d_%s.txt" % (args.num_peers, args.cohorts, args.seed, args.method)


# Determine the number of samples per client
learning_settings = LearningSettings(
    learning_rate=0,
    momentum=0,
    weight_decay=0,
    batch_size=512,
    local_steps=0,  # Not used in our training
)

full_settings = SessionSettings(
    dataset=args.dataset,
    work_dir="",
    learning=learning_settings,
    participants=["a"],
    all_participants=["a"],
    target_participants=args.num_peers,
    partitioner=args.partitioner,
    alpha=args.alpha,
)

if full_settings.dataset in ["cifar10", "mnist", "fashionmnist", "svhn"]:
    train_dir = args.data_dir
else:
    train_dir = os.path.join(args.data_dir, "per_user_data", "train")

client_data = {}
for peer_id in range(args.num_peers):
    start_time = time.time()
    dataset = create_dataset(full_settings, peer_id, train_dir=train_dir)
    logger.info("Creating dataset for peer %d took %f sec.", peer_id, time.time() - start_time)
    samples_per_peer = [0] * dataset.get_num_classes()
    for a, (b, clsses) in enumerate(dataset.get_trainset(500, shuffle=False)):
        for cls in clsses:
            samples_per_peer[cls] += 1
    client_data[peer_id] = samples_per_peer
    logger.info("Samples per class for peer %d: %s", peer_id, samples_per_peer)


def cluster_uniform():
    # Calculate the base number of peers per cohort
    base_peers_per_cohort = args.num_peers // args.cohorts

    # Calculate the number of cohorts that will have an extra peer
    extra_peers = args.num_peers % args.cohorts

    # Initialize the index of the first peer to be assigned
    total_peers_in_cohorts = 0
    current_peer = 0

    cohorts = {}

    for cohort_ind in range(args.cohorts):
        # Calculate the number of peers for the current cohort
        peers_in_this_cohort = base_peers_per_cohort + (1 if cohort_ind < extra_peers else 0)
        total_peers_in_cohorts += peers_in_this_cohort

        # Calculate the range of peers for the current cohort
        begin_peer = current_peer
        end_peer = current_peer + peers_in_this_cohort
        cohorts[cohort_ind] = list(range(begin_peer, end_peer))

        # Update the index of the first peer for the next cohort
        current_peer = end_peer

    assert total_peers_in_cohorts == args.num_peers

    return cohorts


def cluster_on_data():
    client_ids = list(client_data.keys())
    feature_vectors = np.array(list(client_data.values()))

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=args.cohorts, random_state=0, n_init=10).fit(feature_vectors)

    # Get cluster assignments
    clusters = kmeans.predict(feature_vectors)

    # Create a dictionary to store the client IDs of each cohort
    cohorts = {i: [] for i in range(args.cohorts)}

    for i, client_id in enumerate(client_ids):
        cohorts[clusters[i]].append(client_id)

    return cohorts


cohorts = None

if args.method == "uniform":
    cohorts = cluster_uniform()
elif args.method == "data":
    cohorts = cluster_on_data()

# Count the number of data samples per cohort
totals_per_cohort = []
for cohort_ind in range(args.cohorts):
    totals = [0] * 10
    for peer_id_in_cohort in cohorts[cohort_ind]:
        for cls_idx, num_samples in enumerate(client_data[peer_id_in_cohort]):
            totals[cls_idx] += num_samples
    totals_per_cohort.append(totals)
    logger.info("Data samples in cohort %d: %s (total: %d)", cohort_ind, totals, sum(totals))

with open(args.output, "w") as cohorts_file:
    for cohort_ind, cohort_peers in cohorts.items():
        peers_str = "-".join(["%d" % i for i in cohort_peers])
        cohorts_file.write("%d,%s\n" % (cohort_ind, peers_str))
