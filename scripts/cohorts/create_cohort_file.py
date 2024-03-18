import argparse
import logging
import os
import sys
import time
from random import Random

from accdfl.core.datasets import create_dataset
from accdfl.core.session_settings import SessionSettings, LearningSettings

from scipy.stats import entropy
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cohort_creator")

parser = argparse.ArgumentParser()
parser.add_argument('num_peers', type=int)
parser.add_argument('cohorts', type=int)
parser.add_argument('--method', type=str, default="uniform", choices=["uniform", "data", "class"])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--partitioner', type=str, default="iid")
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args(sys.argv[1:])

if not args.output:
    args.output = "cohorts_%s_n%d_c%d_s%d_%s.txt" % (args.dataset, args.num_peers, args.cohorts, args.seed, args.method)


if os.path.exists(args.output):
    print("Output file %s already exists - not creating it again." % args.output)
    exit(0)


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
    seed=args.seed,
)

if full_settings.dataset in ["cifar10", "mnist", "fashionmnist", "svhn"]:
    train_dir = args.data_dir
else:
    train_dir = os.path.join(args.data_dir, "per_user_data", "train")

client_data = {}
num_cls = 0
for peer_id in range(args.num_peers):
    start_time = time.time()
    dataset = create_dataset(full_settings, peer_id, train_dir=train_dir)
    logger.info("Creating dataset for peer %d took %f sec.", peer_id, time.time() - start_time)
    if num_cls == 0:
        num_cls = dataset.get_num_classes()
    samples_per_peer = [0] * dataset.get_num_classes()
    for a, (b, clsses) in enumerate(dataset.get_trainset(500, shuffle=False)):
        for cls in clsses:
            samples_per_peer[cls] += 1
    client_data[peer_id] = samples_per_peer

    assert sum(samples_per_peer) > 0, "Peer %d has no samples!" % peer_id

    logger.info("Samples per class for peer %d: %s", peer_id, samples_per_peer)


def cluster_uniform():
    all_peers = list(range(0, args.num_peers))
    rand = Random(args.seed)
    rand.shuffle(all_peers)

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
        cohorts[cohort_ind] = all_peers[begin_peer:end_peer]

        # Update the index of the first peer for the next cohort
        current_peer = end_peer

    assert total_peers_in_cohorts == args.num_peers

    return cohorts


def cluster_on_data():
    client_distributions = np.array(list(client_data.values()))

    # Normalize the client distributions
    client_distributions = client_distributions / client_distributions.sum(axis=1, keepdims=True)

    # Define the Jensen-Shannon divergence
    def JSD(P, Q):
        _P = P / np.linalg.norm(P, ord=1)
        _Q = Q / np.linalg.norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    # We calculate pairwise Jensen-Shannon divergence
    distances = pairwise_distances(client_distributions, metric=JSD)

    # Using K-Medoids clustering with precomputed distances
    kmedoids = KMedoids(n_clusters=args.cohorts, metric='precomputed', random_state=0)

    # Fit and predict the clusters
    clusters = kmedoids.fit_predict(distances)

    # Create a dictionary to store the client IDs of each cohort
    cohorts = {i: [] for i in range(args.cohorts)}

    for peer_id, peer_cohort in enumerate(clusters):
        cohorts[peer_cohort].append(peer_id)

    return cohorts


def cluster_on_class():
    num_classes = len(client_data[0])
    assert args.cohorts == num_classes, "When clustering on class, the number of cohorts and classes must be equal!"
    cohorts = {i: [] for i in range(args.cohorts)}
    for peer_id, peer_data in client_data.items():
        # Get the dominant class of this peer
        cls_with_most = -1
        num_most = 0
        for cls_idx in range(num_classes):
            if peer_data[cls_idx] > num_most:
                num_most = peer_data[cls_idx]
                cls_with_most = cls_idx

        cohorts[cls_with_most].append(peer_id)

    return cohorts


cohorts = None
if args.method == "uniform":
    cohorts = cluster_uniform()
elif args.method == "data":
    cohorts = cluster_on_data()
elif args.method == "class":
    cohorts = cluster_on_class()


# Count the number of data samples per cohort
totals_per_cohort = []
total_peers = 0
for cohort_ind in range(args.cohorts):
    totals = [0] * num_cls
    for peer_id_in_cohort in cohorts[cohort_ind]:
        for cls_idx, num_samples in enumerate(client_data[peer_id_in_cohort]):
            totals[cls_idx] += num_samples
    totals_per_cohort.append(totals)
    peers_in_cohort: int = len(cohorts[cohort_ind])
    total_peers += peers_in_cohort
    logger.info("Data samples in cohort %d: %s (total: %d), peers: %d", cohort_ind, totals, sum(totals), peers_in_cohort)

assert total_peers == args.num_peers, "Not all peers have been assigned to a cohort!"

with open(args.output, "w") as cohorts_file:
    for cohort_ind, cohort_peers in cohorts.items():
        peers_str = "-".join(["%d" % i for i in cohort_peers])
        cohorts_file.write("%d,%s\n" % (cohort_ind, peers_str))
