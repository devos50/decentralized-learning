NUM_PEERS = 100
COHORTS = 10

# TODO for now, we randomly assign peers to cohorts
peers_per_cohort = NUM_PEERS // COHORTS
with open("../../data/cohorts.txt", "w") as cohorts_file:
    for cohort_ind in range(COHORTS):
        begin_peer = cohort_ind * peers_per_cohort
        end_peer = (cohort_ind + 1) * peers_per_cohort
        peers_str = "-".join(["%d" % i for i in list(range(begin_peer, end_peer))])
        cohorts_file.write("%d,%s\n" % (cohort_ind, peers_str))
