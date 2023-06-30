NUM_PEERS = 3550
COHORTS = 20

# Calculate the base number of peers per cohort
base_peers_per_cohort = NUM_PEERS // COHORTS

# Calculate the number of cohorts that will have an extra peer
extra_peers = NUM_PEERS % COHORTS

# Initialize the index of the first peer to be assigned
total_peers_in_cohorts = 0
current_peer = 0

with open("../../data/cohorts.txt", "w") as cohorts_file:
    for cohort_ind in range(COHORTS):
        # Calculate the number of peers for the current cohort
        peers_in_this_cohort = base_peers_per_cohort + (1 if cohort_ind < extra_peers else 0)
        total_peers_in_cohorts += peers_in_this_cohort

        # Calculate the range of peers for the current cohort
        begin_peer = current_peer
        end_peer = current_peer + peers_in_this_cohort

        print("Peers in cohort %d: %d" % (cohort_ind, peers_in_this_cohort))

        # Generate the peers string for the current cohort
        peers_str = "-".join(["%d" % i for i in range(begin_peer, end_peer)])

        # Write the peers string to the file
        cohorts_file.write("%d,%s\n" % (cohort_ind, peers_str))

        # Update the index of the first peer for the next cohort
        current_peer = end_peer

assert total_peers_in_cohorts == NUM_PEERS
