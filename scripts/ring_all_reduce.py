import numpy as np

a = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
]

print("Ref:", np.sum(a, axis=0))


def ring_all_reduce_simple(x):
    x = [
        np.array(data)    
        for data in x
    ]
    total_participants = len(x)
    for round_id in range(total_participants):
        for client_id in range(total_participants):
            i_to_send = (client_id - round_id) % total_participants
            receiving_client_id = (client_id + 1) % total_participants
            x[receiving_client_id][i_to_send] += x[client_id][i_to_send]
            x[client_id][i_to_send] = 0
    for round_id in range(total_participants - 1):
        for client_id in range(total_participants):
            i_to_send = (client_id - round_id) % total_participants
            receiving_client_id = (client_id + 1) % total_participants
            x[receiving_client_id][i_to_send] = x[client_id][i_to_send]
    return x

def ring_all_reduce(x):
    x = [
        np.array(data)    
        for data in x
    ]
    total_participants = len(x)
    for round_id in range(2 * total_participants - 1):
        for client_id in range(total_participants):
            i_to_send = (client_id - round_id) % total_participants
            receiving_client_id = (client_id + 1) % total_participants
            x[receiving_client_id][i_to_send] += x[client_id][i_to_send]
            if round_id < total_participants:
                x[client_id][i_to_send] = 0
    return x

print("Simple implementation:", ring_all_reduce_simple(a))
print("Short implementation:", ring_all_reduce(a))