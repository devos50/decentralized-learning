"""
Convert the CSV file with the information of transfers into a directed graph that can be used for debugging.
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read CSV file
csv_file = "../data/n_100_cifar10/transfers.csv"
df = pd.read_csv(csv_file)

# Create a directed graph
G = nx.DiGraph()

# Replace node IDs with an index
node_ids = set()
for index, row in df.iterrows():
    node_ids.add(row['from'])
    node_ids.add(row['to'])

node_ids_list = list(node_ids)
for index, row in df.iterrows():
    df.loc[index, 'from'] = node_ids_list.index(row['from'])
    df.loc[index, 'to'] = node_ids_list.index(row['to'])

# Make node IDs unique by adding round num
for index, row in df.iterrows():
    df.loc[index, 'from'] = "%s_%d" % (row['from'], row['round'])
    if row['type'] == "aggregated":
        df.loc[index, 'to'] = "%s_%d" % (row['to'], row['round'] + 1)
    else:
        df.loc[index, 'to'] = "%s_%d" % (row['to'], row['round'])

# Add edges and nodes to the graph
labels = {}
for index, row in df.iterrows():
    from_node = row['from']
    to_node = row['to']
    labels[from_node] = from_node.split("_")[0]
    labels[to_node] = to_node.split("_")[0]
    round_num = row['round']
    edge_type = row['type']

    G.add_node(from_node)
    G.add_node(to_node)
    G.add_edge(from_node, to_node, round=round_num, style='dashed' if edge_type == 'trained' else 'solid')

# Spring layout
pos = nx.spring_layout(G, iterations=1)

# Draw nodes
nx.draw_networkx_nodes(G, pos)

# Draw edges
for edge in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], edge_color='black', style=edge[2]['style'])

# Draw node labels
nx.draw_networkx_labels(G, pos, labels=labels)

# Save graph to PDF
plt.axis('off')
plt.savefig('graph_output.pdf', format='pdf')