import networkx as nx

node_id = 1
x, y = 5, 10
is_intersection_value = True
G = nx.DiGraph()


# G.add_node(node_id, x=x, y=y, is_intersection=is_intersection_value)
# print(G.nodes[node_id])
G.add_nodes_from([(1,{"coord":(5,2), "is_intersection":True}), (2,{"coord":(6,3), "is_intersection":True})])
print(G.nodes[1]["coord"])

G.add_edges_from([(1,2, {"weight":15, "behavior":"turn_left"})])
print(G.edges[1,2])