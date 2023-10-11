from tracemalloc import start
import numpy as np

class TrafficManager:


    def __init__(self, node_list, edge_list, G, n_car, car_states, start_nodes, goal_nodes = []):
        self.node_list = node_list
        self.edge_list = edge_list
        self.G = G
        self.n_car = n_car
        self.s_nodes = start_nodes
        self.g_nodes = goal_nodes
        self.StateBuffer = car_states
        self.PathBuffer= []
