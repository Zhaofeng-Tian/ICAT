from car import build_car, get_car_param
from nav_gym.obj.robot.robot import CarRobot
from topo import *
import random
from simulation import TrafficManager

N_CAR = 5
N_NODE = 42
CAR_PARAM = get_car_param()
DT = 0.1

def init_sim():
    node_list = get_node_list()
    edge_list = get_edge_list(node_list)
    G = build_graph(node_list, edge_list)
    node_arr = [i for i in range(1, N_NODE+1)]
    # Sample nodes for start and goal
    nodes = random.sample(node_arr, 2 * N_CAR)
    start_nodes = nodes[:N_CAR]
    goal_nodes = nodes[N_CAR:]
    # Initalize Cars
    cars = []
    car_states = []
    path_buffer = []
    for i in range(N_CAR):
        n = start_nodes[i]
        coord = node_list[n][1]["coord"]
        print(coord)
        # Add cars
        car = build_car(i, CAR_PARAM, coord)
        cars.append(car)
        # Add paths
        path = nx.astar_path(G, start_nodes[i], goal_nodes[i], heuristic=None, weight="weight")
        path_buffer.append(path)
    TM = TrafficManager(node_list=node_list, edge_list=edge_list, G = G, n_car = N_CAR,
                        car_states=car_states, start_nodes=start_nodes, goal_nodes=goal_nodes)
    return cars, TM 

def step(cars, TM):



