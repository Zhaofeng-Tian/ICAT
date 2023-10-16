import numpy as np
import networkx as nx
import random
from topo import *
from car import build_car,get_car_param

class TrafficManager:


    def __init__(self, node_list, edge_list, G, n_car, car_states, car_info, start_nodes, goal_nodes = [], wpts_dist = 0.5):
        self.wpts_dist = wpts_dist
        self.node_list = node_list
        self.node_range = [i+1 for i in range(len(node_list))]
        self.edge_list = edge_list
        self.G = G
        self.n_car = n_car
        self.start_nodes = start_nodes
        self.goal_nodes = goal_nodes
        self.car_info = car_info
        self.PathBuffer = self.init_path()            # contains node like [[1,2], [5,10,12]]
        print("Before Checking PathBuffer: ",self.PathBuffer)
        for i in range(self.n_car):
            self.check_path(i)
        print("After Path buffer check: ", self.PathBuffer)
        self.WptsBuffer = self.init_wpts()            # contains waypoints like [ [(x,y,yaw)...], [], [] ]
        self.StateBuffer = self.init_states(car_states)
        print("Cheking state init", self.StateBuffer)
        self.FrenetBuffer = []
        self.EdgeOccupancy = []
        # Behavior option "speed keeping", "car following", "stop"
        # since cars spawn in safety point, so init as "speed keeping"
        self.BehaviorBuffer =["speed keeping" for i in range(self.n_car)]

    def init_states(self, states):
        StateBuffer = []
        assert self.n_car == len(states), " Car number not equal to len of states!"
        for i in range(self.n_car):
            car_state = {}
            x,y,yaw,v,w = states[i]
            car_state["current_point"] = (x,y,yaw)
            car_state["linear_speed"] = v
            car_state["angular_speed"] = w
            car_state["ahead_point"] = (x,y,yaw)
            StateBuffer.append(car_state)
        return StateBuffer


    def init_path(self):
            PathBuffer = []
            for i in range(self.n_car):
                path = nx.astar_path(self.G, self.start_nodes[i], self.goal_nodes[i], heuristic=None, weight="weight")
                PathBuffer.append(path)

            return PathBuffer

    def init_wpts(self):
        WptsBuffer = []
        for i in range(self.n_car):
            path = self.PathBuffer[i].copy()
            waypoints = []
            for n in range(len(path)-1):
                edge_data = self.G.get_edge_data(path[n],path[n+1])
                waypoints.append(edge_data["waypoints"])
            WptsBuffer.append(waypoints)
        return WptsBuffer

    def update_path_wpts(self, path, i):
        waypoints = []
        for n in range(len(path)-1):
            edge_data = self.G.get_edge_data(path[n],path[n+1])
            waypoints.append(edge_data["waypoints"])
        self.WptsBuffer[i] = waypoints
        
    def check_path(self, id, if_update_wpts = False):
        """
        Check if the path contains more than two nodes,
        if not, append another node to replan the path.
        """
        path = self.PathBuffer[id]
        while len(path) <= 2:
            print(" ***************************** In check path, doing adding")
            new_goal_node = sample_one_node(len(self.node_list))
            while new_goal_node in path:
                new_goal_node = sample_one_node(len(self.node_list))
            self.goal_nodes[id] = new_goal_node
            path = [path[0]] + nx.astar_path(self.G, path[1], new_goal_node, heuristic=None, weight="weight")
            self.PathBuffer[id] = path
            if if_update_wpts:
                self.update_path_wpts(path, id)

    def localize_to_road(self, state, car_id):
        # Make sure no change to path
        in_edge = (-1, -1)
        path = self.PathBuffer[car_id]
        print("path: ", path)
        print("car state: ",state)
        for n in range(len(path)-1):
            edge_points = self.G.get_edge_data(path[n],path[n+1])["waypoints"]
            # print("section points: ", edge_points)
            closest_index = find_closest_waypoint(state[0], state[1], edge_points)
            closest_point = edge_points[closest_index]
            s,d = frenet_transform(state[0], state[1], edge_points)
            if n == 0 and s < 0 :
                in_edge = (-1, path[n])
                print(" In edge: ", path[n],path[n+1])
                return s,d, in_edge, closest_index, closest_point
            if s >= 0 and s < self.G.get_edge_data(path[n],path[n+1])["weight"]:
                in_edge = (path[n], path[n+1])
                print(" In edge: ", path[n],path[n+1])
                return s,d,in_edge, closest_index, closest_point
            
            print("the closest index: ", closest_index)
            print(" print s,d : ", s,d)
        if in_edge == [-1,-1]:
            raise ValueError (" Cannot localize to the path !")
      
    def path_pop(self, in_edge, car_id, if_pop_wpts= True):
        path = self.PathBuffer[car_id]
        wpts = self.WptsBuffer[car_id]
        if in_edge[0] in path:
            index = path.index(in_edge[0])
            if index > 0:
                orig_len = len(wpts)
                del path[:index]
                self.PathBuffer[car_id] = path
                if if_pop_wpts:
                    del wpts[:index]
                    self.WptsBuffer[car_id] = wpts
                    assert len(self.WptsBuffer[car_id])!=orig_len, "Fail to pop out passed waypoints!"
        else:
            raise ValueError (" Edge not in the path, in path state update!")

    def get_look_ahead_point(self,clst_index,look_ahead_dist, id):
        u,v,w = self.PathBuffer[id][:3]
        ahead_in_edge = (u,v)
        n_ahead_indices = round(look_ahead_dist/self.wpts_dist)
        n1 = self.G.get_edge_data(u,v)["n_points"]
        ahead_points = self.WptsBuffer[id][0]+self.WptsBuffer[id][1]
        ahead_index = clst_index + n_ahead_indices
        if clst_index + n_ahead_indices >= n1-1:
            ahead_in_edge = (v,w)
            ahead_index = ahead_index - n1
        ahead_point = ahead_points[clst_index+n_ahead_indices]
        return ahead_in_edge, ahead_index, ahead_point
            

        
            
            
    """          
    car_state["current_point"] = (x,y,yaw)
    car_state["linear_speed"] = v
    car_state["angular_speed"] = w
    car_state["ahead_point"] = (x,y,yaw)
    """    

    def sd_occupancy_update(self,car_states):
        sd_coords = []
        occupied_edges = []
        for i in range(self.n_car):
            s,d,in_edge,clst_index, clst_point = self.localize_to_road(car_states[i],i)
            x,y,yaw,v,w = car_states[i]
            self.path_pop(in_edge,car_id=i)
            self.check_path(id=i, if_update_wpts=True)
            if dist(clst_point,self.StateBuffer["ahead_point"]) <= 0.5:
                ahead_in_edge, ahead_index, ahead_point = self.get_look_ahead_point()
            self.StateBuffer[i]["current_point"] = (x,y,yaw)
            self.StateBuffer[i]["linear_speed"] = v
            self.StateBuffer[i]["angular_speed"] = w
            self.StateBuffer[i]["sd"] = (s,d)
            self.StateBuffer[i]["in_edge"] = in_edge
            self.StateBuffer[i]["clst_index"] = clst_index
            self.StateBuffer[i]["clst_point"] = clst_point
            self.StateBuffer[i]["ahead_in_edge"] = ahead_in_edge
            self.StateBuffer[i]["ahead_index"] = ahead_index
            self.StateBuffer[i]["ahead_point"] = ahead_point
            
            sd_coords.append((s,d))
            occupied_edges.append(in_edge)
        self.FrenetBuffer = sd_coords
        self.EdgeOccupancy = occupied_edges



    def traffic_state_update(self):
        """
        Phase I: localize all agents and update path and wpts
        """
        car_states = self.get_car_state() # Euclidean state
        self.sd_occupancy_update(car_states) # Update PathBuffer, FrenetBuffer, EdgeOccupancy
        # self.check_path(if_update_wpts=True) # Update PathBuffer, WptsBuffer if len(path) <= 2
        """
        Phase II: plan behavior for all agents
        """
        edge_cache = {}
        # build edge cache
        for i in range(self.n_car):
            in_edge = self.EdgeOccupancy[i].copy()
            s,d = self.FrenetBuffer[i]
            if in_edge not in edge_cache:
                edge_cache[in_edge] = [(i,s,d)]
            else:
                edge_cache[in_edge].append((i,s,d))
        for in_edge in edge_cache:
            if len(edge_cache[in_edge])>1:
                sorted_data = sorted(edge_cache[in_edge], key=lambda x: x[1])
                for i in range(len(sorted_data)-1):
                    distance = sorted_data[i+1][1] - sorted_data[i][1]


WAYPOINT_DISTANCE = 0.5
N_CAR = 5
N_NODE = 42
CAR_PARAM = get_car_param()
CAR_INFO = {"hl":1.775, "hw": 1.0} # half length, half width of the car
DT = 0.1

node_list = get_node_list()
edge_list = get_edge_list(node_list)
G = build_graph(node_list, edge_list)
node_arr = [i for i in range(1, N_NODE+1)]
# Sample nodes for start and goal
nodes = random.sample(node_arr, 2 * N_CAR)
# start_nodes = nodes[:N_CAR]
# goal_nodes = nodes[N_CAR:]
# start_nodes[0] = 5
# goal_nodes[0] = 6
start_nodes = [3, 37, 28, 7, 39]
goal_nodes = [4, 5, 4, 18, 36]
# Initalize Cars
cars = []
car_states = []
path_buffer = []
print(" *********************************")
# print("node list: ", node_list)
print("start nodes: ", start_nodes)
print("goal nodes: ", goal_nodes)
for i in range(N_CAR):
    n = start_nodes[i]
    coord = node_list[n-1][1]["coord"]
    print("car {}, start nodes id {}, coord {} ".format(i,n,coord))
    print(coord)
    # Add cars
    car = build_car(i, CAR_PARAM, coord)
    cars.append(car)
    car_states.append(car.state)
    # Add paths
    path = nx.astar_path(G, start_nodes[i], goal_nodes[i], heuristic=None, weight="weight")
    path_buffer.append(path)


TM = TrafficManager(node_list=node_list, edge_list=edge_list, G = G, n_car = N_CAR,
                    car_states=car_states, car_info = CAR_INFO, start_nodes=start_nodes, goal_nodes=goal_nodes,
                    wpts_dist=WAYPOINT_DISTANCE)
# # ax = view_topo(node_list, edge_list, if_arrow=True)
# for edge_wpts in TM.WptsBuffer[0]:
#     print("edge_wpts: ", edge_wpts)
print(" car[0] state: ",cars[0].state)
edge_wpts = [(21.0, 43.25, 0.0), (21.5, 43.25, 0.0), (22.0, 43.25, 0.0), (22.5, 43.25, 0.0), (23.0, 43.25, 0.0), (23.5, 43.25, 0.0)]
# TM.localize_to_road(cars[0].state, car_id = 0)
TM.localize_to_road(cars[0].state, car_id = 0)
s,d, in_edge, closest_index = TM.localize_to_road([37.901,43.784, 0.,0.,0.], car_id = 0)
print("In edge: ", in_edge)

# [(37.6, 43.25, 0.0), (38.1, 43.25, 0.0), (38.6, 43.25, 0.0), (39.1, 43.25, 0.0), (39.6, 43.25, 0.0),
