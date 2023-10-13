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
        self.StateBuffer = car_states
        self.PathBuffer = self.init_path()            # contains node like [[1,2], [5,10,12]]
        self.check_path()

        print("Path buffer check: ", self.PathBuffer)
        self.WptsBuffer = self.init_wpts()            # contains waypoints like [ [(x,y,yaw)...], [], [] ]

    def init_path(self):
            PathBuffer = []
            for i in range(self.n_car):
                path = nx.astar_path(self.G, self.start_nodes[i], self.goal_nodes[i], heuristic=None, weight="weight")
                PathBuffer.append(path)
            
            # final_buffer = self.check_path(PathBuffer)
            # return final_buffer
            return PathBuffer

    def init_wpts(self):
        WptsBuffer = []
        for i in range(self.n_car):
            path = self.PathBuffer[i].copy()
            waypoints = []
            for n in range(len(path)-1):
                edge_data = self.G.get_edge_data(path[n],path[n+1])
                waypoints += edge_data["waypoints"]
            for k in range(len(waypoints)-1):
                x1, y1, _ = waypoints[k]
                x2, y2, _ = waypoints[k+1]
                if (x1-x2)**2+(y1-y2)**2 <= 0.04:
                    print(" Cheking waypoints distance! distance < 0.2: ", waypoints[k], waypoints[k+1])
            WptsBuffer.append(waypoints)
        return WptsBuffer

    def check_path(self):
        """
        Check if the path contains more than two nodes,
        if not, append another node to replan the path.
        """
        print("Before Checking PathBuffer: ",self.PathBuffer)
        for path in self.PathBuffer:
            i = 0
            while len(path) <= 2:
                print(" ***************************** In check path, doing adding")
                new_goal_node = self.sample_one_node()
                while new_goal_node in path:
                    new_goal_node = self.sample_one_node()
                self.goal_nodes[i] = new_goal_node
                path = nx.astar_path(self.G, path[0], new_goal_node, heuristic=None, weight="weight")
                self.PathBuffer[i] = path
            i += 1
        print("After Checking PathBuffer: ", self.PathBuffer)


    def sample_one_node(self):
        node = random.randint(1, len(self.node_list))
        return node
    
    def localize_to_road(self, state, car_id):
        # Make sure no change to path
        in_edge = [-1, -1]
        path = self.PathBuffer[car_id]
        print("path: ", path)
        print("car state: ",state)
        for n in range(len(path)-1):
            edge_points = self.G.get_edge_data(path[n],path[n+1])["waypoints"]
            print("section points: ", edge_points)
            closest_index = self.find_closest_waypoint(state[0], state[1], edge_points)
            s,d = self.frenet_transform(state[0], state[1], edge_points)
            if n == 0 and s < 0 :
                in_edge = [-1, path[n]]
                return s,d, in_edge
            if s >= 0 and s < self.G.get_edge_data(path[n],path[n+1])["weight"]:
                in_edge = [path[n], path[n+1]]
                return s,d,in_edge
            print(" In edge: ", path[n],path[n+1])
            print("the closest index: ", closest_index)
            print(" print s,d : ", s,d)
        if in_edge == [-1,-1]:
            raise ValueError (" Cannot localize to the path !")
                
    def path_state_update(self, in_edge, car_id):
        path = self.PathBuffer[car_id]
        if in_edge[0] in path:
            index = path.index(in_edge[0])
            if index > 0:
                del path[:index]
        else:
            raise ValueError (" Edge not in the path, in path state update!")

    def find_closest_waypoint(self, x, y, waypoints):
        min_distance = float('inf')
        closest_index = 0

        for i, (wx, wy, _) in enumerate(waypoints):
            distance = math.sqrt((x-wx)**2 + (y-wy)**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index
        
    # def project_to_segment(self, x, y, start, end):
    #     dx = end[0] - start[0]
    #     dy = end[1] - start[1]
    #     t = ((x-start[0])*dx + (y-start[1])*dy) / (dx*dx + dy*dy)
    #     if t < 0: t = 0
    #     if t > 1: t = 1
    #     x_proj = start[0] + t*dx
    #     y_proj = start[1] + t*dy
    #     return x_proj, y_proj

    def frenet_transform(self,x, y, waypoints):
        closest_idx = self.find_closest_waypoint(x, y, waypoints)
        wx, wy, wyaw = waypoints[closest_idx]

        # Calculate direction vector of the path
        dx_path = math.cos(wyaw)
        dy_path = math.sin(wyaw)

        # Calculate the vector from the waypoint to the car
        dx_car = x - wx
        dy_car = y - wy

        # Project onto the direction vector (for s)
        s_proj = dx_car * dx_path + dy_car * dy_path
        s = closest_idx * 0.5 + s_proj

        # Project onto the normal of the direction vector (for d)
        dx_norm = -dy_path
        dy_norm = dx_path
        d_proj = dx_car * dx_norm + dy_car * dy_norm
        d = d_proj

        return s, d

    # def frenet_transform(self, x, y, waypoints):
    #     closest_idx = self.find_closest_waypoint(x, y, waypoints)
        
    #     # Calculate s
    #     s = closest_idx * 0.5
    #     if closest_idx < len(waypoints) - 1:
    #         x_proj, y_proj = self.project_to_segment(x, y, waypoints[closest_idx], waypoints[closest_idx + 1])
    #         s += math.sqrt((x_proj - waypoints[closest_idx][0])**2 + (y_proj - waypoints[closest_idx][1])**2)
        
    #     # Calculate d
    #     if closest_idx < len(waypoints) - 1:
    #         d = math.sqrt((x - x_proj)**2 + (y - y_proj)**2)
    #         # Check side
    #         side = (x - waypoints[closest_idx][0]) * (waypoints[closest_idx + 1][1] - waypoints[closest_idx][1]) - (y - waypoints[closest_idx][1]) * (waypoints[closest_idx + 1][0] - waypoints[closest_idx][0])
    #         if side < 0: d = -d
    #     else:
    #         d = 0
    #     return s, d

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
start_nodes = [40, 37, 28, 7, 39]
goal_nodes = [38, 5, 4, 18, 36]
# Initalize Cars
cars = []
car_states = []
path_buffer = []
print(" *********************************")
print("node list: ", node_list)
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
    # Add paths
    path = nx.astar_path(G, start_nodes[i], goal_nodes[i], heuristic=None, weight="weight")
    path_buffer.append(path)
TM = TrafficManager(node_list=node_list, edge_list=edge_list, G = G, n_car = N_CAR,
                    car_states=car_states, car_info = CAR_INFO, start_nodes=start_nodes, goal_nodes=goal_nodes,
                    wpts_dist=WAYPOINT_DISTANCE)

print(TM.WptsBuffer[0])
print(" car[0] state: ",cars[0].state)
edge_wpts = [(21.0, 43.25, 0.0), (21.5, 43.25, 0.0), (22.0, 43.25, 0.0), (22.5, 43.25, 0.0), (23.0, 43.25, 0.0), (23.5, 43.25, 0.0)]
# TM.localize_to_road(cars[0].state, car_id = 0)
TM.localize_to_road(cars[0].state, car_id = 0)
TM.localize_to_road([37.901,43.784, 0.,0.,0.], car_id = 0)

# [(37.6, 43.25, 0.0), (38.1, 43.25, 0.0), (38.6, 43.25, 0.0), (39.1, 43.25, 0.0), (39.6, 43.25, 0.0),
