import numpy as np
import networkx as nx
import math
import json
from bezier import *

# G = nx.DiGraph()
# G.add_nodes_from([(1,{"coord":(5,2), "itsc":True}), (2,{"coord":(6,3), "itsc":True})])
# print(G.nodes[1]["coord"])

# G.add_edges_from([(1,2, {"weight":15, "behavior":"turn_left"})])
# print(G.edges[1,2])
WAYPOINT_DISTANCE = 0.5
BEZIER_CONTROL_PARAMETER = 0.6

def build_graph():
    node_list = get_node_list()
    edge_list = get_edge_list(node_list)
    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    return G

def A_star_path(G,ns,ng): # node_start, node_goal
    path = nx.astar_path(G, ns, ng, heuristic=None, weight="weight")
    cost = nx.astar_path_length(G, ns, ng, heuristic=None, weight="weight")
    # print(path)
    return path, cost

def get_node_list():

    node_list= [
        (1,  {"coord":(11.0,3.25), "pre":[(10,'l')], "next":[(2,'s')],"itsc": False}),
        (2,  {"coord":(21,3.25), "pre":[(1,'s')], "next":[(3,'s'),(19,'l')], "itsc": False}),
        (3,  {"coord":(38,3.25), "pre":[(2,'s'),(15,'l')], "next":[(4,'s')], "itsc": False}),
        (4,  {"coord":(49.0,3.25), "pre":[(3,'s')], "next":[(5,'l')], "itsc": False}),
        (5,  {"coord":(56.5,10.3), "pre":[(4,'l')], "next":[(6,'s')], "itsc": False}),
        (6,  {"coord":(56.5, 39.6), "pre":[(5,'s')], "next":[(7,'l')], "itsc": False}),
        (7,  {"coord":(49.0,47.2), "pre":[(6,'l')], "next":[(8,'s')], "itsc": False}),
        (8,  {"coord":(11.5,47.2), "pre":[(7,'s')], "next":[(9,'l')], "itsc": False}),
        (9,  {"coord":(3.9,39.6), "pre":[(8,'l')], "next":[(10,'s')], "itsc": False}),
        (10,  {"coord":(3.9,10.3), "pre":[(9,'s')], "next":[(1,'l')], "itsc": False}),

        (11,  {"coord":(7.93,10.3), "pre":[(27,'r')], "next":[(12,'s')], "itsc": False}),
        (12,  {"coord":(7.93,16.8), "pre":[(11,'s')], "next":[(13,'s'),(31,'r')], "itsc": False}),
        (13,  {"coord":(7.93,33.6), "pre":[(12,'s'),(35,'r')], "next":[(14,'s')], "itsc": False}),
        (14,  {"coord":(7.93,39.6), "pre":[(13,'s')], "next":[(39,'r')], "itsc": False}),

        (15,  {"coord":(28,13), "pre":[(16,'s')], "next":[(28,'r'),(3,'l')], "itsc": False}),
        (16,  {"coord":(28,17), "pre":[(17,'s'),(32,'r'),(37,'l')], "next":[(15,'s')], "itsc": True}),
        (17,  {"coord":(28,33.6), "pre":[(18,'s')], "next":[(16,'s'),(36,'r'),(33,'l')], "itsc": True}),
        (18,  {"coord":(28,38.1), "pre":[(40,'r')], "next":[(17,'s')], "itsc": False}),

        (19,  {"coord":(32,13), "pre":[(29,'r'),(2,'l')], "next":[(20,'s')], "itsc": False}),
        (20,  {"coord":(32,17), "pre":[(19,'s')], "next":[(21,'s'),(33,'r'),(36,'l')], "itsc": True}),
        (21,  {"coord":(32,33.6), "pre":[(20,'s'),(37,'r'),(32,'l')], "next":[(22,'s')], "itsc": True}),
        (22,  {"coord":(32,38.1), "pre":[(21,'s')], "next":[(41,'r')], "itsc": False}),

        (23,  {"coord":(52.4,10.3), "pre":[(24,'s')], "next":[(30,'r')], "itsc": False}),
        (24,  {"coord":(52.4,16.8), "pre":[(25,'s'),(34,'r')], "next":[(23,'s')], "itsc": False}),
        (25,  {"coord":(52.4,33.6), "pre":[(26,'s')], "next":[(24,'s'),(38,'r')], "itsc": False}),
        (26,  {"coord":(52.4,39.6), "pre":[(42,'r')], "next":[(25,'s')], "itsc": False}),

        (27,  {"coord":(11.0,7.25), "pre":[(28,'s')], "next":[(11,'r')], "itsc": False}),
        (28,  {"coord":(21,7.25), "pre":[(29,'s'),(15,'r')], "next":[(27,'s')], "itsc": False}),
        (29,  {"coord":(38,7.25), "pre":[(30,'s')], "next":[(28,'s'),(19,'r')], "itsc": False}),
        (30,  {"coord":(49.0,7.25), "pre":[(23,'r')], "next":[(29,'s')], "itsc": False}),

        (31,  {"coord":(13.42,23.5), "pre":[(12,'r')], "next":[(32,'s')], "itsc": False}),
        (32,  {"coord":(21,23.5), "pre":[(31,'s')], "next":[(33,'s'),(16,'r'),(21,'l')], "itsc": True}),
        (33,  {"coord":(38.8,23.5), "pre":[(32,'s'),(20,'r'),(17,'l')], "next":[(34,'s')], "itsc": True}),
        (34,  {"coord":(46,23.5), "pre":[(33,'s')], "next":[(24,'r')], "itsc": False}),

        (35,  {"coord":(13.42,27.2), "pre":[(36,'s')], "next":[(13,'r')], "itsc": False}),
        (36,  {"coord":(21,27.2), "pre":[(37,'s'),(17,'r'),(20,'l')], "next":[(35,'s')], "itsc": True}),
        (37,  {"coord":(38.8,27.2), "pre":[(38,'s')], "next":[(36,'s'),(21,'r'),(16,'l')], "itsc": True}),
        (38,  {"coord":(46,27.2), "pre":[(25,'r')], "next":[(37,'s')], "itsc": False}),

        (39,  {"coord":(11.5,43.25), "pre":[(14,'r')], "next":[(40,'s')], "itsc": False}),
        (40,  {"coord":(21,43.25), "pre":[(39,'s')], "next":[(18,'r'),(41,'s')], "itsc": False}),
        (41,  {"coord":(37.6,43.25), "pre":[(22,'r'),(40,'s')], "next":[(42,'s')], "itsc": False}),
        (42,  {"coord":(49.0,43.25), "pre":[(41,'s')], "next":[(26,'r')], "itsc": False}),
    ]
    return node_list

def get_edge_list(node_list):
    edge_list = []
    for node_id, att in node_list:
        print(' ')
        print(" **** Iteration Id is: ",node_id, " attr: ",att)
        coord = att["coord"]
        for next_node, behavior in att['next']:
            next_coord = node_list[next_node-1][1]["coord"]
            print(" ------->"," next node: " ,next_node, " behavior: ", behavior, "next  coord: ",next_coord)
            edge = build_edge(node_id, next_node, coord, next_coord, behavior)
            edge_list.append(edge)
    return edge_list

def build_edge(u,v,u_coord,v_coord, behavior):
    ux, uy = u_coord
    vx, vy = v_coord
    if behavior == 's':
        # go straight
        waypoints, d = get_straight_waypoints(u_coord, v_coord, distance = WAYPOINT_DISTANCE)
    elif behavior == 'r':
        waypoints, d = get_curve_waypoints(u_coord, v_coord, 'r', distance= WAYPOINT_DISTANCE)
    elif behavior == 'l':
        waypoints, d = get_curve_waypoints(u_coord, v_coord, 'l', distance= WAYPOINT_DISTANCE)
    edge = (u,v,{"weight": d,"behavior":behavior, "waypoints":waypoints})
    return edge

def get_straight_waypoints(u_coord, v_coord, distance=WAYPOINT_DISTANCE):
    # Calculate the distance between u and v
    print("u: ",u_coord, "  v: ", v_coord)
    ux, uy = u_coord; vx,vy = v_coord
    dx = vx - ux
    dy = vy - uy
    print("dx: ", dx, "  dy: ", dy)
    d = math.sqrt(dx**2 + dy**2)

    # Calculate the normalized directional vectors
    dir_x = dx / d
    dir_y = dy / d
    # Calculate the yaw angle 
    yaw = math.atan2(dy, dx)
    # Calculate the number of waypoints
    num_waypoints = int(d / distance) + 1

    
    # Generate the waypoints
    waypoints = [(ux + i * distance * dir_x, uy + i * distance * dir_y) for i in range(num_waypoints)]
    
    # Ensure the last waypoint is v for precision issues
    # waypoints[-1] = v_coord
    waypoints = [(round(x,3),round(y,3), yaw) for x,y in waypoints]
    
    return waypoints, d

# def get_curve_waypoints(u,v,b,distance):
#     ux, uy = u; vx,vy = v
#     dx = vx-ux; dy = vy-uy
#     points = calculate_control_points(u,v,dx, dy, b) # two control points between u,v
#     # if dx > 0 && dy > 0:
#     t_values = np.linspace(0, 1, 1000)
#     curve_points = [bezier_curve(*points, t) for t in t_values]
#     d = compute_curve_length(curve_points)
#     waypoints = sample_waypoints(curve_points, distance)
#     waypoints = [(round(x,3),round(y,3)) for x,y in waypoints]
#     return waypoints, d

def calculate_yaw(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

def get_curve_waypoints(u, v, b, distance):
    ux, uy = u; vx, vy = v
    dx = vx-ux; dy = vy-uy
    points = calculate_control_points(u, v, dx, dy, b)  # two control points between u, v

    t_values = np.linspace(0, 1, 1000)
    curve_points = [bezier_curve(*points, t) for t in t_values]

    # Sample the curve to get waypoints
    waypoints = sample_waypoints(curve_points, distance)
    
    # Add yaw to the waypoints
    waypoints_with_yaw = []
    for i in range(len(waypoints) - 1):
        yaw = calculate_yaw(waypoints[i], waypoints[i+1])
        waypoints_with_yaw.append((round(waypoints[i][0], 3), round(waypoints[i][1], 3), yaw))
    
    # For the last waypoint, use the yaw of the previous point
    waypoints_with_yaw.append((round(waypoints[-1][0], 3), round(waypoints[-1][1], 3), yaw))

    d = compute_curve_length(curve_points)
    
    return waypoints_with_yaw, d






def calculate_control_points(u, v,dx,dy,b,k=BEZIER_CONTROL_PARAMETER):
    ux, uy = u; vx, vy = v
    if dx> 0 and dy > 0:
        if b == 'r':
            p2 = (ux, uy+k*dy)
            p3 = (vx-k*dx, vy)
        elif b == 'l':
            p2 = (ux+k*dx, uy)
            p3 = (vx, vy-k*dy)
    elif dx <0 and dy > 0:
        if b == 'r':
            p2 = (ux+k*dx, uy)
            p3 = (vx, vy-k*dy)
        elif b == 'l':
            p2 = (ux, uy+k*dy)
            p3 = (vx-k*dx, vy)
    elif dx < 0 and dy < 0:
        if b == 'r':
            p2 = (ux,uy+k*dy)
            p3 = (vx-k*dx, vy)
        elif b == 'l':
            p2 = (ux+k*dx, uy)
            p3 = (vx, vy-k*dy)
    elif dx > 0 and dy < 0:
        if b == 'r':
            p2 = (ux + k*dx, uy     )
            p3 = (vx       , vy-k*dy)
        elif b == 'l':
            p2 = (ux, uy+k*dy)
            p3 = (vx-k*dx, vy)
    else:
        raise ValueError (" Check dx or dy, should not be 0!")
    points = [u, p2, p3, v]
    return points

            

def save_edges(filename, edge_list):
    # edge_list = [(u, v, data) for u, v, data in edge_list if "waypoints" in data]
    # Convert the edge list to JSON
    json_data = json.dumps(edge_list, indent=4)

    # Write to file
    with open(filename, 'w') as file:
        file.write(json_data)

def load_edges(filename):
    with open(filename, 'r') as file:
        json_data = file.read()
        
    # Deserialize the JSON data
    edge_list = json.loads(json_data)
    
    return edge_list


# # node_list = get_node_list()
# # print(node_list)

# node_list = get_node_list()
# # get_edge_list(node_list)
# # print(node_list)

# wps, d = get_straight_waypoints((21,3.25),(38,3.25))
# print("way points: ", wps, d)