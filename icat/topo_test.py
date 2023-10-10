from topo import *
from nav_gym.map.util import load_img
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arrow


def view_topo(if_arrow = False):
    node_list = get_node_list()
    edge_list = get_edge_list(node_list=node_list)
    for edge in edge_list:
        print("----------------------")
        print("----> ", edge)

    fig,ax = plt.subplots()
    img = load_img('icat.png')

    for node_id, data in node_list:
        x, y = data["coord"]
        
        # Differentiate color based on "itsc" property
        color = 'green' if data["itsc"] else 'blue'
        
        # Draw the disk patch (circle)
        circle = Circle((x, y), radius=1, color=color, ec="black")  # ec stands for edgecolor
        ax.add_patch(circle)
        
        # Add node ID inside the circle
        ax.text(x, y, str(node_id), ha='center', va='center', color='white')

    for edge in edge_list:
        points = edge[2]["waypoints"]
        #points = [(10, 10), (30, 40), (55, 5), (5, 45)]

        # Unzip the points to separate x and y coordinates for easy plotting
        x_coords, y_coords, yaw = zip(*points)

        # Plotting the points
        plt.scatter(x_coords, y_coords, color='red')  # You can change the color and other properties as needed.
        # Drawing lines connecting the points
        plt.plot(x_coords, y_coords, color='red')  # You can change the color and other properties as needed.

    if if_arrow:
        for edge in edge_list:
            points = edge[2]["waypoints"]
            for x, y, yaw in points:
                arrow_length = 0.5  # Adjust as needed
                # plt.scatter(x, y, color='red')
                dx = arrow_length * np.cos(yaw)
                dy = arrow_length * np.sin(yaw)
                
                arrow = Arrow(x, y, dx, dy, width=0.2, color='red')  # Adjust width and color as needed
                ax.add_patch(arrow)

    # Plotting the image
    # plt.imshow(img, origin='lower', extent=[0, 60, 0, 50])  # The extent parameter sets the axis limits and 'origin' is now set to 'lower'.
    plt.imshow(img, extent=[0, 60, 0, 50]) 
    # Setting the x and y limits for the axes
    plt.xlim(0, 60)
    plt.ylim(0, 50)

    # Displaying the plot
    plt.show()

def test_save_edges():
    filename = "edges.json"
    node_list = get_node_list()
    edge_list = get_edge_list(node_list=node_list)
    save_edges(filename, edge_list)

def test_graph():
    G = build_graph()
    print(G)
    path, cost= A_star_path(G, 1, 2)
    print("cost: ", cost, path)

def test_load_edges():
    filename = "edges.json"
    edges = load_edges(filename)
    print("-----------------Testing Loading Edge:")
    print("len of edges: ", len(edges))
    print("edge[0]:  ", edges[0])
        
    
# test_graph()
view_topo()
# test_save_edges()
# test_load_edges()


