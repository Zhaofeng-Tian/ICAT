import numpy as np
from math import pi,cos,sin, atan2
from nav_gym.map.util import load_img
import matplotlib.pyplot as plt
from topo import *
from nav_gym.obj.geometry.util import topi
# Define the points
ndt_pts = np.array([
    (5.13, 2.480),
    (3.55, 0.19),
    (2.40, -0.20),
    (0.0, -0.715),
    (0.52, 2.42),
    (-0.09, 4.33),
    (2.20, 3.82),
    (5.61, 4.42),
    (2.82,1.77)
])

icat_pts = np.array([
    (5.87, 19.2),
    (23.36, 45.16),
    (35.56, 45.23),
    (60, 50),
    (54.33, 19.55),
    (59.56, 0.42),
    (36.3, 5.1),
    (0.71, 0.49),
    (30, 25.4)
])
icat_centered = icat_pts - icat_pts[-1]

import numpy as np

def calc_inverse_transformed_points(icat_pts, scaling_factor, theta, dx, dy):
    # Inverse Translation
    translated_points = icat_pts - np.array([dx, dy])

    # Creating Inverse Rotation Matrix
    inv_rotation_matrix = np.array([
                                    [np.cos(-theta), -np.sin(-theta)],
                                    [np.sin(-theta), np.cos(-theta)]
                                  ])

    # Adjusting for the original last point translation
    rotated_points = translated_points - translated_points[-1]
    
    # Inverse Rotation
    rotated_points = np.dot(rotated_points, inv_rotation_matrix)
    
    # Revert the adjustment for the original last point translation
    rotated_points = rotated_points + translated_points[-1]

    # Inverse Scaling
    final_pts = rotated_points / scaling_factor
    final_pts += np.array([0.08, 0.07])

    return final_pts
    # return translated_points
    # return rotated_points


def calc_tranformed_points(ndt_pts, scaling_factor, theta, dx,dy):
    scaled_ndt_pts =  ndt_pts *scaling_factor
    rotation_matrix = np.array([
                                    [cos(theta), -sin(theta)],
                                    [sin(theta), cos(theta)]
                                ])
    translated_points = scaled_ndt_pts - scaled_ndt_pts[-1]
    rotated_points = np.dot(translated_points, rotation_matrix)
    rotated_points = rotated_points + scaled_ndt_pts[-1]
    final_pts = rotated_points+ np.array([dx,dy])
    return final_pts





def get_transformed_nodes():
    # scaling_factor, theta, dx, dy = best_param
    scaling_factor, theta, dx, dy = (10.193, 3.18, 2.0, 8.0)
    # scaling_factor, theta, dx, dy = (10.1, 3.17, 2.0, 8.0)
    # scaling_factor, theta, dx, dy = (10.0, 3.20, 0.0, 10.0)
    final_pts = calc_tranformed_points(ndt_pts, scaling_factor, theta, dx,dy)
    inverse_pts = calc_inverse_transformed_points(icat_pts, scaling_factor, theta, dx,dy)

    node_list = get_node_list()
    edge_list = get_edge_list(node_list=node_list)

    node_pts = []
    for node in node_list:
        node_pts.append(np.array(node[1]["coord"][:2]))
    node_pts.append(np.array([30, 25.4]))
    node_pts = np.array(node_pts)
    print("node_pts: ", node_pts)
    transformed_pts = calc_inverse_transformed_points(node_pts,scaling_factor, theta, dx,dy)
    dx, dy = transformed_pts[1] - transformed_pts[0]
    theta = atan2(dy,dx)
    print(" node 1 and 2 yaw angle: ", theta)
    yaw25_24 = -pi/2 + theta
    print(" yaw25_24 {}, topi {} ".format(yaw25_24, yaw25_24+2*pi))


    for i in range(len(node_list)):
        _, _, yaw = node_list[i][1]["coord"]
        x,y = transformed_pts[i]
        node_list[i][1]["coord"] = (x,y, topi(yaw+ theta))

    print("node list: ", node_list)
    return node_list



transformed_nodes = get_transformed_nodes()
transformed_pts = get_points_from_nodes(transformed_nodes)
fig,ax = plt.subplots()
# print(rotated_points)
# Plotting the points
# plt.figure(figsize=(10, 8))
# plt.scatter(icat_pts[:, 0], icat_pts[:, 1], color='red', label='Target ICAT Points')

# plt.scatter(ndt_pts[:, 0], ndt_pts[:, 1], color='blue', label='Original NDT Points')
# plt.scatter(inverse_pts[:, 0], inverse_pts[:, 1], color='red', label='inverse ICAT Points')
# plt.scatter(final_pts[:, 0], final_pts[:, 1], color='green', label='Rotated NDT Points')
plt.scatter(transformed_pts[:, 0], transformed_pts[:, 1], color='pink', label='Rotated NDT Points')
# img = load_img('icat.png')
# plt.imshow(img, origin='lower', extent=[0, 60, 0, 50])  # The extent parameter sets the axis limits and 'origin' is now set to 'lower'.
# plt.imshow(img, extent=[0, 60, 0, 50])
# Labels and legend
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Point Transformation Validation')
plt.legend()
plt.axis('equal')  # Equal scaling on both axes for correct aspect ratio


for node_id, data in enumerate(transformed_pts[:-1]):
    x,y = data
    ax.text(x, y, str(node_id), ha='center', va='center', color='white')

# Show the plot
plt.show()





