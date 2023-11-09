import numpy as np
from math import pi,cos,sin
from nav_gym.map.util import load_img

# ndt_pts =[
#                 (5.13, 2.480),
#                 (3.55,0.19),
#                 (2.40,-0.20),
#                 (0.0,-0.715),
#                 (0.52,2.42),
#                 (-0.09,4.33),
#                 (2.20,3.82),
#                 (5.61,4.42),
#                 (2.82,1.77)
#             ]

# icat_pts = [

#     (5.87,19.2),
#     (23.36,45.16),
#     (35.56,45.23),
#     (60,50),
#     (54.33,19.55),
#     (59.56,0.42),
#     (36.3, 54.1),
#     (0.71,0.49)
#     (30, 25.4)
# ]



import numpy as np
import matplotlib.pyplot as plt

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

# best = np.inf
# best_param = None
# for scaling_factor in np.arange(9.8, 10.2+0.1, 0.1):
#     for theta in np.arange(pi-0.3, pi+0.3, 0.1):
#         for dx in np.arange(5.0, 10.0, 0.1):
#             for dy in np.arange(15.0, 20.0, 0.1):
#                 final_pts = calc_tranformed_points(ndt_pts, scaling_factor, theta, dx,dy)
#                 disp = icat_pts-final_pts
#                 cost = np.sum(np.hypot(disp[:,0], disp[:,1]))
#                 print("cost: ", cost)
#                 if cost <= best:
#                     best = cost
#                     best_param = (scaling_factor,theta, dx, dy)
                    
# print("best_param" ,best_param)

# scaling_factor, theta, dx, dy = best_param
scaling_factor, theta, dx, dy = (10.193, 3.18, 2.0, 8.0)
# scaling_factor, theta, dx, dy = (10.0, 3.20, 0.0, 10.0)
final_pts = calc_tranformed_points(ndt_pts, scaling_factor, theta, dx,dy)


# print(rotated_points)
# Plotting the points
plt.figure(figsize=(10, 8))
plt.scatter(icat_pts[:, 0], icat_pts[:, 1], color='red', label='Target ICAT Points')

# plt.scatter(rotated_points[:, 0], rotated_points[:, 1], color='blue', label='Rotated NDT Points')
plt.scatter(final_pts[:, 0], final_pts[:, 1], color='green', label='Rotated NDT Points')
img = load_img('icat.png')
plt.imshow(img, origin='lower', extent=[0, 60, 0, 50])  # The extent parameter sets the axis limits and 'origin' is now set to 'lower'.
plt.imshow(img, extent=[0, 60, 0, 50])
# Labels and legend
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Point Transformation Validation')
plt.legend()
plt.axis('equal')  # Equal scaling on both axes for correct aspect ratio

# Show the plot
plt.show()





