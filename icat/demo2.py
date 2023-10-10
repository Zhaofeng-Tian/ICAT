import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import make_interp_spline

# Define the points
points = np.array([
    [56.53, 11],
    [56.53, 26],
    [56.53, 41],
    [50, 47],
    [30, 47],
    [10, 47],
    [3.78, 41],
    [3.78, 26],
    [3.78, 11],
    [10, 3.23],
    [30, 3.23],
    [48, 3.23],
    [56.53, 11],  # Close the loop
])

# Separate the points into x and y coordinates
x, y = points[:, 0], points[:, 1]

# Define the degree of the spline
k = 3

# Create the t values
t = np.linspace(0, 1, len(x))

# Create the B-spline
tck = make_interp_spline(t, np.c_[x, y], k=k)
tnew = np.linspace(0, 1, 100)
xnew, ynew = tck(tnew).T

# Compute the yaw angles
tck_derivative = tck.derivative()
dxnew, dynew = tck_derivative(tnew).T
yaw_angles = np.arctan2(dynew, dxnew)

# Define the car parameters
car_length = 3
car_width = 2
num_cars = 3
car_spacing = 3  # Spacing between cars in meters

# Initialize the plot
fig, ax = plt.subplots()

# Plot the interpolated points and yaw angles
plt.plot(xnew, ynew, '-', label='Path')

# Create the car patches
cars = [Rectangle((0, 0), car_length, car_width, fc='b') for _ in range(num_cars)]
for car in cars:
    ax.add_patch(car)

plt.xlim(min(xnew) - 10, max(xnew) + 10)
plt.ylim(min(ynew) - 10, max(ynew) + 10)
plt.axis('equal')
plt.legend()

# Function to compute the distance between two points
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Initialize the car positions
car_positions = [0] * num_cars

# Update the plot in each animation step
for i in range(len(xnew)):
    for j, car in enumerate(cars):
        # Update the car position
        car_positions[j] += car_spacing / len(xnew)
        if car_positions[j] >= len(xnew):
            car_positions[j] -= len(xnew)
        index = int(car_positions[j])
        x, y, theta = xnew[index], ynew[index], yaw_angles[index]
        car.set_xy((x - car_length / 2 * np.cos(theta), y - car_length / 2 * np.sin(theta)))
        car.angle = np.degrees(theta)
        plt.draw()
        plt.pause(1)
    

plt.show()