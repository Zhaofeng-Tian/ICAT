import numpy as np
import matplotlib.pyplot as plt
import math

def draw_curve(p1, p2, turn_direction, radius):
    # Based on the turn direction, determine the circle's center
    if turn_direction == "left":
        cx, cy = p1[0], p1[1] + radius
    elif turn_direction == "right":
        cx, cy = p1[0], p1[1] - radius
    else:
        raise ValueError("Turn direction must be 'left' or 'right'.")

    # Calculate angles for our arc
    start_angle = math.atan2(p1[1] - cy, p1[0] - cx)
    end_angle = math.atan2(p2[1] - cy, p2[0] - cx)

    # Create the arc
    theta = np.linspace(start_angle, end_angle, 100)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-b', label='curve')
    plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', label='endpoints')
    plt.scatter(cx, cy, color='green', label='center')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

    # def plan_car_following(self, id):
    #     leading_id = self.DistBuffer[id][0]
    #     ego_v = self.StateBuffer[id]["linear_speed"]
    #     leading_v = self.StateBuffer[leading_id]["linear_speed"]
    #     desired_v = ego_v
    #     """
    #     This is car following logic.
    #     """
    #     if leading_id == -1:
    #         desired_v = self.constant_speed
    #     else:
    #         d = self.DistBuffer[id][1]
    #         if d <= self.safe_clearance + self.car_info["hl"]*2:
    #             desired_v = 0.0
    #         else:                               
    #             total_dist = d - self.safe_clearance + self.car_info["hl"]*2
    #             acc_speed = total_dist + self.head_time * (leading_v - ego_v) /self.head_time
    #             desired_v = max(acc_speed, self.constant_speed)
    #     current_point = self.StateBuffer[id]["current_point"]
    #     looking_point = self.StateBuffer[id]["ahead_point"]
    #     s,d = self.StateBuffer[id]["sd"]
    #     return current_point, looking_point, s,d, ego_v , desired_v 


    # def excute_plan(self, plan):
    #     """ Pure pursuit Controller """
    #     Kp = 0.45
    #     current_point, looking_point,s,d, ego_v, disired_v = plan
    #     a = disired_v - ego_v
    #     if a >= 0:
    #         a = min(a, self.car_info["amax"])
    #     if a < 0:
    #         a = max(a, self.car_info["amin"])
    #     planned_v = ego_v+a
    #     looking_yaw = looking_point[2]
    #     ego_yaw = current_point[2]
    #     dangle1 = 0 - d
    #     dangle2 = self.to_pi(looking_yaw - ego_yaw)
    #     # print("following yaw: ", round(looking_yaw,3), "  ego yaw: ", round(ego_yaw,3))
    #     # print(" ******* in Control", round(dangle1,3), round(dangle2,3))
    #     planned_w = Kp*(dangle1+dangle2)
    #     return planned_v, planned_w
# Test
p1 = (0, 0)
p2 = (0, 4)
draw_curve(p1, p2, "left", 2)