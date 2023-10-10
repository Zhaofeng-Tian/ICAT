from nav_gym.obj.robot.robot import CarRobot, DiffRobot
from nav_gym.obj.robot.robot_param import CarParam
from nav_gym.sim.config import Config
from nav_gym.map.util import load_map
from nav_gym.obj.geometry.util import rot,line_line, line_polygon
from nav_gym.obj.geometry.objects import Polygon, build_wall
from nav_gym.sim.plot import plot_cars, plot_start_goal, plot_geo_disks, plot_VOs, plot_ORCA
from nav_gym.alg.ddpg.ddpg_torch import Agent2, Agent_VDN,Agent_VDN2,Agent_QMix, Agent_LagMix
from nav_gym.alg.rvo.rvo import RVO
from nav_gym.alg.rvo.orca import ORCA
import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import random
import time
from sklearn.neighbors import KDTree
import gym
from gym import spaces
import os
import time
from nav_gym.obj.geometry.util import rot, topi
from nav_gym.sim.mix.buffer import Buffer

class Env1(gym.Env):

    def __init__(self,if_render = True,render_mode = "human", sensor_update_type = "obj"):
        # self.world_x = 10
        # self.world_y = 10
        self.sensor_update_type = sensor_update_type
        self.world_x =10
        self.world_y = 10
        self.world_reso = 0.01
        self.num_r = round(self.world_y/self.world_reso)
        self.num_c = round(self.world_x/self.world_reso)
        self.carparam = CarParam(shape = np.array([0.53,0.25,0.25,0.35,0.65]),type="car")
        # self.carparam = CarParam()
        self.n_cars =10
        self.cars = []
        self.walls = []
        self.wall_polygons = []
        self.polygons = []
        self.circles = []

        self.obj_centers = []  # geometry centers
        self.obj_radius = []   # geometry disk radius (big disk)
        self.obj_velocity = [] # linear speed, heading angle

        self.policy ="ORCA"  #"ORCA" or "VO"
        self.KD_tree = None
        self.plot_pause_time =0.1
        # if render_mode == "human":
        if if_render:
            self.plot = True
        else:
            self.plot = False
        self.plot_VO = False
        self.plot_lidar = False

        # Gym training param
        # self.observation_space = spaces.Box(low = 0.1, high = 6.0,shape = (34,) , dtype=np.float32)
        self.action_space = spaces.Box(self.carparam.v_limits[1],self.carparam.v_limits[0],shape=(2,),dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
            "lidar":spaces.Box(0.1, 6.0,shape=(self.n_cars, self.carparam.ray_num,),dtype=np.float32),
            "goal":spaces.Box(0.0, 2.0, shape=(self.n_cars,2),dtype=np.float32),
            # "v": spaces.Box(self.carparam.v_limits[0],self.carparam.v_limits[1],shape=(2,),dtype=np.float32)
            }
        )
        self.max_episode_steps = 1000
        self.num_episodes = 1000 

        # ************** 1. Set up starts and goals *********************************
        # self.initial_states = [np.array([1.5,2.5,0.,0.,0.]),np.array([8.5,2.5, pi, 0.,0.])]
        n_cars = self.n_cars; world_x = self.world_x; world_y = self.world_y
        center_x = world_x/2; center_y = world_y/2; r= 4. ; dangle = pi ; n_interval = 1
        self.initial_states = []
        self.global_goals = []
        for i in range(self.n_cars):
            state = np.array([center_x +r*cos(i*2*pi/(n_interval* n_cars)), center_y+r*sin(i*2*pi/(n_interval*n_cars)), topi(i*2*pi/n_cars+pi), 0., 0. ])
            goal = np.array([center_x +r*cos(i*2*pi/n_cars+dangle), center_y+r*sin(i*2*pi/n_cars+dangle), topi(i*2*pi/n_cars+pi), 0., 0. ])
            self.initial_states.append(state)
            self.global_goals.append(goal)
        print( "initial position: ", self.initial_states)
        print(" global goals: ", self.global_goals)
        # assert 1==2 , " 222"
        assert len(self.initial_states)==self.n_cars, "The initial states number does not match robot number"
        # self.observation_space = spaces.Dict(
        #     {
        #     "lidar":spaces.Box(0.1, 6.0,shape=(self.n_cars, self.carparam.ray_num,),dtype=np.float32),
        #     "goal":spaces.Box(0.0, 2.0, shape=(self.n_cars,2),dtype=np.float32),
        #     "v": spaces.Box(self.carparam.v_limits[0],self.carparam.v_limits[1],shape=(2,),dtype=np.float32)
        #     }
        # )

        # *******************  2. Image Loading ***********************
        current_file_path = os.path.abspath(__file__)
        print("Current file path:", current_file_path)
        current_folder_path = os.path.dirname(current_file_path)
        print("Current folder path:", current_folder_path)
        root_folder_path = os.path.dirname(current_folder_path)
        print("Root folder path:", root_folder_path)

        map_path = root_folder_path+'/map/'
        # self.img = load_map(map_path+"corridor.png")
        self.img = load_map("icat.png")
        # map = self.img.copy()
        self.img[np.where(self.img>0.)] = 1.0
        # self.map = 1 - self.img
        self.ocp_map = None
        # ****************** 3. Array Map template  *********************
        if sensor_update_type != "obj":
            self.map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
            self.map[0:20,:]=1.0; self.map[self.num_r-21:self.num_r-1,:]=1.0; self.map[:,0:20]=1.0; self.map[:,self.num_c-21:self.num_c-1]=1.0
            self.img = 1.0 - self.map

            self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
            self.ocp_map = self.map + self.agt_map

        # *********************** 4. Agents Initialization ************
        # a. creating instances of robots
        for i in range(self.n_cars):
            self.cars.append(CarRobot(id = i,
                        param= self.carparam, 
                        initial_state = self.initial_states[i],
                        global_goal= self.global_goals[i]))
            if sensor_update_type != "obj":
                self.cars[i].id_fill_body(self.agt_map)
            self.polygons.append(Polygon(self.cars[i].vertices[:4]))
        self.walls.append(build_wall([0.,0.,], [0.,self.world_y], 0.15))
        self.walls.append(build_wall([0.,0.,], [self.world_x,0.], 0.15))
        self.walls.append(build_wall([self.world_x,0.,], [self.world_x,self.world_y], 0.15))
        self.walls.append(build_wall([0.,self.world_y,], [self.world_x,self.world_y], 0.15))
        for wall in self.walls:
            self.wall_polygons.append(Polygon(wall))        
        # b. fill the occupancy map for sensor update    
        if sensor_update_type != "obj":
            self.ocp_map = self.map + self.agt_map
        
        # c. state and obs update
        for i in range(self.n_cars):
            self.cars[i].state_init(self.initial_states[i])
            self.cars[i].obs_init(self.ocp_map,self.polygons,self.circles,type = self.sensor_update_type)

            self.obj_centers.append(self.cars[i].vertices[5])
            self.obj_velocity.append(self.cars[i].state[2:4])
            self.obj_radius.append(self.cars[i].geo_r)
        ts = time.time()
        self.KD_tree = KDTree(self.obj_centers)
        te = time.time()
        # print("KD tree built time : ", te-ts)
        # assert 1==2, "stop to review KDTree"
        # print("obj_velocity: ", self.obj_velocity)
        # assert 1==2, "stop to review velocity"
        # d. set up the canvas
        if self.plot == True:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            self.ax.set_xlim([0,self.world_x])
            self.ax.set_ylim([0,self.world_y])


    def reset(self):
        # clean checklist 1. agent map 2. obstacle list
        if self.sensor_update_type != "obj":
            self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        self.polygons = []
        self.obj_centers = []
        self.obj_velocity = []
        for car in self.cars:
            car.state_init(self.initial_states[car.id])
            if self.sensor_update_type != "obj":
                car.id_fill_body(self.agt_map)
            self.polygons.append(Polygon(car.vertices[:4]))
            self.obj_centers.append(car.vertices[5])
            self.obj_velocity.append(car.state[2:4])
        self.polygons = self.polygons + self.wall_polygons
        
        self.KD_tree = KDTree(self.obj_centers)
        if self.sensor_update_type != "obj":
            self.ocp_map = self.map + self.agt_map
        
        for car in self.cars:
            car.obs_init(self.ocp_map,self.polygons,self.circles, type = self.sensor_update_type)
        return self._get_obs()

    def step(self,actions,infos): # action shape [n_agents, action(v, phi)] (n,2) array
        # clean checklist 1. agent map 2. obstacle list
        for car in self.cars:
            print("car state: ************************************", car.state)
        if self.sensor_update_type != "obj":
            self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        self.polygons = []
        self.obj_centers = []
        self.obj_velocity = []
        # states update loop
        for car in self.cars:
            if car.done:
                print("Use reset function~~~~~~~~~~~~~~~~")
                car.state_init(self.initial_states[car.id])
            else:
                car.state_update(actions[car.id])
            if self.sensor_update_type != "obj":   
                car.id_fill_body(self.agt_map)
            # --- appending starts ----
            self.polygons.append(Polygon(car.vertices[:4]))
            self.obj_centers.append(car.vertices[5])
            self.obj_velocity.append(car.state[2:4])
        self.polygons = self.polygons + self.wall_polygons

            # --- appending ends ------
        self.KD_tree = KDTree(self.obj_centers)
        if self.sensor_update_type != "obj":
            self.ocp_map = self.map + self.agt_map
        # observation, done, and reward update loop
        for car in self.cars:
            car.obs_update(self.ocp_map,self.polygons,self.circles, type = self.sensor_update_type)
        
        # print("Obj velocity: ", self.obj_velocity)
        # for car in self.cars:
        #     print("*******************step: ", " *********************")
        #     print("Car ID: ", car.id)
        #     print("lidar obs: ",car.lidar_obs)
        #     print("car state: ",car.state)
        #     print("car local goal: ", car.local_goal)
        #     print("collision: ",car.collision)
        #     print("done: ", car.done )
        #     print("reward: ", car.reward)
        #     print("reward info: ", car.reward_info)
        
        # render

        if self.plot:
            self.render_frame(infos)

    
        for car in self.cars:
            print("car state_: ************************************", car.state)
        # get data
        obs_ = self._get_obs()
        done = self._get_done()
        achieve = self._get_achieve()
        collision = self._get_collision()
        reward = self._get_reward()
        truncated = None
        info = {}
        return obs_,  reward, achieve, collision, truncated, info
        
        

    def _get_reward(self):
        reward_list=[]
        for car in self.cars:
            reward_list.append(car.reward)
        return np.array(reward_list, dtype=np.float32)

    def _get_done(self):
        done_list = []
        for car in self.cars:
            done_list.append(car.done)
        return np.array(done_list,dtype=np.int32)

    def _get_achieve(self):
        achieve_list = []
        for car in self.cars:
            achieve_list.append(car.achieve)
        return np.array(achieve_list,dtype=np.int32)

    def _get_collision(self):
        collision_list = []
        for car in self.cars:
            collision_list.append(car.collision)
        
        return np.array(collision_list,dtype=np.int32)

    def _get_safe_ranges(self):
        safe_range_list = []
        for car in self.cars:
            safe_range_list.append(car.safe_ranges)
        
        return np.array(safe_range_list)