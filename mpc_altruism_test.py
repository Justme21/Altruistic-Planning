from casadi import *
import numpy as np
import sys

sys.path.insert(0,'../Driving-Simulator')
import simulator
from vehicle_classes import Car

#####################################################################################################
#Simulator Definition
def initialiseSimulator(cars,speed_limit,graphics,init_speeds,lane_width=None):
    """Takes in a list of cars and a boolean indicating whether to produce graphics.
       Outputs the standard straight road simulator environment with the input cars initialised
       on the map with the first car (presumed ego) ahead of the second"""
    num_junctions = 5
    num_roads = 4
    road_angles = [90 for _ in range(num_junctions)]
    road_lengths = [5,20,5,270]
    junc_pairs = [(i,i+1) for i in range(num_roads)]

    starts = [[(0,1),1],[(0,1),0]] #Follower car is initialised on the first road, leading car on the 3rd (assuring space between them)
    dests = [[(1,2),0],[(1,2),0]] #Simulation ends when either car passes the end of the 

    run_graphics = graphics
    draw_traj = False #trajectories are uninteresting by deafault
    debug = False #don't want debug mode

    runtime = 120.0 #max runtime; simulation will terminate if run exceeds this length of time

    #Initialise the simulator object, load vehicles into the simulation, then initialise the action simulation
    sim = simulator.Simulator(run_graphics,draw_traj,runtime,debug,dt=cars[0].timestep)
    sim.loadCars(cars)

    sim.initialiseSimulator(num_junctions,num_roads,road_angles,road_lengths,junc_pairs,\
                                                    init_speeds,starts,dests,lane_width=lane_width)

    return sim

#####################################################################################################

if __name__ == "__main__":
    ###########################################
    #PREAMBLE
    ###################################
    #Define Vehicles
    debug = False
    T = 10
    dt = .2

    ego = Car(controller=None,is_ego=True,debug=debug,label="Ego",timestep=dt)
    other = Car(controller=None,is_ego=False,debug=debug,label="Other",timestep=dt)

    ###################################
    ###################################
    #Initialise Simulator
    lane_width = 4
    graphics = False
    speed_limit = 29 #m/s
    accel_range=[-9,3]
    init_speeds = [20,20] #m/s
    sim = initialiseSimulator([ego,other],speed_limit,graphics,init_speeds,lane_width=lane_width)

    #Reorient Cars (cars initially face same direction as lane they are on)
    other.heading = (other.heading+180)%360
    other.initialisation_params["heading"] = other.heading
    other.sense()

    #Cars next to each other for joint planning experiments
    other.y_com = ego.y_com
    other.initialisation_params["prev_disp_y"] = ego.initialisation_params["prev_disp_y"]

    #Ensure state of both agents is accurate
    ego.sense()
    other.sense()

    ###################################
    ###################################
    ##Define Trajectory Options
    vehicle_traj_specs = {}
    vehicle_traj_rewards = {}
    vehicle_traj_specs[ego] = [(lane_width,0),(0,-10)]
    vehicle_traj_rewards[ego] = [1,-1]
    vehicle_traj_specs[other] = [(0,5),(0,-10)]
    vehicle_traj_rewards[other] = [1,-1]

    #Define initial trajectories
    orig_states = {}
    vehicle_traj_dict = {}
    for veh in [ego,other]:
        vehicle_traj_list = []
        veh_state = filterState(veh.state,veh.Lr+veh.Lf)
        spec = vehicle_traj_specs[veh]
        vehicle_traj_dict[veh] = makeTrajectories(veh_state,spec,T)
        orig_states[veh] = dict(veh_state)

    ###################################
    ###########################################

    ###########################################
    #RUNTIME
    ###################################
    #Initialise Altruism Ground Truth Values and Distribution
    alt_ego = 0
    alt_other = 0

    num_altruism_estimates = 4
    alt_stepsize = 1/(num_altruism_estimates-1)
    #alt_other_prior = [   alt_stepsize for _ in range(num_altruism_estimates)]
    ################################### 
    ###################################
    #For checking for collision
    safety_distance_s = 5
    safety_distance_d = 2
    time_radius = 1
    safety_params = (time_radius,safety_distance_d,safety_distance_s)
    ###################################
    ###################################
    #Initialise Trajectories and Priors    
    init_trajectory_costs = {}
    init_trajectory_costs[other] = [trajCost(x) for x in vehicle_traj_dict[other]]
    other_prior = [1/len(vehicle_traj_dict[other]) for _ in vehicle_traj_dict[other]] 

    ###################################
    ###################################
    #Initialise cur_trajectories so Bayesian Estimation of Trajectory can be performed
    cur_trajectories = {}

    ###################################
    ###################################
    #Construct Reward Grid so Other can choose trajectory
    true_reward_grid = makeRewardGrid(vehicle_traj_dict[other],vehicle_traj_rewards[other],alt_other,vehicle_traj_dict[ego],vehicle_traj_rewards[ego],alt_ego,dt,safety_params)

    max_row_r = None
    max_row_index = None
    for j in range(true_reward_grid.shape[0]):
       if max_row_r is None or max(true_reward_grid[j,:,0])>max_row_r:
           max_row_r = max(true_reward_grid[j,:,0])
           max_row_index = j

    other_trajectory = vehicle_traj_dict[other][max_row_index]

    ###################################
    ###################################
    #Run Simulation to estimate Trajectory and Altruism simultaneously
    for t in np.arange(0,T,dt):
        #Update distribution over what trajectory is being followed
        other_state = filterState(other.state,other.Lf+other.Lr)
        cur_trajectories[other] = makeTrajectories(other_state,vehicle_traj_specs[other],T-t,orig_states[other])
        other_prior = updatePreference(cur_trajectories[other],init_trajectory_costs[other],other_prior)

        #Update Prior over Altruism
        #Needs to be done every timestep only if we are doing decision-making over altruism
        # Basically a HMM with the estimated trajectory type as hidden state
        alt_other_prior = updateAltruism(vehicle_traj_dict[other],vehicle_traj_rewards[other],vehicle_traj_dict[ego],vehicle_traj_rewards[ego],alt_ego,dt,safety_params,other_prior,alt_stepsize)

        if t%.5 == 0:
            print("T: {}\tTraj {}".format(t,[round(x,2) for x in other_prior]))
            print("T: {}\tAlt: {}\n".format(t,[round(x,2) for x in alt_other_prior]))

        action = (idm(other.state,ego.state),0) # Controlled by reactive IDM
        #action = (idm(other.state,None),0) #IDM going in straight line
        #action = other_trajectory.action(t,other.Lf+other.Lr) # Following preset trajectory

        other.setAction(*action)
        other.move()
        other.sense()

        #If ego does not move then IDM immediately slows down, but also moves past the ego car
        # immediately. Every timestep thereafter the longitudinal distance between other and ego
        # is increasing and IDM becomes more focused on reaching target velocity
        ego.move()
        ego.sense()
        #Other_state is defined at the start of the loop

    ###################################
    import pdb
    pdb.set_trace()
    ###########################################

