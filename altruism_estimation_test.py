import math
import numpy as np
import random
import sys

sys.path.insert(0,'../Driving-Simulator')
import simulator
from vehicle_classes import Car

from trajectory_type_definitions import Trajectory

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
#####################################################################################################
#Trajectory Stuff

def makeTrajectories(cur_state,spec,T,init_state=None):
    """Returns a list of trajectories starting from cur_state, of length T.
       Spec is a list of (dx,dv) pairs, where each pair corresponds to a distinct trajectory
       specification.
       If init_state is specified then the destination states for the trajectories will be set
       from init_state (as opposed to cur state)"""
    traj_list = []

    # Init_state is the state the trajectories are supposed to have originated at. If init_state is None then
    # assume the current state is the initial state of the trajectory
    if init_state is None:
        init_state = cur_state

    for dx,dv in zip([x[0] for x in spec], [x[1] for x in spec]):
        label = "dx-{},dv-{}".format(dx,dv)
        dest_state = dict(init_state)
        dest_state["position"] = tuple([dest_state["position"][0]+dx,dest_state["position"][1]])
        dest_state["velocity"] += dv
        dest_state["parametrised_acceleration"] = (0,0) #parametrised acceleration is introduced to handle acceleration constraints
        traj = Trajectory(cur_state,dest_state,T,label)
        traj_list.append(traj)

    return traj_list


def checkForCrash(traj1,traj2,dt,safety_params):
    (safe_time_dist,safe_d_dist,safe_s_dist) = safety_params
    for t in np.arange(0,traj1.traj_len_t,dt):
        (d_veh1,s_veh1) = traj1.position(t)
        (d_veh2,s_veh2) = traj2.position(t)

        (_,ds_dt_veh1) = traj1.velocity(t)
        (_,ds_dt_veh2) = traj2.velocity(t)
        ds_dt = max(ds_dt_veh1,ds_dt_veh2)

        s_dist = (s_veh1-s_veh2)**2
        d_dist = (d_veh1-d_veh2)**2

        safety_region_invasion_check = s_dist/(safe_s_dist+safe_time_dist*ds_dt)**2 + d_dist/(safe_d_dist**2)

        if safety_region_invasion_check<1: return True #cars are too close, crash has occurred

    return False


def makeRewardGrid(row_traj_list,row_reward_list,row_alt,column_traj_list,column_reward_list,column_alt,dt,safety_params):
    reward_grid = np.empty((len(row_traj_list),len(column_traj_list),2))
    for i,row_traj in enumerate(row_traj_list):
        for j,column_traj in enumerate(column_traj_list):
            if checkForCrash(column_traj,row_traj,dt,safety_params): reward_grid[i,j] = ((-np.inf,-np.inf))
            else:
                column_r = column_reward_list[j]
                row_r = row_reward_list[i]
                reward_grid[i,j] = (((1-row_alt)*row_r+row_alt*column_r,(1-column_alt)*column_r+column_alt*row_r))

    return reward_grid


def updateAltruism(row_traj_list,row_reward_list,column_traj_list,column_reward_list,column_alt,dt,safety_params,row_traj_prior,alt_stepsize):
    #NOTE: For the time being we presume to only be estimating the altruism of the row player
    row_alt_prior = []
    normalising_total = 0
    #arange(a,b,c) gives a+i*c<b. So to include 1 into distribution an extra step must be added
    for row_alt in np.arange(0,1+alt_stepsize,alt_stepsize):

        reward_grid = makeRewardGrid(row_traj_list,row_reward_list,row_alt,column_traj_list,column_reward_list,column_alt,dt,safety_params)

        #Stackelberg Game, assumes Other/Non-Ego/In-Lane/Row agent is initial decider
        #max_row_r,max_column_r = None,None
        max_row_r = None
        for j in range(reward_grid.shape[0]):
            if max_row_r is None or max(reward_grid[j,:,0])>max_row_r:
                #NOTE: I don't fully trust this.
                # Currently assume that row chooses maximum payoff, and column chooses
                # maximum payoff given row is choosing maximum payoff
                # but if the row the row player chooses does not match the row expected by
                # column player then there is a problem
                max_row_r = max(reward_grid[j,:,0])
                #max_column_r = max(reward_grid[j,:,1])

        #NOTE: Arbitrary Constant being used here
        likelihood_tot = sum([math.exp(-4*abs(max(reward_grid[j,:,0])-max_row_r))*row_traj_prior[j] for j in range(reward_grid.shape[0])])
        normalising_total += likelihood_tot
        row_alt_prior.append(likelihood_tot)

    row_alt_prior = [x/normalising_total for x in row_alt_prior]

    return row_alt_prior

#####################################################################################################
#####################################################################################################
#For computing the Bayes Filter
def trajCost(traj):
    #Adapted from Cooperative Planning implementation for w=1
    T = traj.traj_len_t

    delta = [x[0] for x in traj.line_x.coefs]
    sigma = [y[0] for y in traj.line_y.coefs]

    #Derived by hand from trajectory definitions
    i_sigma = (144/5)*(sigma[4]**2)*(T**5) + (144/4)*sigma[3]*sigma[4]*(T**4) + (48/3)*sigma[2]*sigma[4]*(T**3) + (36/3)*(sigma[3]**2)*(T**3) +\
                  (24/2)*sigma[2]*sigma[3]*(T**2) + 4*(sigma[2]**2)*T
    i_delta = (400/7)*(delta[5]**2)*(T**7) + (480/6)*delta[4]*delta[5]*(T**6) + (240/5)*delta[3]*delta[5]*(T**5) + (80/4)*delta[2]*delta[5]*(T**4) + (144/5)*(delta[4]**2)*(T**5) +\
                     (144/4)*delta[3]*delta[4]*(T**4) + (48/3)*delta[2]*delta[4]*(T**3) + (36/3)*(delta[3]**2)*(T**3) + (24/2)*delta[2]*delta[3]*(T**2) + 4*(delta[2]**2)*T

    c_sigma = (1/2)*i_sigma
    c_delta = (1/2)*i_delta

    return (1/T)*(c_sigma+c_delta)


def computeSimilarity(init_cost,cur_traj):
    """Compute Similarity between state and goal state"""
    #e^{-||state-goal_state||_2}
    cur_cost = trajCost(cur_traj)
    ################################################
    #This is the defintion used in Stefano's paper
    #sim = abs(cur_cost-init_cost)
    ################################################
    ################################################
    #Proportional cost like this has appeal, but run into issue when magnitudes of both numbers are
    # small, but relatively orders of magnitude different (e.g. e^-31 vs e^-29)
    #This could be resolved by inserting a special condition for having cost<some magnitude just defaulting to 0,
    # and then handle x/0 separately
    precision = 1e-6
    if init_cost<precision: init_cost = precision
    if cur_cost<precision: cur_cost = precision

    sim = cur_cost/init_cost
    ###############################################
    return math.exp(-sim)


def updatePreference(current_traj_list,init_traj_cost_list,veh_pref):
    """We update the preferences of both agents based on the previous observations using a Bayes Filter"""
    new_pref = []
    num_sum = 0

    #uncertainty = random.random()
    uncertainty = .0001

    #Compute Likelihood
    prob_state_given_traj = []
    for traj_init_cost,traj_cur in zip(init_traj_cost_list,current_traj_list):
        prob_state_given_traj.append(computeSimilarity(traj_init_cost,traj_cur))

    try:
        #Bayes Rule to compute posterior
        for i in range(len(current_traj_list)):
            #new_pref_val = veh_pref[i]*prob_state_given_traj[i]
            new_pref_val = (veh_pref[i]+uncertainty)*prob_state_given_traj[i] #still unsure how to include noise
            num_sum += new_pref_val
            new_pref.append(new_pref_val)

        new_pref = [x/num_sum for x in new_pref]
        return new_pref

    except ZeroDivisionError:
        print("Divide by zero error")
        #NOTE: This is kind of hacky. If num_sum is 0 then the likelihood (prob_state_given_traj) is 0 for all entries
        #      So they are all equally unlikely. This should be resolved by checking if the intention specified by
        #      each trajectory has already been satisfied (which means the cost should be based on how well it satisfies the objective)
        #      For the time being we just return the prior unchanged, the result from a uniform likelihood (since all trajectories are uniformly unlikely)
        return veh_pref

    return new_pref

####################################################################################################################
#######Translating from Simulator to Symbolic Model#################################################################

def getParametrisedAcceleration(vel,heading,accel,yaw_rate,axle_length):
    x_dot = vel*math.cos(math.radians(heading))
    y_dot = vel*math.sin(math.radians(heading))
    x_dot_dot = (vel*accel/x_dot) - (y_dot/x_dot)*(1/vel)*(y_dot*accel - (x_dot*(vel**2)*math.tan(math.radians(yaw_rate))/axle_length))
    y_dot_dot = (1/vel)*(y_dot*accel - (x_dot*(vel**2)*math.tan(math.radians(yaw_rate))/axle_length))

    return (x_dot_dot,y_dot_dot)


def filterState(state,axle_length):
    state = dict(state)
    state["heading"] *= -1 #Account for the fact that the y-axis in the simulator is flipped
    state["parametrised_acceleration"] = getParametrisedAcceleration(state["velocity"],state["heading"],state["acceleration"],state["yaw_rate"],axle_length)
    
    return state

#####################################################################################################
#######Intelligent Driver Model######################################################################

def IDM(ego_state,obstacle_state,delta,max_accel,max_decel,v_desired,time_headway,minimum_spacing):
    """Returns the linear acceleration recommended by IDM controller with specified parameters"""
    # Presume obstacle is leading ego
    # Presume vehicles are heading up vertically (so distance between vehicles is distance along y-axis)
    # a is the maximum acceleration rate. b is the maximum permissable/comfortable deceleration rate
    v_ego = ego_state["velocity"]

    if obstacle_state is None:
        dist_term = 0
    else:
        v_obs = obstacle_state["velocity"]
        del_v = v_ego - v_obs
        del_s = abs(ego_state["position"][1] - obstacle_state["position"][1])
        #NOTE: Hardcoding this in to avoid divide by 0 error
        if del_s<2: del_s = 2

        s_star = minimum_spacing + time_headway*v_ego + v_ego*del_v/(2*math.sqrt(max_accel*abs(max_decel))) # ensuring b is positive
        dist_term = s_star/del_s

    accel = max_accel*(1-(v_ego/v_desired)**delta - dist_term)

    return accel
        


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
    #Define IDM controller
    delta = 4
    max_accel = 3
    max_decel = -3
    v_desired = 25 #m/s
    time_headway = 1 #seconds headway
    minimum_spacing = 4 #metres distance bumper to bumper between cars

    idm = lambda x,y: IDM(x,y,delta,max_accel,max_decel,v_desired,time_headway,minimum_spacing)

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

    #NOTE: IGNORE THIS FOR NOW. USED TO CONFIRM BAYES FILTER WAS WORKING
    #Ego car must lead other for IDM experiments
    #ego.v = 0.001 # setting vel=0 creates problems when computer parametrised accel
    #ego.x_com = other.x_com
    #ego.initialisation_params["prev_disp_x"] = other.initialisation_params["prev_disp_x"]
    #ego.y_com = other.y_com-100
    #ego.initialisation_params["prev_disp_y"] = other.initialisation_params["prev_disp_y"]-100

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
    #For computing Rewards
    #Safety Params also go with IDM, but keeping separate for the moment
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
