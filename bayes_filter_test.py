import math
import numpy as np
import random
import sys

sys.path.insert(0,'../Driving-Simulator')
import simulator
from vehicle_classes import Car

from trajectory_type_definitions import Trajectory


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


def makeTrajectories(cur_state,dx_list,dv_list,T,init_state=None):
    traj_list = []

    # Init_state is the state the trajectories are supposed to have originated at. If init_state is None then
    # assume the current state is the initial state of the trajectory
    if init_state is None:
        init_state = cur_state

    for dx in dx_list:
        for dv in dv_list:
            label = "dx-{},dv-{}".format(dx,dv)
            dest_state = dict(init_state)
            dest_state["position"] = tuple([dest_state["position"][0]+dx,dest_state["position"][1]])
            dest_state["velocity"] += dv
            dest_state["parametrised_acceleration"] = (0,0) #parametrised acceleration is introduced to handle acceleration constraints
            traj = Trajectory(cur_state,dest_state,T,label)
            traj_list.append(traj)

    return traj_list

######################################################################################################################
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

####################################################################################################################
#######Intelligent Driver Model#####################################################################################

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

        s_star = minimum_spacing + time_headway*v_ego + v_ego*del_v/(2*math.sqrt(max_accel*abs(max_decel))) # ensuring b is positive
        dist_term = s_star/del_s

    accel = max_accel*(1-(v_ego/v_desired)**delta - dist_term)

    return accel
        


####################################################################################################################

if __name__ == "__main__":
    ###########################################
    #PREAMBLE
    ###################################
    #Define Vehicles
    debug = False
    T = 10
    dt = .1

    ego = Car(controller=None,is_ego=True,debug=debug,label="Ego",timestep=dt)
    other = Car(controller=None,is_ego=False,debug=debug,label="Other",timestep=dt)
    
    ###################################
    #Define IDM controller
    delta = 4
    max_accel = 3
    max_decel = -3
    v_desired = 25 #m/s
    time_headway = 2 #seconds headway
    minimum_spacing = 2 #metres distance bumper to bumper between cars

    idm = lambda x,y: IDM(x,y,delta,max_accel,max_decel,v_desired,time_headway,minimum_spacing)

    ###################################
    ###################################
    #Initialise Simulator
    lane_width = 4 
    graphics = False
    speed_limit = 29 #m/s
    accel_range=[-9,3]
    init_speeds = [25,25] #m/s
    sim = initialiseSimulator([ego,other],speed_limit,graphics,init_speeds,lane_width=lane_width)

    #Reorient Cars (cars initially face same direction as lane they are on)
    other.heading = (other.heading+180)%360
    other.initialisation_params["heading"] = other.heading
    other.sense()

    ego.v = 0
    ego.x_com = other.x_com # Put ego in same lane as other
    ego.initialisation_params["prev_disp_x"] = other.initialisation_params["prev_disp_x"]
    other.y_com = ego.y_com + 100 # Move other behind ego
    other.initialisation_params["prev_disp_y"] = ego.initialisation_params["prev_disp_y"] + 100
    
    ego.sense()
    other.sense()

    ###################################
    ###################################
    ##Define Trajectory Options
    dx_dict = {}
    dx_dict[ego] = [lane_width]
    dx_dict[other] = [lane_width,0]
    
    dv_dict = {}
    dv_dict[ego] = [-10,-5,0,5,10]
    dv_dict[other] = [-10,-5,0,5,10]
    #dv_dict[other] = [-10,10]
        
    ###################################
    ###########################################
    
    ###########################################
    #RUNTIME
    ###################################
    #Initialise Trajectories, Priors and labels
    init_trajectory_costs = {}
    for veh in [ego,other]:
        veh_state = dict(veh.state)
        veh_state["parametrised_acceleration"] = (0,0) #initial acceleration is 0
        veh_state["heading"] *= -1 #trying to account for orientation flip between car and coordinate frame
        init_trajectory_costs[veh] = [trajCost(x) for x in makeTrajectories(veh_state,dx_dict[veh],dv_dict[veh],T)]

    labels = {}
    for veh in [ego,other]:
        labels[veh] = []
        for dx in dx_dict[veh]:
            for dv in dv_dict[veh]:
                labels[veh].append("dx-{},dv-{}".format(dx,dv))

    other_prior = [1/len(labels[other]) for _ in range(len(labels[other]))] 
    ###################################
    ###################################
    #Run Simulation with Bayes Filter
    #chosen_trajectory = "dx-{},dv-{}".format(lane_width,-10)
    chosen_trajectory = "dx-{},dv-{}".format(lane_width,-5)
    traj_index = labels[other].index(chosen_trajectory)

    print("\n\n######## Bayes Filter Starts Here #################\n")

    other_state = filterState(other.state,other.Lf+other.Lr)

    other_trajectory = makeTrajectories(other_state,dx_dict[other],dv_dict[other],T)[traj_index]

    orig_state = dict(other_state)

    print("Index: {}".format(traj_index))

    for t in np.arange(0,T,dt):
        cur_trajectories = {}
        
        cur_trajectories[other] = makeTrajectories(other_state,dx_dict[other],dv_dict[other],T-t,orig_state)
        #for veh in [ego,other]:
        #    cur_trajectories[veh] = makeTrajectories(veh.state,dx_dict[veh],dv_dict[veh],T-t)

        other_prior = updatePreference(cur_trajectories[other],init_trajectory_costs[other],other_prior)

        if t%.5==0:
            print("T: {}\t{}".format(t,[round(x,2) for x in other_prior]))

        #action = (idm(other.state,ego.state),0)
        action = (idm(other.state,None),0) #IDM going in a straight line
        #action = other_trajectory.action(t,other.Lf+other.Lr) # Following preset trajectory
        other.setAction(*action)
        other.move()
        other.sense()
        #other_state = other_trajectory.state(t+dt)
        other_state = filterState(other.state,other.Lf+other.Lr)
        #print("Other: Posit: {} V: {} Act: ({},{})\t Obstacle: {}\n".format(other_state["position"],other_state["velocity"],other_state["acceleration"],other_state["yaw_rate"],ego.state["position"]))

        #if t>=4:
        #    import pdb
        #    pdb.set_trace()

        #cur_trajectory = cur_trajectories[other][traj_index]
        #print("T: {}\t State: {}".format(t,other_state))
        
    ###################################
    import pdb
    pdb.set_trace()
    ###########################################
