import math
import numpy as np
import sys

sys.path.insert(0,'../Driving-Simulator')
sys.path.insert(0,'../driving_simulator')
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


#####################################################################################################
#Define Cost Function
def trajCost(traj,T,w):
    delta = [x[0] for x in traj.line_x.coefs]
    sigma = [y[0] for y in traj.line_y.coefs]
   
    #Given in paper
    #c_delta = ((T**6)/7)*(delta[3]**2) + ((T**5)/3)*(delta[3]*sigma[2]) + ((T**4)/5)*(delta[2]**2)
    #c_sigma = ((T**4)/5)*(sigma[2]**2) + ((T**3)/2)*(sigma[3]*sigma[2]) + ((T**2)/3)*(sigma[1]**2)
    #return (w/(1+w))*c_delta + (1/(1+w))*c_sigma

    #Derived by hand from trajectory definitions
    i_sigma = (144/5)*(sigma[4]**2)*(T**5) + (144/4)*sigma[3]*sigma[4]*(T**4) + (36/3)*(sigma[3]**2)*(T**3)
    i_delta = (400/7)*(delta[5]**2)*(T**7) + (480/6)*delta[4]*delta[5]*(T**6) + (240/5)*delta[3]*delta[5]*(T**5) + (144/5)*(delta[4]**2)*(T**5) +\
                     (144/4)*delta[3]*delta[4]*(T**4) + (36/3)*(delta[3]**2)*(T**3)
    
    c_sigma = (1/1+w)*i_sigma
    c_delta = (w/1+w)*i_delta

    return (1/T)*(c_sigma+c_delta)

######################################################################################################
#State stuff
def getParametrisedAcceleration(vel,heading,accel,yaw_rate,axle_length):
    x_dot = vel*math.cos(math.radians(heading))
    y_dot = vel*math.sin(math.radians(heading))
    try:
        x_dot_dot = (vel*accel/x_dot) - (y_dot/x_dot)*(1/vel)*(y_dot*accel - (x_dot*(vel**2)*math.tan(math.radians(yaw_rate))/axle_length))
    except ZeroDivisionError:
        x_dot_dot = 0
    try:
        y_dot_dot = (1/vel)*(y_dot*accel - (x_dot*(vel**2)*math.tan(math.radians(yaw_rate))/axle_length))
    except ZeroDivisionError:
        y_dot_dot = 0

    return (x_dot_dot,y_dot_dot)

######################################################################################################
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


def computeCost(traj1,traj2,traj1_cost_function,traj2_cost_function,traj1_alt,traj2_alt,dt,safety_params):
    if checkForCrash(traj1,traj2,dt,safety_params):
        return np.inf, np.inf
    else:
        traj1_cost = traj1_cost_function(traj1)
        traj2_cost = traj2_cost_function(traj2)
        return (1-traj1_alt)*traj1_cost + traj1_alt*traj2_cost,traj2_alt*traj1_cost + (1-traj2_alt)*traj2_cost


def computeCost2(label1,label2,traj1_cost_function,traj2_cost_function,traj1_alt,traj2_alt,dt,safety_params):
    if label1 == "Stop":
        if label2 == "Accelerate": r1,r2 = 10,0
        else: r1,r2 = 100,10
    else:
        if label2 == "Accelerate": r1,r2 = np.inf,np.inf
        else: r1,r2 = 0,10

    return (1-traj1_alt)*r1 + traj1_alt*r2,traj2_alt*r1 + (1-traj2_alt)*r2


def trajTest(traj,T,speed_limit,accel_range,lane_width):
    deltas = [x[0] for x in traj.line_x.coefs]
    sigmas = [y[0] for y in traj.line_y.coefs]
    ##########################
    #Speed Limit
    if sigmas[4]!=0:
        vel_extremum = ((sigmas[3])**3)/(4*(sigmas[4]**2)) + traj.velocity(0)[1] # this presumes that car is initially travelling in longitudinal direction
    else:
        #if sigmas[4] == 0 then acceleration is linear, so check velocity at extremes to ensure within bounds
        vel_extremum = 4*sigmas[4]*(T**3) + 3*sigmas[3]*(T**2) + sigmas[1] # sigma[2] = 0 by assumption.
    if vel_extremum>speed_limit or vel_extremum<0: 
        print("Rejected for exceeding speed constraint: {} (0,{})".format(vel_extremum,speed_limit))
        return False
    ##########################
    ##########################
    #Accel Limits
    if sigmas[4] != 0:
        accel_extremum = (-3/4)*(sigmas[3]**2)/(sigmas[4])
    else:
        #if sigmas[4] == 0 then acceleration is linear, initial accel is presumed to be 0, so only need to check if final acceleration within limits
        accel_extremum = 12*sigmas[4]*(T**2) + 6*sigmas[3]*T + 2*sigmas[2]
    if accel_extremum<accel_range[0] or accel_extremum>accel_range[1]:
        print("Rejected for exceeding acceleration constraints: {} ({},{})".format(accel_extremum,accel_range[0],accel_range[1]))
        return False
    ##########################
    ##########################
    #Going off the Road
    posit = traj.position(T)
    #This works on the assumption road starts right on edge of screen and there are only 2 lanes
    #if posit[0]<0 or posit[0]>2*lane_width: return False
    ##########################
    print("Passed Traj Test")
    return True


if __name__ == "__main__":
    ###########################################
    #PREAMBLE
    ###################################
    #Define Vehicles
    debug = False
    dt = .1
    ego = Car(controller=None,is_ego=True,debug=debug,label="Ego",timestep=dt)
    other = Car(controller=None,is_ego=False,debug=debug,label="Other",timestep=dt)
    
    ###################################
    ###################################
    #Initialise Simulator
    safety_distance_s = 5
    safety_distance_d = 2
    time_radius = 1
    safety_params = (time_radius,safety_distance_d,safety_distance_s)
    
    wd = 1
    dt = 1
    T = 10
    
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

    other.y_com = ego.y_com + 20
    other.initialisation_params["prev_disp_y"] = ego.initialisation_params["prev_disp_y"] + 20
    other.sense()

    ###################################
    ###################################
    ##Define Trajectory Options
    #dx_dict = {}
    #dx_dict[ego] = [lane_width]
    #dx_dict[other] = [lane_width,0]
    #
    #dv_dict = {}
    #dv_dict[ego] = [-10,-5,0,5,10]
    #dv_dict[other] = [-10,-5,0,5,10]
    #    
    ###################################
    ###########################################
    
    ###########################################
    #RUNTIME
    ###################################
    #Initialise Trajectory options for each vehicle and each trajectory type and target velocity
    #trajectories = {}
    #for veh in [ego,other]:
    #    traj_list = []
    #    init_state = dict(veh.state)
    #    for dx in dx_dict[veh]:
    #        for dv in dv_dict[veh]:
    #            dest_state = dict(init_state)
    #            dest_state["position"] = tuple([dest_state["position"][0]+dx,dest_state["position"][1]])
    #            dest_state["velocity"] += dv
    #            traj = Trajectory(init_state,dest_state,T)
    #            ###################################
    #            #Eliminate Trajectory options that violate constraints
    #            #Omit Trajectories that don't satify constraints (speed limit,etc.)
    #            print("Testing ({},{}) for {}. Cost: {}".format(dx,dv,veh.label,trajCost(traj,T,wd)))
    #            if trajTest(traj,T,speed_limit,accel_range,lane_width): traj_list.append(traj)
    #            print("\n")
    #            ###################################
    #
    #    print("\n")
    #
    #    trajectories[veh] = list(traj_list)
    
    trajectories = {}
    trajectories[ego] = []
    ego_labels = ["Stop","Lane Change"]
    other_labels = ["Accelerate","Give Way"]
    ego_init_state = dict(ego.state)
    ego_init_state["parametrised_acceleration"] = getParametrisedAcceleration(ego_init_state["velocity"],ego_init_state["heading"],ego_init_state["acceleration"],ego_init_state["yaw_rate"],axle_length=ego.length)
    dest_state = dict(ego_init_state)
    dest_state["velocity"] =0
    dest_state["parametrised_acceleration"] = getParametrisedAcceleration(dest_state["velocity"],dest_state["heading"],dest_state["acceleration"],dest_state["yaw_rate"],axle_length=ego.length)
    trajectories[ego].append(Trajectory(ego_init_state,dest_state,T))
    dest_state = dict(ego_init_state)
    dest_state["position"] = tuple([dest_state["position"][0]+lane_width,dest_state["position"][1]])
    dest_state["parametrised_acceleration"] = getParametrisedAcceleration(dest_state["velocity"],dest_state["heading"],dest_state["acceleration"],dest_state["yaw_rate"],axle_length=ego.length)
    trajectories[ego].append(Trajectory(ego_init_state,dest_state,T))

    trajectories[other] = []
    other_init_state = dict(other.state)
    other_init_state["parametrised_acceleration"] = getParametrisedAcceleration(other_init_state["velocity"],other_init_state["heading"],other_init_state["acceleration"],other_init_state["yaw_rate"],axle_length=other.length)
    dest_state = dict(other_init_state)
    dest_state["velocity"] += 10
    dest_state["parametrised_acceleration"] = getParametrisedAcceleration(dest_state["velocity"],dest_state["heading"],dest_state["acceleration"],dest_state["yaw_rate"],axle_length=other.length)
    trajectories[other].append(Trajectory(other_init_state,dest_state,T))
    dest_state = dict(other_init_state)
    dest_state["velocity"] -= 10
    dest_state["parametrised_acceleration"] = getParametrisedAcceleration(dest_state["velocity"],dest_state["heading"],dest_state["acceleration"],dest_state["yaw_rate"],axle_length=other.length)
    trajectories[other].append(Trajectory(other_init_state,dest_state,T))

    ###################################
    ###################################
    #Run Decision-making algorithm
    ego_alt = .49
    other_alt = 1
    min_ego_cost,min_other_cost,min_ego_traj,min_other_traj,min_ego_traj_label,min_other_traj_label = None,None,None,None,None,None
    for ego_label,ego_traj in zip(ego_labels,trajectories[ego]):
        for other_label,other_traj in zip(other_labels,trajectories[other]):
            #ego_cost,other_cost = computeCost(ego_traj,other_traj,lambda x: trajCost(x,T,wd),lambda x: trajCost(x,T,wd),ego_alt,other_alt,dt,safety_params)
            ego_cost,other_cost = computeCost2(ego_label,other_label,lambda x: trajCost(x,T,wd),lambda x: trajCost(x,T,wd),ego_alt,other_alt,dt,safety_params)
            print("Got to here: {}\t{}".format(ego_cost,other_cost))
            if min_ego_cost is None or ego_cost<min_ego_cost:
                min_ego_cost = ego_cost
                min_ego_traj = ego_traj
                min_ego_traj_label = ego_label
            if min_other_cost is None or other_cost<min_other_cost:
                min_other_cost = other_cost
                min_other_traj = other_traj
                min_other_traj_label = other_label
    ###################################
    ###################################
    #Implement Result
    ego_init = min_ego_traj.init_state
    ego_dest = min_ego_traj.dest_state
    other_init = min_other_traj.init_state
    other_dest = min_other_traj.dest_state
    print("Optimal Paths Found")
    print("Ego: Label: {}\t Init: {}\tDest: {}".format(min_ego_traj_label,(ego_init["position"],ego_init["velocity"]),(ego_dest["position"],ego_dest["velocity"])))
    print("Other: Label: {}\t Init: {}\tDest: {}".format(min_other_traj_label,(other_init["position"],other_init["velocity"]),(other_dest["position"],other_dest["velocity"])))
    import pdb
    pdb.set_trace()
    ###################################
    ###########################################
