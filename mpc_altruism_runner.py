#Optimal Control problem using multiple-shooting
#Multiple-shooting: whole state, trajectory and control trajectory, are decision variables

from casadi import *
import math
import matplotlib.pyplot as plt # for  plotting results
import numpy as np # to get teh size of matrices
import random # to add noise in mpc
import time # for pausing when plotting dynamic plots
from trajectory_type_definitions import Trajectory

import pdb

def makeIntegrator(dt):
    ##########################################################
    ########## Initialise Variables ##########################

    #2-D state 
    x = MX.sym('x',4) # state <- x,y,v,heading
    u = MX.sym('u',2) # control input <- a,yaw_rate

    ##########################################################
    ########### Define ODE/Dynamics model  ###################

    #computational graph definition of dynamics model
    #Bicycle model
    L = 4 # Length of vehicle #NOTE: this is hardcoded here
    ode = vertcat(x[2]*cos(x[3]+u[1]),x[2]*sin(x[3]+u[1]),u[0],(2*x[2]/L)*sin(u[1]))

    #f is a function that takes as input x and u and outputs the
    # state specified by the ode

    f = Function('f',[x,u],[ode],['x','u'],['ode']) # last 2 arguments name the inputs/outputs (Optional)
    #f([0.2,0.8],0.1) # to see sample output

    ##########################################################
    ########### Implementing the Integrator ##################
    #N = int(T*(1/dt)) # number of control intervals

    #Options for integrator to discretise the system
    # Options are optional
    intg_options = {}
    intg_options['tf'] = dt
    intg_options['simplify'] = True
    intg_options['number_of_finite_elements'] = 4 #number of intermediate steps to integration (?)

    #DAE problem structure/problem definition
    dae = {}
    dae['x'] = x  #What are states    #Define initial trajectories
    dae['p'] = u  # What are parameters (fixed during integration horizon)
    dae['ode'] = f(x,u) # Expression for x_dot = f(x,u)

    # Integrating using Runga-Kutte integration method
    intg = integrator('intg','rk',dae,intg_options) #function object over CasADi symbols

    #Sample output from integrator
    #res = intg(x0=[0,1],p=0) # include object labels to make it easier to identify inputs
    #res['xf'] #print the final value of x at the end of the integration

    #Can call integrator function symbolically
    res = intg(x0=x,p=u) # no numbers give, just CasADi symbols
    x_next = res['xf']

    #This allows us to simplify API
    # Maps straight from inital state x to final state xf, given control input u
    F = Function('F',[x,u],[x_next],['x','u'],['x_next'])

    #Sample output to test simpler API
    #F([0,1],0)
    #F([0.1,.09],0.1)

    return F


def makeOptimiser(dt,horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range):
    #########################################################
    ##### Make Integrator ###################################
    F = makeIntegrator(dt)

    ##########################################################
    ########### Initialise Optimisation Problem ##############

    N = int(horizon/dt)
    #x_low,x_high,speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
    bounds = [veh_width/2,2*lane_width-veh_width/2,0,speed_limit,0,math.pi,accel_range[0],accel_range[1],\
              yaw_rate_range[0],yaw_rate_range[1]]

    safe_x_radius = veh_width/2 + 1
    safe_y_radius = veh_length/2 + 1 

    opti = casadi.Opti()

    x = opti.variable(4,N+1) # Decision variables for state trajectory
    u = opti.variable(2,N)
    init_state = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x
    dest_state = opti.parameter(4,1)
    x_other = opti.parameter(4,N+1) # (x,y) position for other vehicle at each timestep
    bnd = opti.parameter(10,1)
    opti.set_value(bnd,bounds)

    safety_params = opti.parameter(2,1)
    opti.set_value(safety_params,[safe_x_radius,safe_y_radius])

    weight = opti.parameter(4,1)
    opti.set_value(weight,[50,0,50,10])

    safety_constr = sum2(1-(((x[0,:]-x_other[0,:])/safety_params[0])**2 + ((x[1,:]-x_other[1,:])/safety_params[1])**2))
    
    opti.minimize(sumsqr((x[:,1:]-dest_state)*weight) + 100*safety_constr + 0*sumsqr(u[0,:])) #Distance to destination
    #opti.minimize(sumsqr((x[:,1:]-dest_state)*weight)) # Distance to destination
    #opti.minimize(sumsqr(x-goal) + sumsqr(u)) # Distance to destination
    #opti.minimize(sumsqr(x)+sumsqr(u))

    #This can also be done with functional programming (mapaccum)
    for k in range(N):
        opti.subject_to(x[:,k+1]==F(x[:,k],u[:,k]))
        
        ####################################
        #NOTE: For debug purposes
        #opti.subject_to(x[1,k+1]>=x[1,k])
        #opti.subject_to(-.0174533<=u[1,k])
        #opti.subject_to(u[1,k]<=.0174533)
        ####################################

    #X-coord constraints
    opti.subject_to(bnd[0]<=x[0,:])
    opti.subject_to(x[0,:]<=bnd[1])
    #Velocity Contraints
    opti.subject_to(bnd[2]<=x[2,:])
    opti.subject_to(x[2,:]<=bnd[3])
    #Heading Constraints
    opti.subject_to(bnd[4]<=x[3,:])
    opti.subject_to(x[3,:]<=bnd[5])
    #Accel Constraints
    opti.subject_to(bnd[6]<=u[0,:])
    opti.subject_to(u[0,:]<=bnd[7])
    #Yaw Rate Constraints
    opti.subject_to(bnd[8]<=u[1,:])
    opti.subject_to(u[1,:]<=bnd[9])
    #Initial position contraints
    opti.subject_to(x[:,0]==init_state) #Initial state

    ##########################################
    #NOTE: For debug purposes
    #opti.subject_to(x[1,0]==init_state[1])
    #opti.subject_to(-.0174533<=u[1,0])
    #opti.subject_to(u[1,0]<=.0174533)
    ##########################################

    ###########################################################
    ########### Define Optimizer ##############################

    ipopt_opts = {}
    #Stop IPOPT printing output
    ipopt_opts["ipopt.print_level"] = 0;
    ipopt_opts["ipopt.sb"] = "yes";
    ipopt_opts["print_time"] = 0
    #Cap the maximum number of iterations
    ipopt_opts["ipopt.max_iter"] = 500

    opti.solver('ipopt',ipopt_opts)

    #Turn optimisation to CasADi function
    #M = opti.to_function('M',[init_state,dest_state,x_other],[x[:,1:],u[:,1:]],\
    M = opti.to_function('M',[init_state,dest_state,x_other],[x[:,:],u[:,:]],\
                           ['init','dest','x_other'],['x_opt','u_opt'])

    #M contains SQP method, which maps to a QP solver, all contained in a single, differentiable,
    #computational graph

    return M


def makeJointOptimiser(dt,horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range):
    #########################################################
    ##### Make Integrator ###################################
    F = makeIntegrator(dt)

    ##########################################################
    ########### Initialise Optimisation Problem ##############

    N = int(horizon/dt)
    #x_low,x_high,speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
    bounds = [veh_width/2,2*lane_width-veh_width/2,0,speed_limit,0,math.pi,accel_range[0],accel_range[1],\
              yaw_rate_range[0],yaw_rate_range[1]]

    safe_x_radius = veh_width/2 + 1
    safe_y_radius = veh_length/2 + 1 

    opti = casadi.Opti()

    x1 = opti.variable(4,N+1) # Decision variables for state trajectory
    u1 = opti.variable(2,N)
    init_state1 = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x
    dest_state1 = opti.parameter(4,1)
    
    x2 = opti.variable(4,N+1) # Decision variables for state trajectory
    u2 = opti.variable(2,N)
    init_state2 = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x
    dest_state2 = opti.parameter(4,1)

    bnd = opti.parameter(10,1)
    opti.set_value(bnd,bounds)

    safety_params = opti.parameter(2,1)
    opti.set_value(safety_params,[safe_x_radius,safe_y_radius])

    weight = opti.parameter(4,1)
    opti.set_value(weight,[50,0,50,10])

    opti.minimize(sumsqr((x1[:,-1]-dest_state1)*weight) + 1*sumsqr(u1[0,:]) +\
                    sumsqr((x2[:,-1]-dest_state2)*weight) + 1*sumsqr(u2[0,:])) #Distance to destination

    #This can also be done with functional programming (mapaccum)
    for k in range(N):
        opti.subject_to(x1[:,k+1]==F(x1[:,k],u1[:,k]))
        opti.subject_to(x2[:,k+1]==F(x2[:,k],u2[:,k]))
    
    safety_constr = (((x1[0,:]-x2[0,:])/safety_params[0])**2 + ((x1[1,:]-x2[1,:])/safety_params[1])**2)
    opti.subject_to(safety_constr>=1)
    
        
    #X-coord constraints
    opti.subject_to(bnd[0]<=x1[0,:])
    opti.subject_to(x1[0,:]<=bnd[1])
    #Velocity Contraints
    opti.subject_to(bnd[2]<=x1[2,:])
    opti.subject_to(x1[2,:]<=bnd[3])
    #Heading Constraints
    opti.subject_to(bnd[4]<=x1[3,:])
    opti.subject_to(x1[3,:]<=bnd[5])
    #Accel Constraints
    opti.subject_to(bnd[6]<=u1[0,:])
    opti.subject_to(u1[0,:]<=bnd[7])
    #Yaw Rate Constraints
    opti.subject_to(bnd[8]<=u1[1,:])
    opti.subject_to(u1[1,:]<=bnd[9])
    #Initial position contraints
    opti.subject_to(x1[:,0]==init_state1) #Initial state

    #X-coord constraints
    opti.subject_to(bnd[0]<=x2[0,:])
    opti.subject_to(x2[0,:]<=bnd[1])
    #Velocity Contraints
    opti.subject_to(bnd[2]<=x2[2,:])
    opti.subject_to(x2[2,:]<=bnd[3])
    #Heading Constraints
    opti.subject_to(bnd[4]<=x2[3,:])
    opti.subject_to(x2[3,:]<=bnd[5])
    #Accel Constraints
    opti.subject_to(bnd[6]<=u2[0,:])
    opti.subject_to(u2[0,:]<=bnd[7])
    #Yaw Rate Constraints
    opti.subject_to(bnd[8]<=u2[1,:])
    opti.subject_to(u2[1,:]<=bnd[9])
    #Initial position contraints
    opti.subject_to(x2[:,0]==init_state2) #Initial state

    ###########################################################
    ########### Define Optimizer ##############################

    ipopt_opts = {}
    #Stop IPOPT printing output
    ipopt_opts["ipopt.print_level"] = 0;
    ipopt_opts["ipopt.sb"] = "yes";
    ipopt_opts["print_time"] = 0
    #Cap the maximum number of iterations
    ipopt_opts["ipopt.max_iter"] = 500

    opti.solver('ipopt',ipopt_opts)

    #Turn optimisation to CasADi function
    M = opti.to_function('M',[init_state1,dest_state1,init_state2,dest_state2],\
                            [x1[:,:],u1[:,:],x2[:,:],u2[:,:]],['init1','dest1','init2','dest2'],\
                            ['x1_opt','u1_opt','x2_opt','u2_opt'])

    #M contains SQP method, which maps to a QP solver, all contained in a single, differentiable,
    #computational graph

    return M

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

    #pdb.set_trace()

    for dx,dv in zip([x[0] for x in spec], [x[1] for x in spec]):
        label = "dx-{},dv-{}".format(dx,dv)
        dest_state = dict(init_state)
        dest_state["position"] = tuple([dest_state["position"][0]+dx,dest_state["position"][1]])
        dest_state["velocity"] += dv
        dest_state["parametrised_acceleration"] = (0,0) #parametrised acceleration is introduced to handle acceleration constraints
        traj = Trajectory(cur_state,dest_state,T,label)
        traj_list.append(traj)

    return traj_list


def getParametrisedAcceleration(vel,heading,accel,yaw_rate,axle_length):
    x_dot = vel*math.cos(math.radians(heading))
    y_dot = vel*math.sin(math.radians(heading))
    x_dot_dot = (vel*accel/x_dot) - (y_dot/x_dot)*(1/vel)*(y_dot*accel - (x_dot*(vel**2)*math.tan(math.radians(yaw_rate))/axle_length))
    y_dot_dot = (1/vel)*(y_dot*accel - (x_dot*(vel**2)*math.tan(math.radians(yaw_rate))/axle_length))

    return (x_dot_dot,y_dot_dot)


def filterState(state,axle_length):
    state = dict(state)
    state["heading"] = math.degrees(state["heading"])
    state["parametrised_acceleration"] = getParametrisedAcceleration(state["velocity"],state["heading"],state["acceleration"],state["yaw_rate"],axle_length)

    return state

def makeTrajState(pos_x,pos_y,v,heading,accel,yaw_rate,axle_length):
    return filterState({"position":(pos_x,pos_y),"velocity":v,"heading":heading,"acceleration": accel, "yaw_rate": yaw_rate},axle_length)


###################################################################################################
####### Reward Grid Stuff #########################################################################

def makeBaselineRewardGrid(reward_grid):
    return reward_grid


def makeVanillaAltRewardGrid(reward_grid,alt1,alt2):
    alt_reward = np.copy(reward_grid)
    alt_reward[:,:,0] = (1-alt1)*reward_grid[:,:,0] + alt1*reward_grid[:,:,1]
    alt_reward[:,:,1] = (1-alt2)*reward_grid[:,:,1] + alt2*reward_grid[:,:,0]

    return alt_reward


def makeAugmentedAltRewardGrid(reward_grid,alt1,alt2):
    alt_reward = np.copy(reward_grid)
    alt_reward[:,:,0] = ((1-alt1)*reward_grid[:,:,0] + alt1*(1-alt2)*reward_grid[:,:,1])/(1-alt1*alt2)
    alt_reward[:,:,1] = ((1-alt2)*reward_grid[:,:,1] + alt2*(1-alt1)*reward_grid[:,:,0])/(1-alt1*alt2)

    return alt_reward


def makeSVORewardGrid(reward_grid,svo1,svo2):
    alt_reward = np.copy(reward_grid)
    alt_reward[:,:,0] = math.cos(svo1)*reward_grid[:,:,0] + math.sin(svo1)*reward_grid[:,:,1]
    alt_reward[:,:,1] = math.cos(svo2)*reward_grid[:,:,1] + math.sin(svo2)*reward_grid[:,:,0]

    return alt_reward

        

###################################################################################################
################ Other ############################################################################

def dynamicPlotter(mpc_x1,mpc_x2):
    c1_plt_x = []
    c1_plt_y = []
    c2_plt_x = []
    c2_plt_y = []

    y_lim = max(np.max(mpc_x1[1,:]),np.max(mpc_x2[1,:]))*1.1

    plt.ion()
    plt.figure()
    plt.xlim(0,2*lane_width)
    plt.ylim(0,y_lim)

    for i in range(mpc_x1.shape[1]):
        c1_plt_x.append(mpc_x1[0,i])
        c1_plt_y.append(mpc_x1[1,i])
        c2_plt_x.append(mpc_x2[0,i])
        c2_plt_y.append(mpc_x2[1,i])
        plt.plot(c1_plt_x,c1_plt_y,'g-')
        plt.plot(c2_plt_x,c2_plt_y,'r-')
        plt.draw()
        plt.pause(1e-17)
        time.sleep(dt)


def computeDistance(x1,x2):
    #distance from desired x-position and velocity
    return math.sqrt((x1[0]-x2[0])**2 + (x1[2]-x2[2])**2)

if __name__ == "__main__":
    ###################################
    #Vehicle Dimensions
    veh_length = 4.6
    veh_width = 2

    ###################################
    #Optimiser Parameters
    axle_length = 2.7
    dt = .2
    epsilon = .5
    lane_width = 4
    T = 10 #Trajectory length
    lookahead_horizon = 4 # length of time MPC plans over
    N = int(lookahead_horizon/dt)

    speed_limit = 15
    accel_range = [-9,3] #range of accelerations permissable for optimal control
    yaw_rate_range = [-math.pi/180,math.pi/180]    

    ###################################
    #Defining initial states for both cars
    init_c1_posit = [0.5*lane_width,0] # middle of right lane
    init_c1_vel = 15
    init_c1_heading = math.pi/2    
    init_c1_accel = 0
    init_c1_yaw_rate = 0

    init_c2_posit = [1.5*lane_width,0] # middle of right lane
    init_c2_vel = 15
    init_c2_heading = math.pi/2
    init_c2_accel = 0
    init_c2_yaw_rate = 0

    c1_init_state = makeTrajState(init_c1_posit[0],init_c1_posit[1],init_c1_vel,\
                                  init_c1_heading,init_c1_accel,init_c1_yaw_rate,axle_length)
    c2_init_state = makeTrajState(init_c2_posit[0],init_c2_posit[1],init_c2_vel,\
                                  init_c2_heading,init_c2_accel,init_c2_yaw_rate,axle_length)

    ###################################
    #Define Trajectory Options
    c1_traj_specs = [(0,-14),(lane_width,0)]
    c2_traj_specs = [(0,-7),(0,0)]

    ###################################
    #Define Optimser
    optimiser = makeOptimiser(dt,lookahead_horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)

    ###################################
    #Define Game Theory Stuff
    #Use float values or else numpy will round to int
    #reward_grid = np.array([[[-np.inf,-np.inf],[0,1]],[[1,0],[-np.inf,-np.inf]]])
    reward_grid = np.array([[[-1.0,-1.0],[0.0,1.0]],[[1.0,0.0],[-1.0,-1.0]]])

    a1 = .9
    a2 = .1

    #goal_grid = makeBaselineRewardGrid(reward_grid,a1,a2)
    goal_grid = makeVanillaAltRewardGrid(reward_grid,a1,a2)
    #goal_grid = makeAugmentedAltRewardGrid(reward_grid,a1,a2)
    #goal_grid = makeSVORewardGrid(reward_grid,a1,a2)
    
    #Index of c1's preferred action    
    c1_index = np.unravel_index(np.argmax(goal_grid[:,:,0]),goal_grid[:,:,0].shape)[0]
    #Index of action c1 expects c2 to take (c2's optimal choice if c1 is lead)
    c1_c2_index = np.unravel_index(np.argmax(goal_grid[c1_index,:,1]),\
                          goal_grid[c1_index,:,1].shape)[0]
    #Index of c2's preferred action
    c2_index = np.unravel_index(np.argmax(goal_grid[:,:,1]),goal_grid[:,:,1].shape)[1]
    #Index of action c1 expects c2 to take (c2's optimal choice if c1 is lead)
    c2_c1_index = np.unravel_index(np.argmax(goal_grid[:,c2_index,0]),\
                          goal_grid[:,c2_index,0].shape)[0]

    ########################################################################
    #Comparing MPC with fit trajectory
    c1_traj = makeTrajectories(c1_init_state,[c1_traj_specs[c1_index]],T)[0]
    c1_traj_ssu = sum([x[0]**2+x[1]**2 for x in c1_traj.completeActionList(axle_length,dt)])

    c2_traj = makeTrajectories(c2_init_state,[c2_traj_specs[c2_index]],T)[0]
    c2_traj_ssu = sum([x[0]**2+x[1]**2 for x in c2_traj.completeActionList(axle_length,dt)])
    
    #########################################################################
    #Defining Vehicle States for Optimiser
    c1_x = np.array([*init_c1_posit,init_c1_vel,init_c1_heading]).reshape(4,1)
    c2_x = np.array([*init_c2_posit,init_c2_vel,init_c2_heading]).reshape(4,1)

    #Defining Vehicle actions for Optimiser
    c1_u = np.array([0,0]).reshape(2,1)
    c2_u = np.array([0,0]).reshape(2,1)

    #Recording trajectory generated by MPC loop
    c1_mpc_x,c2_mpc_x = np.array(c1_x),np.array(c2_x)
    c1_mpc_u,c2_mpc_u = np.array(c1_u),np.array(c2_u)
    
    ##########################################################################
    #Defining vehicle states for trajectory definition
    c1_init = np.array([*init_c1_posit,init_c1_vel,init_c1_heading]).reshape(4,1)
    c2_init = np.array([*init_c2_posit,init_c2_vel,init_c2_heading]).reshape(4,1)
    
    c1_dest = np.copy(c1_init)
    c1_dest[0] += c1_traj_specs[c1_index][0]
    c1_dest[2] += c1_traj_specs[c1_index][1]
    
    c2_dest = np.copy(c2_init)
    c2_dest[0] += c2_traj_specs[c2_index][0]
    c2_dest[2] += c2_traj_specs[c2_index][1]

    ########################################################################
    #For testing/debugging joint optimiser function
    #true_c1_index = np.unravel_index(np.argmax(reward_grid[:,:,0]),reward_grid[:,:,0].shape)[0]
    #true_c2_index = np.unravel_index(np.argmax(reward_grid[:,:,1]),reward_grid[:,:,1].shape)[1]
    #
    #true_c1_dest = np.copy(c1_init)
    #true_c1_dest[0] += c1_traj_specs[true_c1_index][0]
    #true_c1_dest[2] += c1_traj_specs[true_c1_index][1]
    #
    #true_c2_dest = np.copy(c2_init)
    #true_c2_dest[0] += c2_traj_specs[true_c2_index][0]
    #true_c2_dest[2] += c2_traj_specs[true_c2_index][1]
    #
    #jointOpt = makeJointOptimiser(dt,4,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)
    #c1_joint_opt_x,c1_joint_opt_u,c2_joint_opt_x,c2_joint_opt_u =\
    #           jointOpt(c1_init,true_c1_dest,c2_init,true_c2_dest)
    #
    #for j in range(c1_joint_opt_x.shape[1]):
    #    dist = math.sqrt((c1_joint_opt_x[0,j]-c2_joint_opt_x[0,j])**2+(c1_joint_opt_x[1,j]-c2_joint_opt_x[1,j])**2)
    #    print("J: {}".format(dist))

    ## Plot Resulting Trajectories

    #dynamicPlotter(c1_joint_opt_x,c2_joint_opt_x)
    #pdb.set_trace()
    ########################################################################

    ##########################################################################
    #MPC Loop
    t = 0
    c1_t,c2_t = None,None #time at which each car completed their true objective
    c1_to_global,c2_to_global = False, False #if car has satisfied true objective
    num_timesteps = 2 # How many timesteps are followed per iteration

    while t<T and (c1_t is None or c2_t is None):
        ###########################################
        #### MPC for C1 ###########################
        #How C1 expects C2 to behave
        c1_c2_traj = makeTrajectories(makeTrajState(*[x[0] for x in c2_x.tolist()],*[x[0] for x in c2_u.tolist()],axle_length),\
                                      [c2_traj_specs[c1_c2_index]],T-t,c2_init_state)[0]
        c1_c2_posit = c1_c2_traj.completePositionList(dt)
        c1_c2_vel = c1_c2_traj.completeVelocityList(dt)
        c1_c2_heading = [math.radians(x) for x in c1_c2_traj.completeHeadingList(dt)]
        # Not enough trajectory left, assume constant velocity thereafter
        if len(c1_c2_posit)<N+1:
            c1_c2_backup_traj = makeTrajectories(c1_c2_traj.state(T-t,axle_length),[(0,0)],T)[0]
            c1_c2_posit += c1_c2_backup_traj.completePositionList(dt)[1:]
            c1_c2_vel += c1_c2_backup_traj.completeVelocityList(dt)[1:]
            c1_c2_heading += [math.radians(x) for x in c1_c2_backup_traj.completeHeadingList(dt)[1:]]
 
        #Behaviour within lookahead horizon
        c1_c2_posit = c1_c2_posit[:N+1]
        c1_c2_vel = c1_c2_vel[:N+1]
        c1_c2_heading = c1_c2_heading[:N+1]

        c1_c2_x = np.array([[x[0] for x in c1_c2_posit],[x[1] for x in c1_c2_posit],\
                            c1_c2_vel,c1_c2_heading])
        
        #Run MPC for C1
        c1_opt_x,c1_opt_u = optimiser(c1_x,c1_dest,c1_c2_x)

        ############################################
        #### MPC for C2 ############################
        #How C2 expects C1 to behave
        c2_c1_traj = makeTrajectories(makeTrajState(*[x[0] for x in c1_x.tolist()],*[x[0] for x in c1_u.tolist()],axle_length),\
                                      [c1_traj_specs[c2_c1_index]],T-t,c1_init_state)[0]
        c2_c1_posit = c2_c1_traj.completePositionList(dt)
        c2_c1_vel = c2_c1_traj.completeVelocityList(dt)
        c2_c1_heading = [math.radians(x) for x in c2_c1_traj.completeHeadingList(dt)]
        # Not enough trajectory left, assume constant velocity thereafter
        if len(c2_c1_posit)<N+1:
            c2_c1_backup_traj = makeTrajectories(c2_c1_traj.state(T-t,axle_length),[(0,0)],T)[0]
            c2_c1_posit += c2_c1_backup_traj.completePositionList(dt)[1:]
            c2_c1_vel += c2_c1_backup_traj.completeVelocityList(dt)[1:]
            c2_c1_heading += [math.radians(x) for x in c2_c1_backup_traj.completeHeadingList(dt)[1:]]

        #Behaviour within lookahead horizon
        c2_c1_posit = c2_c1_posit[:N+1]
        c2_c1_vel = c2_c1_vel[:N+1]
        c2_c1_heading = c2_c1_heading[:N+1]
        c2_c1_x = np.array([[x[0] for x in c2_c1_posit],[x[1] for x in c2_c1_posit],\
                            c2_c1_vel,c2_c1_heading])
        
        #Run MPC for C2
        c2_opt_x,c2_opt_u = optimiser(c2_x,c2_dest,c2_c1_x)
      
        #############################################
        #Debugging
        if True in [round(x,2)<round(c1_x.tolist()[1][0],2) for x in np.array(c1_opt_x[1,:num_timesteps-1]).tolist()[0]]:
            print("New setting of C1_x is behind current")
            import pdb
            pdb.set_trace()
        if True in [round(x,2)<round(c2_x.tolist()[1][0],2) for x in np.array(c2_opt_x[1,:num_timesteps-1]).tolist()[0]]:
            print("New setting of C2_x is behind current")
            import pdb
            pdb.set_trace()

        #############################################

        ##MPC state and action if optimiser does not return current state as first output
        #c1_x = np.array(c1_opt_x[:,num_timesteps-1])
        #c2_x = np.array(c2_opt_x[:,num_timesteps-1])
        #c1_u = np.array(c1_opt_u[:,num_timesteps-1])
        #c2_u = np.array(c2_opt_u[:,num_timesteps-1])

        #MPC state and action if optimiser returns current state as first output
        c1_x = np.array(c1_opt_x[:,num_timesteps])
        c2_x = np.array(c2_opt_x[:,num_timesteps])
        c1_u = np.array(c1_opt_u[:,num_timesteps])
        c2_u = np.array(c2_opt_u[:,num_timesteps])
        
        ##############################################
        #If MPC does not have safety as constraint then test for crash out here
        if (c1_x[0]-c2_x[0])**2+(c1_x[1]-c2_x[1])**2<math.sqrt((veh_length/2)**2+(veh_width/2)**2):
            print("Cars have crashed")
            import pdb
            pdb.set_trace()
        ##############################################

        t += num_timesteps*dt

        ##############################################
        #Debugging
        if True in [c1_opt_x[1,i]>c1_opt_x[1,i+1]+.02 for i in range(c1_opt_x.shape[1]-1)]:
            print("Problem in C1 MPC")
            import pdb
            pdb.set_trace()

        if True in [c2_opt_x[1,i]>c2_opt_x[1,i+1]+.02 for i in range(c2_opt_x.shape[1]-1)]:
            print("Problem in C2 MPC")
            import pdb
            pdb.set_trace()
        ###############################################

        ##############################################
        #Store MPC generated trajectories
        c1_mpc_x = np.hstack((c1_mpc_x,np.array(c1_opt_x[:,:num_timesteps])))
        c2_mpc_x = np.hstack((c2_mpc_x,np.array(c2_opt_x[:,:num_timesteps])))
        c1_mpc_u = np.hstack((c1_mpc_u,np.array(c1_opt_u[:,:num_timesteps])))
        c2_mpc_u = np.hstack((c2_mpc_u,np.array(c2_opt_u[:,:num_timesteps])))

        ################################################
        #If C1 satisfies their current objective
        if c1_t is None and computeDistance(c1_x,c1_dest)<epsilon:
            #Objective is "true" objective
            if c1_to_global or max(reward_grid[c1_index,:,0]) == 1:
                c1_to_global = True
                c1_t = t #Time C1 satisfied trajectory
                print("C1_T set: {}".format(c1_t))
            #Objective is not "true" objective, c1 presumes c2 has been accommodated
            # pursues their own objective.
            else:
               c1_index = np.unravel_index(np.argmax(reward_grid[:,:,0]),reward_grid[:,:,0].shape)[0]
               c1_dest = np.copy(c1_init)
               c1_dest[0] += c1_traj_specs[c1_index][0]
               c1_dest[2] += c1_traj_specs[c1_index][1]
               c1_to_global = True #Now definitely going to global objective
               print("Changing C1 Index: {}".format(c1_index))

               #Used for computing relative jerk used
               c1_traj = makeTrajectories(makeTrajState(*[x[0] for x in c1_x.tolist()],*[x[0] for x in c1_u.tolist()],axle_length),\
                                      [c1_traj_specs[c1_index]],T-t,c1_init_state)[0]
               c1_traj_ssu += sum([x[0]**2+x[1]**2 for x in c1_traj.completeActionList(axle_length,dt)])

        #C1 has drifted from their objective, reset value
        elif c1_t is not None and computeDistance(c1_x,c1_dest)>epsilon: c1_t = None

        ###############################################
        #If C2 satisfies their current objective
        if c2_t is None and computeDistance(c2_x,c2_dest)<epsilon:
            #Objective is "true" objective
            if c2_to_global or max(reward_grid[:,c2_index,1]) == 1:
                c2_to_global = True
                c2_t = t
                print("C2_T set: {}".format(c2_t))
            #Objective is not "true" objective, c2 presumes c1 has been accomodated
            # pursues their own objective. 
            else:
               c2_index = np.unravel_index(np.argmax(reward_grid[:,:,1]),reward_grid[:,:,1].shape)[1]
               c2_dest = np.copy(c2_init)
               c2_dest[0] += c2_traj_specs[c2_index][0]
               c2_dest[2] += c2_traj_specs[c2_index][1]
               c2_to_global = True #Now definitely going to global objective
               print("Changing C2 Index: {}".format(c2_index))

               #Used for computing relative jerk used
               c2_traj = makeTrajectories(makeTrajState(*[x[0] for x in c2_x.tolist()],*[x[0] for x in c2_u.tolist()],axle_length),\
                                      [c2_traj_specs[c2_index]],T-t,c2_init_state)[0]
               c2_traj_ssu += sum([x[0]**2+x[1]**2 for x in c2_traj.completeActionList(axle_length,dt)])

        #C" has drifted from their objective, reset value.
        elif c2_t is not None and computeDistance(c2_x,c2_dest)>epsilon: c2_t = None

        print("T is: {}\tD1: {}\t D2: {}".format(t,computeDistance(c1_x,c1_dest),computeDistance(c2_x,c2_dest)))


    #print("MPC Complete")
    #t2 = datetime.datetime.now()
    #print("Time: {}".format(t2-t1))
    #pdb.set_trace()

    ########################################################################
    #Comparing MPC with fit trajectory
    true_c1_index = np.unravel_index(np.argmax(reward_grid[:,:,0]),reward_grid[:,:,0].shape)[0]
    true_c2_index = np.unravel_index(np.argmax(reward_grid[:,:,1]),reward_grid[:,:,1].shape)[1]
    
    true_c1_dest = np.copy(c1_init)
    true_c1_dest[0] += c1_traj_specs[true_c1_index][0]
    true_c1_dest[2] += c1_traj_specs[true_c1_index][1]
    
    true_c2_dest = np.copy(c2_init)
    true_c2_dest[0] += c2_traj_specs[true_c2_index][0]
    true_c2_dest[2] += c2_traj_specs[true_c2_index][1]
 
    jointOpt = makeJointOptimiser(dt,t,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)
    c1_joint_opt_x,c1_joint_opt_u,c2_joint_opt_x,c2_joint_opt_u =\
               jointOpt(c1_init,true_c1_dest,c2_init,true_c2_dest)
    
    c1_joint_ssu = np.sum(c1_joint_opt_u**2)
    c2_joint_ssu = np.sum(c2_joint_opt_u**2)

    c1_mpc_ssu = np.sum(c1_mpc_u**2)
    c2_mpc_ssu = np.sum(c2_mpc_u**2)

    c1_ssu_diff = c1_mpc_ssu - c1_joint_ssu
    c2_ssu_diff = c2_mpc_ssu - c2_joint_ssu

    print("C1 SSU: {}\tOPT: {}\tDiff: {}".format(c1_mpc_ssu,c1_joint_ssu,c1_ssu_diff))
    print("C2 SSU: {}\tOPT: {}\tDiff: {}".format(c2_mpc_ssu,c2_joint_ssu,c2_ssu_diff))
    print("MPC SSU Total: {}\t OPT SSU Total: {}\t Diff Total: {}".format(c1_mpc_ssu+c2_mpc_ssu,c1_joint_ssu+c2_joint_ssu,c1_ssu_diff+c2_ssu_diff))

    # Plot Resulting Trajectories
    pdb.set_trace()
    dynamicPlotter(c1_mpc_x,c2_mpc_x)
    dynamicPlotter(c1_joint_opt_x,c2_joint_opt_x)
    pdb.set_trace()

#####################################
