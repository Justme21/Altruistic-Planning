from casadi import *
import math
import matplotlib.pyplot as plt # for  plotting results
import numpy as np # to get teh size of matrices
import random # to add noise in mpc
import time # for pausing when plotting dynamic plots
from trajectory_type_definitions import Trajectory

import pdb

def makeIntegrator(dt,veh_length):
    ##########################################################
    ########## Initialise Variables ##########################

    #2-D state 
    x = MX.sym('x',4) # state <- x,y,v,heading
    u = MX.sym('u',2) # control input <- a,yaw_rate

    ##########################################################
    ########### Define ODE/Dynamics model  ###################

    #computational graph definition of dynamics model
    #Bicycle model
    L = veh_length # Length of vehicle #NOTE: this is hardcoded here
    ode = vertcat(x[2]*cos(x[3]+u[1]),x[2]*sin(x[3]+u[1]),u[0],(2*x[2]/L)*sin(u[1]))

    #f is a function that takes as input x and u and outputs the
    # state specified by the ode

    f = Function('f',[x,u],[ode],['x','u'],['ode']) # last 2 arguments name the inputs/outputs (Optional)

    ##########################################################
    ########### Implementing the Integrator ##################

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

    #Can call integrator function symbolically
    res = intg(x0=x,p=u) # no numbers give, just CasADi symbols
    x_next = res['xf'] #final value of x at the end of the integration

    #This allows us to simplify API
    # Maps straight from inital state x to final state xf, given control input u
    F = Function('F',[x,u],[x_next],['x','u'],['x_next'])

    return F

def makeIDMModel(has_new_leader,accel_bounds,accel_range,v_goal,d_goal,T_safe,veh_length):
    x_ego = MX.sym('IDM',4) # state <- x,y,v,heading
    x_lead = MX.sym('other',4) # state <- x,y,v,heading
    v_dot = MX.sym('vdot',1) # control input <- a,yaw_rate

    del_v = x_ego[2]-x_lead[2]
    d_star = d_goal + T_safe*x_ego[2] + x_ego[2]*del_v/(2*sqrt(accel_bounds[1]*(-accel_bounds[0])))
    
    d = x_lead[1]-x_ego[1] - veh_length #rear to front distance = midpoint_distance-rear_half_of_lead-front_half_of_follower
    
    v_dot = if_else(has_new_leader==1.0,accel_bounds[1]*(1-(x_ego[2]/v_goal)**4-(d_star/d)**2),\
                                        accel_bounds[1]*(1-(x_ego[2]/v_goal)**4))

    idm = Function('idm',[x_ego,x_lead],[v_dot],['x_ego','x_leader'],['v_dot'])
    return idm


def makeJointIDMOptimiser(dt,horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range):
    #########################################################
    ##### Make Integrator ###################################
    F = makeIntegrator(dt,veh_length)
    ##########################################################
    ########### Initialise Optimisation Problem ##############

    N = int(horizon/dt)
    #x_low,x_high,speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
    bounds = [veh_width/2,2*lane_width-veh_width/2,0,speed_limit,0,math.pi,accel_range[0],accel_range[1],\
              yaw_rate_range[0],yaw_rate_range[1]]

    safe_x_radius = veh_width
    safe_y_radius = veh_length

    opti = casadi.Opti()

    #IDM Model
    other_has_lead = opti.parameter(1,1)
    comfort_accel_range = [-2.5,2] # NOTE: Manually specifying values
    FIDM = makeIDMModel(other_has_lead,comfort_accel_range,accel_range,15,.1,0.1,veh_length)

    #Optimisation Parameters
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

    #Optimisation
    #Minimise trajectory duration for planning car
    c1_traj_duration_weight = opti.parameter(4,1)
    opti.set_value(c1_traj_duration_weight,[1,0,1,1])
    c1_min_traj_duration = sumsqr((x1[:,:]-dest_state1)*c1_traj_duration_weight)
    #Minimise final distance from objective for planning car
    c1_final_distance_weight = opti.parameter(4,1)
    opti.set_value(c1_final_distance_weight,[1,0,1,100])
    c1_min_final_dist = sumsqr((x1[:,-1]-dest_state1)*c1_final_distance_weight)
    #Minimise Acceleration Magnitude
    c1_action_weight = opti.parameter(2,1)
    opti.set_value(c1_action_weight,[0,10])
    c1_min_accel = sumsqr(u1*c1_action_weight)
    #Minimise Jerk
    c1_jerk_weight = opti.parameter(2,1)
    opti.set_value(c1_jerk_weight,[0,0])
    c1_min_jerk = sumsqr((u1[:,1:]-u1[:,:-1])*c1_jerk_weight)

    #Minimise trajectory duration for other car
    c2_traj_duration_weight = opti.parameter(4,1)
    #Velocity of c2 dictated by IDM, so don't drive to velocity
    opti.set_value(c2_traj_duration_weight,[0,0,0,0])
    c2_min_traj_duration = sumsqr((x2[:,:]-dest_state2)*c2_traj_duration_weight)
    #Minimise final distance from objective for other car
    c2_final_distance_weight = opti.parameter(4,1)
    opti.set_value(c2_final_distance_weight,[0,0,0,0])
    c2_min_final_dist = sumsqr((x2[:,-1]-dest_state2)*c2_final_distance_weight)
    #Minimise Acceleration Magnitude
    c2_action_weight = opti.parameter(2,1)
    opti.set_value(c2_action_weight,[0,10]) #[5,100]
    c2_min_accel = sumsqr(u2*c2_action_weight)
    #Minimise Jerk
    c2_jerk_weight = opti.parameter(2,1)
    opti.set_value(c2_jerk_weight,[0,0])
    c2_min_jerk = sumsqr((u2[:,1:]-u2[:,:-1])*c2_jerk_weight)

    #Encourage other vehicle action solution to follow specified IDM model
    c2_to_idm_weight = 1 #10
    c2_to_idm = c2_to_idm_weight*sum([sumsqr(u2[0,k]-FIDM(x2[:,k],x1[:,k])) for k in range(N)])

    #Encourage cars to stay maximise distance between each other
    safety_weight = 0
    safety = safety_weight*sumsqr(1-(((x1[0,:]-x2[0,:])/safety_params[0])**2 + \
                          ((x1[1,:]-x2[1,:])/safety_params[1])**2))

    opti.minimize(c1_min_traj_duration+c1_min_final_dist+c1_min_accel+c1_min_jerk+\
                   c2_min_traj_duration+c2_min_final_dist+c2_min_accel+c2_min_jerk+\
                   c2_to_idm+safety)

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
    M = opti.to_function('M',[init_state1,dest_state1,init_state2,dest_state2,other_has_lead],\
                            [x1[:,:],u1[:,:],x2[:,:],u2[:,:]],['init1','dest1','init2','dest2','other_has_lead'],\
                            ['x1_opt','u1_opt','x2_opt','u2_opt'])

    return M

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
    lookahead_horizon = 6 # length of time MPC plans over
    N = int(lookahead_horizon/dt)

    speed_limit = 15
    accel_range = [-9,3] #range of accelerations permissable for optimal control
    yaw_rate_range = [-math.pi/180,math.pi/180]    

    ###################################
    #Defining initial states for both cars
    init_c1_posit = [0.5*lane_width,2.5*veh_length] # middle of right lane
    init_c1_vel = 15
    init_c1_heading = math.pi/2    
    init_c1_accel = 0
    init_c1_yaw_rate = 0

    init_c2_posit = [0.5*lane_width,0] # middle of right lane
    init_c2_vel = 15
    init_c2_heading = math.pi/2
    init_c2_accel = 0
    init_c2_yaw_rate = 0

    ###################################
    #Define Optimser
    optimiser = makeJointIDMOptimiser(dt,lookahead_horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)

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
    c1_dest[0] += 0
    c1_dest[2] += 0
    
    c2_dest = np.copy(c2_init)
    c2_dest[0] += 0
    c2_dest[2] += 0

    c1_joint_opt_x,c1_joint_opt_u,c2_joint_opt_x,c2_joint_opt_u =\
               optimiser(c1_init,c1_dest,c2_init,c2_dest,1)
 
    for j in range(c1_joint_opt_x.shape[1]):
        dist = math.sqrt((c1_joint_opt_x[0,j]-c2_joint_opt_x[0,j])**2+(c1_joint_opt_x[1,j]-c2_joint_opt_x[1,j])**2)
        dx = abs(c1_joint_opt_x[0,j]-c2_joint_opt_x[0,j])
        dy = abs(c1_joint_opt_x[1,j]-c2_joint_opt_x[1,j])
        print("J: {}\tD: {}\t Dx: {}\t Dy: {}".format(j,dist,dx,dy))

    print("\n")

    for i in range(c2_joint_opt_u.shape[1]):
        print("{}\tD: {}\tV2: {}\tU2:{}\tV1: {}\tU1: {}".format(i,c1_joint_opt_x[1,i]-c2_joint_opt_x[1,i] - veh_length,c2_joint_opt_x[2,i],c2_joint_opt_u[:,i],c1_joint_opt_x[2,i],c1_joint_opt_u[:,i]))

    # Plot Resulting Trajectories

    dynamicPlotter(c1_joint_opt_x,c2_joint_opt_x)
    pdb.set_trace()
    ########################################################################

