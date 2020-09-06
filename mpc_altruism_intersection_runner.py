from casadi import *
import math
import matplotlib.pyplot as plt # for  plotting results
import numpy as np # to get teh size of matrices
import random # to add noise in mpc
import time # for pausing when plotting dynamic plots

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


def makeJointIntersectionOptimiser(dt,horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range):
    #########################################################
    ##### Make Integrator ###################################
    F = makeIntegrator(dt,veh_length)
    ##########################################################
    ########### Initialise Optimisation Problem ##############

    N = int(horizon/dt)
    #speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
    bounds = [0,speed_limit,accel_range[0],accel_range[1],yaw_rate_range[0],yaw_rate_range[1]]

    safe_x_radius = veh_width + .25
    safe_y_radius = veh_length + .5 

    opti = casadi.Opti()

    #Parameters identifying the presumed leader and follower roles
    is_lead1 = opti.parameter(1,1)
    is_lead2 = opti.parameter(1,1)

    #Optimisation Parameters
    x1 = opti.variable(4,N+1) # Decision variables for state trajectory
    u1 = opti.variable(2,N)
    init_state1 = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x
    
    x2 = opti.variable(4,N+1) # Decision variables for state trajectory
    u2 = opti.variable(2,N)
    init_state2 = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x

    bnd = opti.parameter(6,1)
    opti.set_value(bnd,bounds)

    safety_params = opti.parameter(2,1)
    opti.set_value(safety_params,[safe_y_radius,safe_x_radius])

    #Optimisation
    #Minimise trajectory duration for planning car
    init_lane1 = opti.parameter(3,1)
    init_dx_lane1 = sumsqr(x1[0,:]-init_lane1[0])
    init_dy_lane1 = sumsqr(x1[1,:]-init_lane1[1])
    init_dv_lane1 = sumsqr(speed_limit-x1[2,:])
    init_dtheta_lane1 = sumsqr(init_lane1[2]-x1[3,:])
    init_lane_dist1 = 0*vertcat(init_dx_lane1,init_dy_lane1,init_dv_lane1,init_dtheta_lane1)

    dest_lane1 = opti.parameter(3,1)
    #dest_dx_lane1 = sumsqr(fmax(dest_lane1[0]-x1[0,:],0))
    #dest_dy_lane1 = sumsqr(fmax(dest_lane1[1]-x1[1,:],0))
    dest_dx_lane1 = sumsqr(cos(dest_lane1[2])*fmax(dest_lane1[0]-x1[0,-1],0)+\
                           sin(dest_lane1[2])*(dest_lane1[0]-x1[0,:]))
    dest_dy_lane1 = sumsqr(sin(dest_lane1[2])*fmax(dest_lane1[1]-x1[1,-1],0)+\
                           cos(dest_lane1[2])*(dest_lane1[1]-x1[1,:]))
    dest_dv_lane1 = sumsqr(speed_limit-x1[2,:])
    dest_dtheta_lane1 = sumsqr(dest_lane1[2]-x1[3,:])
    dest_lane_dist1 = 1*vertcat(dest_dx_lane1,dest_dy_lane1,dest_dv_lane1,dest_dtheta_lane1)

    c1_traj_duration_weight = opti.parameter(4,1) 
    opti.set_value(c1_traj_duration_weight,[1,1,1,10])
    c1_min_traj_duration = sum1(dest_lane_dist1*c1_traj_duration_weight)
    #c1_min_traj_duration = sum1((init_lane_dist1+dest_lane_dist1)*c1_traj_duration_weight)
    #Minimise Acceleration Magnitude
    #c1_action_weight = opti.parameter(2,1)
    #opti.set_value(c1_action_weight,[0,0])
    #c1_min_accel = sumsqr(u1*c1_action_weight)
    #Minimise Jerk
    #c1_jerk_weight = opti.parameter(2,1)
    #opti.set_value(c1_jerk_weight,[0,0])
    #c1_min_jerk = sumsqr((u1[:,1:]-u1[:,:-1])*c1_jerk_weight)

    #If the car has a leader, motivate it to get behind the other car
    centre_dx_lane1 = sumsqr(cos(dest_lane1[2])*fmax(x2[0,:]-x1[0,:],0))
    centre_dy_lane1 = sumsqr(sin(dest_lane1[2])*fmax(x2[1,:]-x1[1,:],0))
    c1_lead_weight = 500*is_lead1
    c1_behind_mid = sum2(centre_dx_lane1+centre_dy_lane1)*c1_lead_weight

    #Minimise trajectory duration for other car
    init_lane2 = opti.parameter(3,1)
    init_dx_lane2 = sumsqr(x2[0,:]-init_lane2[0])
    init_dy_lane2 = sumsqr(x2[1,:]-init_lane2[1])
    init_dv_lane2 = sumsqr(speed_limit-x2[2,:])
    init_dtheta_lane2 = sumsqr(init_lane2[2]-x2[3,:])
    init_lane_dist2 = 0*vertcat(init_dx_lane2,init_dy_lane2,init_dv_lane2,init_dtheta_lane2)

    dest_lane2 = opti.parameter(3,1)
    #dest_dx_lane2 = sumsqr(fmax(dest_lane2[0]-x2[0,:],0))
    #dest_dy_lane2 = sumsqr(fmax(dest_lane2[1]-x2[0,:],0))
    dest_dx_lane2 = sumsqr(cos(dest_lane2[2])*fmax(dest_lane2[0]-x2[0,-1],0)+\
                           sin(dest_lane2[2])*(dest_lane2[0]-x2[0,:]))
    dest_dy_lane2 = sumsqr(sin(dest_lane2[2])*fmax(dest_lane2[1]-x2[1,-1],0)+\
                           cos(dest_lane2[2])*(dest_lane2[1]-x2[1,:]))
    dest_dv_lane2 = sumsqr(speed_limit-x2[2,:])
    dest_dtheta_lane2 = sumsqr(dest_lane2[2]-x2[3,:])
    dest_lane_dist2 = 1*vertcat(dest_dx_lane2,dest_dy_lane2,dest_dv_lane2,dest_dtheta_lane2)

    c2_traj_duration_weight = opti.parameter(4,1) 
    opti.set_value(c2_traj_duration_weight,[1,1,1,10])
    c2_min_traj_duration = sum1(dest_lane_dist2*c2_traj_duration_weight)
    #c2_min_traj_duration = sum1((init_lane_dist2+dest_lane_dist2)*c2_traj_duration_weight)
    #Minimise Acceleration Magnitude
    #c2_action_weight = opti.parameter(2,1)
    #opti.set_value(c2_action_weight,[0,0]) #[5,100]
    #c2_min_accel = sumsqr(u2*c2_action_weight)
    #Minimise Jerk
    #c2_jerk_weight = opti.parameter(2,1)
    #opti.set_value(c2_jerk_weight,[0,0])
    #c2_min_jerk = sumsqr((u2[:,1:]-u2[:,:-1])*c2_jerk_weight)

    #If the car has a leader, motivate it to get behind the other car
    centre_dx_lane2 = sumsqr(cos(dest_lane2[2])*fmax(x1[0,:]-x2[0,:],0))
    centre_dy_lane2 = sumsqr(sin(dest_lane2[2])*fmax(x1[1,:]-x2[1,:],0))
    c2_lead_weight = 500*is_lead2
    c2_behind_mid = sum2(centre_dx_lane2+centre_dy_lane2)*c2_lead_weight

    #Encourage cars to stay maximise distance between each other
    safety_weight = 0
    safety = safety_weight*sumsqr(1-(((x1[0,:]-x2[0,:])/safety_params[1])**2 + \
                          ((x1[1,:]-x2[1,:])/safety_params[0])**2))

    ######################################################
    #Debugging
    #c1_min_traj_duration=0
    #c1_min_accel = 0
    #c1_min_jerk = 0
    #c1_behind_mid = 0
    #c2_min_traj_duration=0
    #c2_min_accel = 0
    #c2_min_jerk = 0
    #c2_behind_mid = 0
    #safety = 0
    #######################################################

    opti.minimize(c1_min_traj_duration+c1_behind_mid+\
                  c2_min_traj_duration+c2_behind_mid+\
                  safety)

    for k in range(N):
        opti.subject_to(x1[:,k+1]==F(x1[:,k],u1[:,k]))
        opti.subject_to(x2[:,k+1]==F(x2[:,k],u2[:,k]))
    
    safety_constr = (((x1[0,:]-x2[0,:])/safety_params[1])**2 + ((x1[1,:]-x2[1,:])/safety_params[0])**2)
    opti.subject_to(safety_constr>=1)
    
    #re_crash_1 = ((cos(x1[3,:])*(x2[0,:]-x1[0,:]) + sin(x1[3,:])*(x2[1,:]-x1[1,:]))**2)/(safety_params[0])**2 + ((sin(x1[3,:])*(x2[0,:]-x1[0,:]) - cos(x1[3,:])*(x2[1,:]-x1[1,:]))**2)/(safety_params[1])**2
    #re_crash_2 = ((cos(x2[3,:])*(x1[0,:]-x2[0,:]) + sin(x2[3,:])*(x1[1,:]-x2[1,:]))**2)/(safety_params[0])**2 + ((sin(x2[3,:])*(x1[0,:]-x2[0,:]) - cos(x2[3,:])*(x1[1,:]-x2[1,:]))**2)/(safety_params[1])**2
    
    #opti.subject_to(re_crash_1>=1)
    #opti.subject_to(re_crash_2>=1)
    
    #Velocity Contraints
    opti.subject_to(bnd[0]<=x1[2,:])
    opti.subject_to(x1[2,:]<=bnd[1])
    #Heading Constraints
    opti.subject_to(dest_lane1[2]-math.pi/180<=x1[3,:])
    opti.subject_to(x1[3,:]<=dest_lane1[2]+math.pi/180)
    #Accel Constraints
    opti.subject_to(bnd[2]<=u1[0,:])
    opti.subject_to(u1[0,:]<=bnd[3])
    #Yaw Rate Constraints
    opti.subject_to(bnd[4]<=u1[1,:])
    opti.subject_to(u1[1,:]<=bnd[5])
    #Initial position contraints
    opti.subject_to(x1[:,0]==init_state1) #Initial state
    opti.subject_to(u1[1,:]==0) #NOTE: Cars can't turn

    #Velocity Contraints
    opti.subject_to(bnd[0]<=x2[2,:])
    opti.subject_to(x2[2,:]<=bnd[1])
    #Heading Constraints
    opti.subject_to(dest_lane2[2]-math.pi/180<=x2[3,:])
    opti.subject_to(x2[3,:]<=dest_lane2[2]+math.pi/180)
    #Accel Constraints
    opti.subject_to(bnd[2]<=u2[0,:])
    opti.subject_to(u2[0,:]<=bnd[3])
    #Yaw Rate Constraints
    opti.subject_to(bnd[4]<=u2[1,:])
    opti.subject_to(u2[1,:]<=bnd[5])
    #Initial position contraints
    opti.subject_to(x2[:,0]==init_state2) #Initial state
    opti.subject_to(u2[1,:]==0) #NOTE: Cars can't turn

    ###########################################################
    ########### Define Optimizer ##############################

    ipopt_opts = {}
    #Stop IPOPT printing output
    ipopt_opts["ipopt.print_level"] = 0;
    ipopt_opts["ipopt.sb"] = "yes";
    ipopt_opts["print_time"] = 0
    #Cap the maximum number of iterations
    ipopt_opts["ipopt.max_iter"] = 1500

    opti.solver('ipopt',ipopt_opts)
    
    #Turn optimisation to CasADi function
    M = opti.to_function('M',[init_state1,init_lane1,dest_lane1,is_lead1,init_state2,\
                      init_lane2,dest_lane2,is_lead2],[x1[:,:],u1[:,:],x2[:,:],u2[:,:]],\
                      ['init1','initlane1','destlane1','is_lead1','init2','initlane2','destlane2',\
                       'is_lead2'],['x1_opt','u1_opt','x2_opt','u2_opt'])

    return M

#####################################################################################################
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

def dynamicIntersectionPlotter(mpc_x1,mpc_x2,mdpt,lane_width):
    c1_plt_x = []
    c1_plt_y = []
    c2_plt_x = []
    c2_plt_y = []

    #y_lim = max(np.max(mpc_x1[1,:]),np.max(mpc_x2[1,:]))*1.1
    #x_lim = max(np.max(mpc_x1[0,:]),np.max(mpc_x2[0,:]))*1.1
    y_lim = 6*lane_width
    x_lim = 6*lane_width


    plt.ion()
    plt.figure()
    plt.xlim(0,x_lim)
    plt.ylim(0,y_lim)

    for i in range(mpc_x1.shape[1]):
        plt.plot([0,mdpt[0]-lane_width,mdpt[0]-lane_width],[mdpt[1]+lane_width,mdpt[1]+lane_width,y_lim],'k-')
        plt.plot([x_lim,mdpt[0]+lane_width,mdpt[0]+lane_width],[mdpt[1]+lane_width,mdpt[1]+lane_width,y_lim],'k-')
        plt.plot([0,mdpt[0]-lane_width,mdpt[0]-lane_width],[mdpt[1]-lane_width,mdpt[1]-lane_width,0],'k-')
        plt.plot([x_lim,mdpt[0]+lane_width,mdpt[0]+lane_width],[mdpt[1]-lane_width,mdpt[1]-lane_width,0],'k-')

        plt.plot([midpoint[0],midpoint[0]],[0,y_lim],'y--')
        plt.plot([0,x_lim],[midpoint[1],midpoint[1]],'y--')

        c1_plt_x.append(mpc_x1[0,i])
        c1_plt_y.append(mpc_x1[1,i])
        c2_plt_x.append(mpc_x2[0,i])
        c2_plt_y.append(mpc_x2[1,i])
        plt.plot(c1_plt_x,c1_plt_y,'g-')
        plt.plot(c2_plt_x,c2_plt_y,'r-')
        plt.draw()
        plt.pause(1e-17)
        time.sleep(dt)


def computeDistance(x,lane):
    #distance from desired x-position and heading
    return math.sqrt((x[0]-lane[0])**2 + (x[1]-lane[1])**2 + (x[3]-lane[2])**2)

def finCheck(x,lane,lane_width):
    #Check point has passed objective
    obj_val = math.cos(lane[2])*(lane[0]-x[0]) + math.sin(lane[2])*(lane[1]-x[1])
    obj_satisfied = obj_val<0
    #Check on the lane
    lane_val = math.cos(lane[2])*abs(lane[1]-x[1]) + math.sin(lane[2])*abs(lane[0]-x[0])
    lane_satisfied = lane_val<lane_width/2
    #Check with the right heading
    heading_val = abs(lane[2]-x[3])
    heading_satisfied = heading_val<math.pi/90 # 2 degree tolerance
    return obj_satisfied and lane_satisfied and heading_satisfied
###################################################################################################


if __name__ == "__main__":
    ###################################
    #Vehicle Dimensions
    veh_length = 4.6
    veh_width = 2

    ###################################
    #Optimiser Parameters
    axle_length = 2.7
    dt = .2
    epsilon = .05
    lane_width = 4
    T = 10 #Trajectory length
    lookahead_horizon = 4 # length of time MPC plans over
    N = int(lookahead_horizon/dt)

    speed_limit = 15
    accel_range = [-9,3] #range of accelerations permissable for optimal control
    yaw_rate_range = [-math.pi/180,math.pi/180]    

    ###################################
    #Defining initial states for both cars
    delta = 2
    init_c1_posit = [veh_length+delta+lane_width/2,.5*veh_length-1.15] # middle of right lane
    init_c1_vel = 0
    init_c1_heading = math.pi/2 
    init_c1_accel = 0
    init_c1_yaw_rate = 0

    init_c2_posit = [.5*veh_length-4.6,veh_length+delta+1.5*lane_width] # middle of right lane
    init_c2_vel = 0
    init_c2_heading = 0
    init_c2_accel = 0
    init_c2_yaw_rate = 0

    ###################################
    #Define Trajectory Options
    c1_lead = [1,0]
    c2_lead = [1,0]

    #Definition of endpoints of lanes leading into the intersection
    in_lanes = [[veh_length+delta+1.5*lane_width,veh_length+delta+1*lane_width,3*math.pi/2],\
                [veh_length+delta+1*lane_width,veh_length+delta+lane_width/2,math.pi],\
                [veh_length+delta+lane_width/2,veh_length+delta+lane_width,math.pi/2],\
                [veh_length+delta+lane_width,veh_length+delta+1.5*lane_width,0]]

    #Definition of endpoints of lanes leading out of intersection
    out_lanes =[[veh_length+delta+.5*lane_width,veh_length+delta+2*lane_width+veh_length,math.pi/2],\
                [veh_length+delta+2*lane_width+veh_length,veh_length+delta+1.5*lane_width,0],\
                [veh_length+delta+1.5*lane_width,0,3*math.pi/2],\
                [0,veh_length+delta+0.5*lane_width,math.pi]]

    c1_init_lane,c2_init_lane = in_lanes[2],in_lanes[3]
    c1_dest_lane,c2_dest_lane = out_lanes[0],out_lanes[1]
    ###################################
    #Define Optimser
    optimiser = makeJointIntersectionOptimiser(dt,lookahead_horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)

    ###################################
    #Define Game Theory Stuff
    #Use float values or else numpy will round to int
    #reward_grid = np.array([[[-np.inf,-np.inf],[0,1]],[[1,0],[-np.inf,-np.inf]]])
    reward_grid = np.array([[[-1.0,-1.0],[1.0,0.0]],[[0.0,1.0],[-1.0,-1.0]]])

    a1 = .1
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

    #########################################################################
    #Defining Vehicle States for Optimiser
    c1_x = np.array([*init_c1_posit,init_c1_vel,init_c1_heading]).reshape(4,1)
    c2_x = np.array([*init_c2_posit,init_c2_vel,init_c2_heading]).reshape(4,1)

    #Recording trajectory generated by MPC loop
    c1_mpc_x,c2_mpc_x = np.array(c1_x),np.array(c2_x)
    c1_mpc_u,c2_mpc_u = np.array([0,0]).reshape(2,1),np.array([0,0]).reshape(2,1)
    
    ##########################################################################
    #Defining vehicle states for trajectory definition
    c1_init = np.array([*init_c1_posit,init_c1_vel,init_c1_heading]).reshape(4,1)
    c2_init = np.array([*init_c2_posit,init_c2_vel,init_c2_heading]).reshape(4,1)
    
    ########################################################################
    #For testing/debugging joint optimiser function
    true_c1_index = np.unravel_index(np.argmax(goal_grid[:,:,0]),goal_grid[:,:,0].shape)[0]
    true_c2_index = np.unravel_index(np.argmax(goal_grid[:,:,1]),goal_grid[:,:,1].shape)[1]
    
    true_c1_is_lead = c1_lead[true_c1_index]
    true_c2_is_lead = c2_lead[true_c2_index]
    
    c2_c2_joint_opt_x,c2_c2_joint_opt_u,c2_c1_joint_opt_x,c2_c1_joint_opt_u =\
              optimiser(c2_init,c2_init_lane,c2_dest_lane,true_c2_is_lead,c1_init,c1_init_lane,\
              c1_dest_lane,true_c1_is_lead)
    c1_c1_joint_opt_x,c1_c1_joint_opt_u,c1_c2_joint_opt_x,c1_c2_joint_opt_u =\
              optimiser(c1_init,c1_init_lane,c1_dest_lane,true_c1_is_lead,c2_init,c2_init_lane,\
              c2_dest_lane,true_c2_is_lead)

    # Plot Resulting Trajectories
    midpoint = [veh_length+delta+lane_width,veh_length+delta+lane_width]
    #dynamicIntersectionPlotter(c1_c1_joint_opt_x,c1_c2_joint_opt_x,midpoint,lane_width)
    pdb.set_trace()

    ########################################################################

    ##########################################################################
    #MPC Loop
    c1_c1_is_lead = c1_lead[c1_index] #if c1 thinks they are going to cut ahead
    c1_c2_is_lead = c1_lead[c1_c2_index] #if c1 thinks c2 will give way
    c2_c1_is_lead = c2_lead[c2_c1_index] #if c2 thinks c1 will give way
    c2_c2_is_lead = c2_lead[c2_index] #if c2 thinks they are expected to continue

    #c2_c2_joint_opt_x,c2_c2_joint_opt_u,c2_c1_joint_opt_x,c2_c1_joint_opt_u =\
    #          optimiser(c2_init,c2_init_lane,c2_dest_lane,c2_c2_is_lead,c1_init,c1_init_lane,\
    #          c1_dest_lane,c2_c1_is_lead)
    #c1_c1_joint_opt_x,c1_c1_joint_opt_u,c1_c2_joint_opt_x,c1_c2_joint_opt_u =\
    #          optimiser(c1_init,c1_init_lane,c1_dest_lane,c1_c1_is_lead,c2_init,c2_init_lane,\
    #          c2_dest_lane,c1_c2_is_lead)
    #pdb.set_trace()
    t = 0
    c1_t,c2_t = None,None #time at which each car completed their true objective
    num_timesteps = 2 # How many timesteps are followed per iteration
     
    print("T is: {}\tD1: {}\t D2: {}".format(t,computeDistance(c1_x,c1_dest_lane),computeDistance(c2_x,c2_dest_lane)))
 
    prev_d1 = computeDistance(c1_x,c1_dest_lane)
    prev_d2 = computeDistance(c2_x,c2_dest_lane)

    while t<T and (c1_t is None or c2_t is None):
        ###########################################
        #### MPC for C1 ###########################
        c1_opt_x,c1_opt_u,c1_c2_opt_x,c1_c2_opt_u = optimiser(c1_x,c1_init_lane,c1_dest_lane,\
                                 c1_c1_is_lead,c2_x,c2_init_lane,c2_dest_lane,c1_c2_is_lead)

        ############################################
        #### MPC for C2 ############################
        c2_opt_x,c2_opt_u,c2_c1_opt_x,c2_c1_opt_u = optimiser(c2_x,c2_init_lane,c2_dest_lane,\
                                 c2_c2_is_lead,c1_x,c1_init_lane,c1_dest_lane,c2_c1_is_lead)
     
        #############################################
        #Debugging
        #if True in [round(x,2)<round(c1_x.tolist()[1][0],2) for x in np.array(c1_opt_x[1,:num_timesteps-1]).tolist()[0]]:
        #    print("New setting of C1_x is behind current")
        #    import pdb
        #    pdb.set_trace()
        #if True in [round(x,2)<round(c2_x.tolist()[1][0],2) for x in np.array(c2_opt_x[1,:num_timesteps-1]).tolist()[0]]:
        #    print("New setting of C2_x is behind current")
        #    import pdb
        #    pdb.set_trace()

        #############################################
        ##############################################
        #Debugging
        #if True in [c1_opt_x[1,i]>c1_opt_x[1,i+1]+.02 for i in range(c1_opt_x.shape[1]-1)]:
        #    print("Problem in C1 MPC")
        #    import pdb
        #    pdb.set_trace()
        #
        #if True in [c2_opt_x[1,i]>c2_opt_x[1,i+1]+.02 for i in range(c2_opt_x.shape[1]-1)]:
        #    print("Problem in C2 MPC")
        #    import pdb
        #    pdb.set_trace()

        #while (True in [c1_opt_x[1,i]>c1_opt_x[1,i+1]+.02 for i in range(c1_opt_x.shape[1]-1)]) or \
        #   (True in [c2_opt_x[1,i]>c2_opt_x[1,i+1]+.02 for i in range(c2_opt_x.shape[1]-1)]):
            #if True in [c1_opt_x[1,i]>c1_opt_x[1,i+1]+.02 for i in range(c1_opt_x.shape[1]-1)]:
            #    print("Problem in C1 MPC")
            #    import pdb
            #    pdb.set_trace()

            #if True in [c2_opt_x[1,i]>c2_opt_x[1,i+1]+.02 for i in range(c2_opt_x.shape[1]-1)]:
            #    print("Problem in C2 MPC")
            #    import pdb
            #    pdb.set_trace()

            #for j in range(c1_x.shape[0]):
            #    c1_x[j,0] = round(c1_x[j,0],1)
            #    c2_x[j,0] = round(c2_x[j,0],1)

            #c1_opt_x,c1_opt_u,c1_c2_opt_x,c1_c2_opt_u = optimiser(c1_x,c1_dest,c1_c1_has_lead,c2_x,c1_c2_dest,c1_c2_has_lead)
            #c2_opt_x,c2_opt_u,c2_c1_opt_x,c2_c1_opt_u = optimiser(c2_x,c2_dest,c2_c2_has_lead,c1_x,c2_c1_dest,c2_c1_has_lead)
            #import pdb
            #pdb.set_trace()
        ###############################################

        for j in range(num_timesteps):
            u1 = np.array(c1_opt_u[:,j])
            u2 = np.array(c2_opt_u[:,j])

            c1_x = np.array(c1_opt_x[:,j+1])
            c2_x = np.array(c2_opt_x[:,j+1])

            ##############################################
            #If MPC does not have safety as constraint then test for crash out here
            crash_check = (((c1_x[0,:]-c2_x[0,:])/veh_width)**2 + ((c1_x[1,:]-c2_x[1,:])/veh_length)**2)
            if crash_check<1:
                print("Cars have crashed")
                import pdb
                pdb.set_trace()
            ##############################################

            re_crash_1 = ((math.cos(c1_x[3])*(c2_x[0]-c1_x[0]) + math.sin(c1_x[3])*(c2_x[1]-c1_x[1]))**2)/(veh_length/2)**2 + ((math.sin(c1_x[3])*(c2_x[0]-c1_x[0]) - math.cos(c1_x[3])*(c2_x[1]-c1_x[1]))**2)/(veh_width/2)**2
            re_crash_2 = ((math.cos(c2_x[3])*(c1_x[0]-c2_x[0]) + math.sin(c2_x[3])*(c1_x[1]-c2_x[1]))**2)/(veh_length/2)**2 + ((math.sin(c2_x[3])*(c1_x[0]-c2_x[0]) - math.cos(c2_x[3])*(c1_x[1]-c2_x[1]))**2)/(veh_width/2)**2

            print("Alt_crash 1: {}\t Alt_Crash 2: {}".format(re_crash_1,re_crash_2))

            ##############################################
            #Store MPC generated trajectories
            c1_mpc_u = np.hstack((c1_mpc_u,np.array(u1)))
            c2_mpc_u = np.hstack((c2_mpc_u,np.array(u2)))
            c1_mpc_x = np.hstack((c1_mpc_x,np.array(c1_x)))
            c2_mpc_x = np.hstack((c2_mpc_x,np.array(c2_x)))

        t += num_timesteps*dt


        ################################################
        #If C1 satisfies their current objective
        #c1_check_val = math.cos(c1_dest_lane[2])*(c1_dest_lane[0]-c1_x[0]) + \
        #                   math.sin(c1_dest_lane[2])*(c1_dest_lane[1]-c1_x[1])
        c1_fin = finCheck(c1_x,c1_dest_lane,lane_width)
        #if c1_t is None and c1_check_val<0 and computeDistance(c1_x,c1_dest_lane)<5:
        if c1_t is None and c1_fin:
            c1_t = t #Time C1 satisfied trajectory
            print("C1_T set: {}".format(c1_t))
        #C1 has drifted from their objective, reset value
        #elif c1_t is not None and c1_check_val>0: c1_t = None
        elif c1_t is not None and not c1_fin: c1_t = None

        ###############################################
        #If C2 satisfies their current objective
        #c2_check_val = math.cos(c2_dest_lane[2])*(c2_dest_lane[0]-c2_x[0]) + \
        #                   math.sin(c2_dest_lane[2])*(c2_dest_lane[1]-c2_x[1])
        c2_fin = finCheck(c2_x,c2_dest_lane,lane_width)
        #if c2_t is None and c2_check_val<0 and computeDistance(c2_x,c2_dest_lane)<5:
        if c2_t is None and c2_fin:
            c2_t = t
            print("C2_T set: {}".format(c2_t))
        #C2 has drifted from their objective, reset value.
        #elif c2_t is not None and c2_check_val>0: c2_t = None
        elif c2_t is not None and not c2_fin: c2_t = None

        if computeDistance(c1_x,c1_dest_lane)<computeDistance(c1_x,c1_init_lane):
            print("Changing C1_lane_init")
            c1_init_lane = c1_dest_lane # c1 has left starting lane, shouldn't be pulled back
        if computeDistance(c2_x,c2_dest_lane)<computeDistance(c2_x,c2_init_lane):
            print("Changing C2_lane_init")
            c2_init_lane = c2_dest_lane # c2 has left starting lane, shouldn't be pulled back

        #if c1_t is None and prev_d1<computeDistance(c1_x,c1_dest_lane):
        #    print("Break for C1")
        #    pdb.set_trace()
        #if c2_t is None and prev_d2<computeDistance(c2_x,c2_dest_lane):
        #    print("Break for C2")
        #    pdb.set_trace()
        
        prev_d1 = computeDistance(c1_x,c1_dest_lane)
        prev_d2 = computeDistance(c2_x,c2_dest_lane)

        print("T is: {}\tD1: {}\t D2: {}".format(t,computeDistance(c1_x,c1_dest_lane),computeDistance(c2_x,c2_dest_lane)))


    print("MPC Complete")
    #t2 = datetime.datetime.now()
    #print("Time: {}".format(t2-t1))
    #pdb.set_trace()

    dynamicIntersectionPlotter(c1_mpc_x,c2_mpc_x,midpoint,lane_width)
    pdb.set_trace()

#####################################
