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


def makeJointLaneChangeOptimiser(dt,horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range):
    #########################################################
    ##### Make Integrator ###################################
    F = makeIntegrator(dt,veh_length)
    ##########################################################
    ########### Initialise Optimisation Problem ##############

    N = int(horizon/dt)
    #x_low,x_high,speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
    bounds = [veh_width/2,2*lane_width-veh_width/2,0,speed_limit,0,math.pi,accel_range[0],accel_range[1],\
              yaw_rate_range[0],yaw_rate_range[1]]

    safe_x_radius = veh_width + .5
    safe_y_radius = veh_length + 1 

    opti = casadi.Opti()

    #Parameters identifying the presumed leader and follower roles
    has_lead1 = opti.parameter(1,1)
    has_lead2 = opti.parameter(1,1)

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
    opti.set_value(c1_traj_duration_weight,[2,0,1,0])
    c1_min_traj_duration = sumsqr((x1[:,:]-dest_state1)*c1_traj_duration_weight)
    #Minimise final distance from objective for planning car
    c1_final_distance_weight = opti.parameter(4,1)
    opti.set_value(c1_final_distance_weight,[2,0,1,0])
    c1_min_final_dist = sumsqr((x1[:,-1]-dest_state1)*c1_final_distance_weight)
    #Minimise Acceleration Magnitude
    c1_action_weight = opti.parameter(2,1)
    opti.set_value(c1_action_weight,[0,0])
    c1_min_accel = sumsqr(u1*c1_action_weight)
    #Minimise Jerk
    c1_jerk_weight = opti.parameter(2,1)
    opti.set_value(c1_jerk_weight,[0,0])
    c1_min_jerk = sumsqr((u1[:,1:]-u1[:,:-1])*c1_jerk_weight)

    #If the car has a leader, motivate it to get behind the other car
    c1_behind_c2_weight = 10*has_lead1
    c1_behind_c2 = sum2(fmax(x1[1,:]-x2[1,:],0))*c1_behind_c2_weight

    #Minimise trajectory duration for other car
    c2_traj_duration_weight = opti.parameter(4,1)
    opti.set_value(c2_traj_duration_weight,[2,0,1,0])
    c2_min_traj_duration = sumsqr((x2[:,:]-dest_state2)*c2_traj_duration_weight)
    #Minimise final distance from objective for other car
    c2_final_distance_weight = opti.parameter(4,1)
    opti.set_value(c2_final_distance_weight,[2,0,1,0])
    c2_min_final_dist = sumsqr((x2[:,-1]-dest_state2)*c2_final_distance_weight)
    #Minimise Acceleration Magnitude
    c2_action_weight = opti.parameter(2,1)
    opti.set_value(c2_action_weight,[0,0]) #[5,100]
    c2_min_accel = sumsqr(u2*c2_action_weight)
    #Minimise Jerk
    c2_jerk_weight = opti.parameter(2,1)
    opti.set_value(c2_jerk_weight,[0,0])
    c2_min_jerk = sumsqr((u2[:,1:]-u2[:,:-1])*c2_jerk_weight)

    #If the car has a leader, motivate it to get behind the other car
    c2_behind_c1_weight = 10*has_lead2
    c2_behind_c1 = sum2(fmax(x2[1,:]-x1[1,:],0))*c2_behind_c1_weight

    #Encourage cars to stay maximise distance between each other
    safety_weight = 0
    safety = safety_weight*sumsqr(1-(((x1[0,:]-x2[0,:])/safety_params[0])**2 + \
                          ((x1[1,:]-x2[1,:])/safety_params[1])**2))

    opti.minimize(c1_min_traj_duration+c1_min_final_dist+c1_min_accel+c1_min_jerk+c1_behind_c2+\
                   c2_min_traj_duration+c2_min_final_dist+c2_min_accel+c2_min_jerk+c2_behind_c1+\
                   safety)

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
    ipopt_opts["ipopt.max_iter"] = 1500

    opti.solver('ipopt',ipopt_opts)
    
    #Turn optimisation to CasADi function
    M = opti.to_function('M',[init_state1,dest_state1,has_lead1,init_state2,dest_state2,has_lead2],\
                            [x1[:,:],u1[:,:],x2[:,:],u2[:,:]],['init1','dest1','has_lead2','init2','dest2','has_lead1'],\
                            ['x1_opt','u1_opt','x2_opt','u2_opt'])

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
    #distance from desired x-position and heading
    return math.sqrt((x1[0]-x2[0])**2 + (x1[3]-x2[3])**2)
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
    init_c1_posit = [0.5*lane_width,0*veh_length] # middle of right lane
    init_c1_vel = 15
    init_c1_heading = math.pi/2 
    init_c1_accel = 0
    init_c1_yaw_rate = 0

    init_c2_posit = [1.5*lane_width,0*veh_length] # middle of right lane
    init_c2_vel = 15
    init_c2_heading = math.pi/2
    init_c2_accel = 0
    init_c2_yaw_rate = 0

    ###################################
    #Define Trajectory Options
    c1_lead = [1,0]
    c2_lead = [1,0]

    ###################################
    #Define Optimser
    optimiser = makeJointLaneChangeOptimiser(dt,lookahead_horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)

    ###################################
    #Define Game Theory Stuff
    #Use float values or else numpy will round to int
    #reward_grid = np.array([[[-np.inf,-np.inf],[0,1]],[[1,0],[-np.inf,-np.inf]]])
    reward_grid = np.array([[[-1.0,-1.0],[0.0,1.0]],[[1.0,0.0],[-1.0,-1.0]]])

    a1 = .1
    a2 = .9

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
    
    c1_dest = np.copy(c1_init)
    c1_dest[0] += lane_width
    c1_dest[2] += 0
    
    c1_c2_dest = np.copy(c2_init)
    c1_c2_dest[0] += 0
    c1_c2_dest[2] += 0

    c2_dest = np.copy(c2_init)
    c2_dest[0] += 0
    c2_dest[2] += 0

    c2_c1_dest = np.copy(c1_init)
    c2_c1_dest[0] += lane_width
    c2_c1_dest[2] += 0

    ########################################################################
    #For testing/debugging joint optimiser function
    true_c1_index = np.unravel_index(np.argmax(goal_grid[:,:,0]),goal_grid[:,:,0].shape)[0]
    true_c2_index = np.unravel_index(np.argmax(goal_grid[:,:,1]),goal_grid[:,:,1].shape)[1]
    
    true_c1_dest = np.copy(c1_init)
    true_c1_dest[0] = 6 #NOTE: RECALL THIS
    true_c1_dest[2] += 0
    
    true_c1_has_lead = c1_lead[true_c1_index]
    true_c2_has_lead = c2_lead[true_c2_index]
    
    true_c2_dest = np.copy(c2_init)
    true_c2_dest[0] = 6 #NOTE: RECALL THIS
    true_c2_dest[2] += 0
    
    c2_c2_joint_opt_x,c2_c2_joint_opt_u,c2_c1_joint_opt_x,c2_c1_joint_opt_u =\
              optimiser(c2_init,true_c2_dest,true_c2_has_lead,c1_init,true_c1_dest,true_c1_has_lead)
    c1_c1_joint_opt_x,c1_c1_joint_opt_u,c1_c2_joint_opt_x,c1_c2_joint_opt_u =\
              optimiser(c1_init,true_c1_dest,true_c1_has_lead,c2_init,true_c2_dest,true_c2_has_lead)

    #c2_c2_joint_opt_x,c2_c2_joint_opt_u,c2_c1_joint_opt_x,c2_c1_joint_opt_u =\
    #          optimiser(c2_init,true_c2_dest,true_c2_has_lead,c1_init,true_c1_dest,true_c1_has_lead)
   
    pdb.set_trace()

    # Plot Resulting Trajectories
    #dynamicPlotter(c1_joint_opt_x,c2_joint_opt_x)
    #pdb.set_trace()

    ########################################################################

    ##########################################################################
    #MPC Loop
    c1_c1_has_lead = c1_lead[c1_index] #if c1 thinks they are going to cut ahead
    c1_c2_has_lead = c1_lead[c1_c2_index] #if c1 thinks c2 will give way
    c2_c1_has_lead = c2_lead[c2_c1_index] #if c2 thinks c1 will give way
    c2_c2_has_lead = c2_lead[c2_index] #if c2 thinks they are expected to continue
    t = 0
    c1_t,c2_t = None,None #time at which each car completed their true objective
    num_timesteps = 2 # How many timesteps are followed per iteration

    while t<T and (c1_t is None or c2_t is None):
        ###########################################
        #### MPC for C1 ###########################
        c1_opt_x,c1_opt_u,c1_c2_opt_x,c1_c2_opt_u = optimiser(c1_x,c1_dest,c1_c1_has_lead,c2_x,c1_c2_dest,c1_c2_has_lead)

        ############################################
        #### MPC for C2 ############################
        c2_opt_x,c2_opt_u,c2_c1_opt_x,c2_c1_opt_u = optimiser(c2_x,c2_dest,c2_c2_has_lead,c1_x,c2_c1_dest,c2_c1_has_lead)
      
        #pdb.set_trace()
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
            #c1_x += dt*(np.array((c1_x[2]*math.cos(c1_x[3]+u1[1]),c1_x[2]*math.sin(c1_x[3]+u1[1]),u1[0],(2*c1_x[2]/veh_length)*math.sin(u1[1]))).reshape(4,1))
            #c2_x += dt*(np.array((c2_x[2]*math.cos(c2_x[3]+u2[1]),c2_x[2]*math.sin(c2_x[3]+u1[1]),u2[0],(2*c2_x[2]/veh_length)*math.sin(u2[1]))).reshape(4,1))

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

            ##############################################
            #Store MPC generated trajectories
            c1_mpc_u = np.hstack((c1_mpc_u,np.array(u1)))
            c2_mpc_u = np.hstack((c2_mpc_u,np.array(u2)))
            c1_mpc_x = np.hstack((c1_mpc_x,np.array(c1_x)))
            c2_mpc_x = np.hstack((c2_mpc_x,np.array(c2_x)))

        t += num_timesteps*dt


        ################################################
        #If C1 satisfies their current objective
        if c1_t is None and computeDistance(c1_x,c1_dest)<epsilon:
            c1_t = t #Time C1 satisfied trajectory
            print("C1_T set: {}".format(c1_t))
        #C1 has drifted from their objective, reset value
        elif c1_t is not None and computeDistance(c1_x,c1_dest)>epsilon: c1_t = None

        ###############################################
        #If C2 satisfies their current objective
        if c2_t is None and computeDistance(c2_x,c2_dest)<epsilon:
            c2_t = t
            print("C2_T set: {}".format(c2_t))
        #C2 has drifted from their objective, reset value.
        elif c2_t is not None and computeDistance(c2_x,c2_dest)>epsilon: c2_t = None

        print("T is: {}\tD1: {}\t D2: {}".format(t,computeDistance(c1_x,c1_dest),computeDistance(c2_x,c2_dest)))


    print("MPC Complete")
    #t2 = datetime.datetime.now()
    #print("Time: {}".format(t2-t1))
    #pdb.set_trace()

    dynamicPlotter(c1_mpc_x,c2_mpc_x)
    pdb.set_trace()

#####################################
