from casadi import *
import math
import matplotlib.pyplot as plt # for the 'spy' function and plotting results
import numpy as np # to get teh size of matrices
import random # to add noise in mpc
import time

import pdb


np.set_printoptions(suppress=True) # suppress scientific notation
CONTENT_DIVIDER = "~####~"
RESULT_DIVIDER = "-$$$$-"

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
    L = veh_length
    ode = vertcat(x[2]*cos(x[3]+u[1]),x[2]*sin(x[3]+u[1]),u[0],(2*x[2]/L)*sin(u[1]))

    f = Function('f',[x,u],[ode],['x','u'],['ode'])
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
    x_next = res['xf']

    # Maps straight from inital state x to final state xf, given control input u
    F = Function('F',[x,u],[x_next],['x','u'],['x_next'])

    return F


def makeIDMModel(has_new_leader,accel_bounds,accel_range,v_goal,d_goal,T_safe,veh_length):
    x_ego = MX.sym('x',4) # state <- x,y,v,heading
    x_lead = MX.sym('x',4) # state <- x,y,v,heading
    v_dot = MX.sym('vdot',1) # control input <- a,yaw_rate

    del_v = x_ego[2]-x_lead[2]

    d_star = d_goal + T_safe*x_ego[2] + x_ego[2]*del_v/(2*sqrt(accel_bounds[1]*(-accel_bounds[0])))

    d = x_lead[1]-x_ego[1] - veh_length +.00001 #rear to front distance = midpoint_distance-rear_half_of_lead-front_half_of_follower

    v_dot = accel_bounds[1]*(1-(x_ego[2]/v_goal)**4-has_new_leader*(d_star/d)**2)

    v_dot = if_else(v_dot<accel_bounds[0],accel_bounds[0],v_dot)
    v_dot = if_else(v_dot>accel_bounds[1],accel_bounds[1],v_dot)

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

    safe_x_radius = veh_width + .25
    safe_y_radius = veh_length + .5

    opti = casadi.Opti()

    #IDM Model
    has_lead1 = opti.parameter(1,1)
    has_lead2 = opti.parameter(1,1)
    #comfort_accel_range = [-2.5,2] # NOTE: Manually specifying values
    #FIDM1 = makeIDMModel(has_lead1,comfort_accel_range,accel_range,15,1,.8,veh_length)
    #FIDM2 = makeIDMModel(has_lead2,comfort_accel_range,accel_range,15,1,.8,veh_length)

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

    #Encourage other vehicle action solution to follow specified IDM model
    #c1_to_idm_weight = 100 #10
    #c1_to_idm = c1_to_idm_weight*sum([sumsqr(u1[0,k]-FIDM1(x1[:,k],x2[:,k])) for k in range(N)])
    c1_to_idm = 0 

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

    #Encourage other vehicle action solution to follow specified IDM model
    #c2_to_idm_weight = 100 #10
    #c2_to_idm = c2_to_idm_weight*sum([sumsqr(u2[0,k]-FIDM2(x2[:,k],x1[:,k])) for k in range(N)])
    c2_to_idm = 0

    #If the car has a leader, motivate it to get behind the other car
    c2_behind_c1_weight = 10*has_lead2
    c2_behind_c1 = sum2(fmax(x2[1,:]-x1[1,:],0))*c2_behind_c1_weight

    #Encourage cars to stay maximise distance between each other
    safety_weight = 0
    safety = safety_weight*sumsqr(1-(((x1[0,:]-x2[0,:])/safety_params[0])**2 + \
                          ((x1[1,:]-x2[1,:])/safety_params[1])**2))

    opti.minimize(c1_min_traj_duration+c1_min_final_dist+c1_min_accel+c1_min_jerk+c1_to_idm+c1_behind_c2+\
                   c2_min_traj_duration+c2_min_final_dist+c2_min_accel+c2_min_jerk+c2_behind_c1+\
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
    ipopt_opts["ipopt.max_iter"] = 1500

    opti.solver('ipopt',ipopt_opts)

    #Turn optimisation to CasADi function
    M = opti.to_function('M',[init_state1,dest_state1,has_lead1,init_state2,dest_state2,has_lead2],\
                            [x1[:,:],u1[:,:],x2[:,:],u2[:,:]],['init1','dest1','has_lead2','init2','dest2','has_lead1'],\
                            ['x1_opt','u1_opt','x2_opt','u2_opt'])

    return M



###################################################################################################
####### Reward Grid Stuff #########################################################################
def makeVanillaAltRewardGrid(reward_grid,alt1,alt2):
    alt_reward = np.copy(reward_grid)
    alt_reward[:,:,0] = (1-alt1)*reward_grid[:,:,0] + alt1*reward_grid[:,:,1]
    alt_reward[:,:,1] = (1-alt2)*reward_grid[:,:,1] + alt2*reward_grid[:,:,0]

    return alt_reward

###################################################################################################
########## Other ##################################################################################
def computeDistance(x1,x2):
    #distance from desired x-position and heading
    return math.sqrt((x1[0]-x2[0])**2 + (x1[3]-x2[3])**2)

###################################################################################################
######### MPC Subroutine ##########################################################################
def doMPC(num_timesteps,T,c1_init,c1_dests,c1_leads,c2_init,c2_dests,c2_leads,shift_tol,c1_has_shift=False,c2_has_shift=False):
    c1_dest,c1_c2_dest = c1_dests
    c2_dest,c2_c1_dest = c2_dests
    c1_c1_has_lead,c1_c2_has_lead = c1_leads
    c2_c2_has_lead,c2_c1_has_lead = c2_leads
    c1_x,c2_x = np.copy(c1_init),np.copy(c2_init)
    #Recording trajectory generated by MPC loop
    c1_mpc_x,c2_mpc_x = np.array(c1_x),np.array(c2_x)
    c1_mpc_u,c2_mpc_u = np.array([0,0]).reshape(2,1),np.array([0,0]).reshape(2,1)

    t = 0
    c1_t,c2_t = None,None #time at which each car completed their true objective
    while t<T and (c1_t is None or c2_t is None):
        #print("t is: {}\t T is: {}".format(t,T))
        ###########################################
        #### MPC for C1 ###########################
        c1_opt_x,c1_opt_u,c1_c2_opt_x,c1_c2_opt_u = optimiser(c1_x,c1_dest,c1_c1_has_lead,c2_x,c1_c2_dest,c1_c2_has_lead)

        ############################################
        #### MPC for C2 ############################
        c2_opt_x,c2_opt_u,c2_c1_opt_x,c2_c1_opt_u = optimiser(c2_x,c2_dest,c2_c2_has_lead,c1_x,c2_c1_dest,c2_c1_has_lead)

        #print("Optimal Trajectories Generated")

        ############################################
        #### Handle Optimiser Singularity
        if True in [c1_opt_x[1,i]>c1_opt_x[1,i+1]+.02 for i in range(c1_opt_x.shape[1]-1)]:
            #Problem in C1 MPC - shift C1 initial position
            if c1_has_shift:
                #print("Problem for C1 but has already been shifted")
                return 0,t,c1_has_shift,c2_has_shift,None,None,None,None
            else:
                for shift_y in [shift_tol,-shift_tol]:
                    #print("Shifting C1 by {}".format(shift_y))
                    shift_c1_init = np.copy(c1_init)
                    shift_c1_init[1] += shift_y
                    result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u = doMPC(num_timesteps,T,shift_c1_init,c1_dests,c1_leads,c2_init,c2_dests,c2_leads,shift_tol,True,c2_has_shift)
            
                    if c1_mpc_x is not None: #shift resolved issue
                        #print("Returning shifted results for C1")
                        return result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u

            #print("Failed to solve by shifting C1")
            return 0,t,c1_has_shift,c2_has_shift,None,None,None,None # if we get here the issue was not resolved

        if True in [c2_opt_x[1,i]>c2_opt_x[1,i+1]+.02 for i in range(c2_opt_x.shape[1]-1)]:
            #Problem in C1 MPC - shift C1 initial position
            if c2_has_shift:
                #print("Problem for C2 but has already been shifted")
                return 0,t,c1_has_shift,c2_has_shift,None,None,None,None
            else:
                for shift_y in [shift_tol,-shift_tol]:
                    #print("Shifting C2 by {}".format(shift_y))
                    shift_c2_init = np.copy(c2_init)
                    shift_c2_init[1] += shift_y
                    result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u = doMPC(num_timesteps,T,c1_init,c1_dests,c1_leads,shift_c2_init,c2_dests,c2_leads,shift_tol,c1_has_shift,True)
            
                    if c1_mpc_x is not None: #shift resolved issue
                        #print("Returning shifted results for C2")
                        return result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u

            #print("Failed to solve by shifting C2")
            return 0,t,c1_has_shift,c2_has_shift,None,None,None,None # if we get here the issue was not resolved

        ###############################################
        #### Take steps along chosen path
        #print("Running MPC")

        for j in range(num_timesteps):
            u1 = np.array(c1_opt_u[:,j])
            u2 = np.array(c2_opt_u[:,j])

            c1_x = np.array(c1_opt_x[:,j+1])
            c2_x = np.array(c2_opt_x[:,j+1])


            ##############################################
            #Store MPC generated trajectories
            c1_mpc_u = np.hstack((c1_mpc_u,np.array(u1)))
            c2_mpc_u = np.hstack((c2_mpc_u,np.array(u2)))
            c1_mpc_x = np.hstack((c1_mpc_x,np.array(c1_x)))
            c2_mpc_x = np.hstack((c2_mpc_x,np.array(c2_x)))

            ##############################################
            #If MPC does not have safety as constraint then test for crash out here
            crash_check = (((c1_x[0,:]-c2_x[0,:])/veh_width)**2 + ((c1_x[1,:]-c2_x[1,:])/veh_length)**2)
            if crash_check<1:
                return -1,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u
            ##############################################
        t += num_timesteps*dt


        ################################################
        #If C1 satisfies their current objective
        if c1_t is None and computeDistance(c1_x,c1_dest)<epsilon:
            c1_t = t #Time C1 satisfied trajectory
        #C1 has drifted from their objective, reset value
        elif c1_t is not None and computeDistance(c1_x,c1_dest)>epsilon: c1_t = None

        ###############################################
        #If C2 satisfies their current objective
        if c2_t is None and computeDistance(c2_x,c2_dest)<epsilon:
            c2_t = t
        #C2 has drifted from their objective, reset value.
        elif c2_t is not None and computeDistance(c2_x,c2_dest)>epsilon: c2_t = None

    if c1_t is not None and c2_t is not None:
        #Didn't just time out
        result = 1
    else:
        #Timed out. Not a success
        result = 0

    #print("Returning successful results")
    return result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u 


if __name__ == "__main__":
    ###################################
    #Vehicle dimensions
    veh_length = 4.6
    veh_width = 2

    ###################################
    #Optimiser Parameters
    axle_length = 2.7 # length of car axle
    dt = .2 # timestep size
    epsilon = .05 # maximum distance from objective for experiment to finish
    lane_width = 4 # width of a lane
    T = 10 #Trajectory length
    lookahead_horizon = 4 # length of time MPC plans over
    N = int(lookahead_horizon/dt)

    speed_limit = 15
    accel_range = [-9,3]
    yaw_rate_range = [-math.pi/180,math.pi/180]    

    ###################################
    #MPC Parameters
    num_timesteps = 2 #num timesteps of optimal trajectory followed per iteration

    ###################################
    #Experiment Parameters
    rewardDefinition = makeVanillaAltRewardGrid

    alt_values = [.1,.9] #Altruism
    N = 6
    shift_values = [x*.25*veh_length for x in range(N+1)]     
 
    ###################################
    #Define Trajectory Options
    c1_lead = [1,0]
    c2_lead = [1,0]

    shift_tol = .01*veh_length

    ###################################
    #Initialise Experiment File
    import datetime
    exp_name = "IDM_MPC_Vary_init"
    start_time = datetime.datetime.now()
    exp_file = open("{}-{}.txt".format(exp_name,start_time),"w")
    exp_file.write("{}\n\n".format(CONTENT_DIVIDER))
    exp_file.write("axle_length: {}\ndt: {}\nepsilon: {}\tlane_width: {}\nT: {}\nlookahead_horizon: {}\nN: {}\nspeed_limit: {}\taccel_range: {}\tyaw_rate_range: {}\n".format(axle_length,dt,epsilon,lane_width,T,lookahead_horizon,N,speed_limit,accel_range,yaw_rate_range))
    exp_file.write("\n")

    optimiser = makeJointIDMOptimiser(dt,lookahead_horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)

    #Use float values or else numpy will round to int
    #reward_grid = np.array([[[-np.inf,-np.inf],[0,1]],[[1,0],[-np.inf,-np.inf]]])
    reward_grid = np.array([[[-1.0,-1.0],[0.0,1.0]],[[1.0,0.0],[-1.0,-1.0]]])

    exp_file.write("Alt Values: {}\n\n".format(alt_values))
    exp_file.write("Shift Values: {}\n\n".format(shift_values))
    exp_file.close()

    for a1 in alt_values:
        for a2 in alt_values:
            exp_file = open("{}-{}.txt".format(exp_name,start_time),"a")
            exp_file.write("\n{}\n\n".format(CONTENT_DIVIDER))
            exp_file.write("a1: {}\t a2: {}\n".format(a1,a2))
            
            goal_grid = rewardDefinition(reward_grid,a1,a2)

            exp_file.write("goal_grid: \n{}\n".format(goal_grid))

            
            c1_index = np.unravel_index(np.argmax(goal_grid[:,:,0]),goal_grid[:,:,0].shape)[0]
            c1_c2_index = np.unravel_index(np.argmax(goal_grid[c1_index,:,1]),\
                                  goal_grid[c1_index,:,1].shape)[0] #c2's optimal choice if c1 is lead
            c2_index = np.unravel_index(np.argmax(goal_grid[:,:,1]),goal_grid[:,:,1].shape)[1]
            c2_c1_index = np.unravel_index(np.argmax(goal_grid[:,c2_index,0]),\
                                  goal_grid[:,c2_index,0].shape)[0] # c1 optimal choice of c2 lead

            exp_file.write("True Joint Reward: {}\n\n".format(reward_grid[c1_index,c2_index,:]))
            exp_file.close()
            
            for dy_c1 in shift_values:
                for dy_c2 in shift_values:
                    #print("Working on A1: {} S1: {} A2: {} S2: {}".format(a1,dy_c1,a2,dy_c2))
                    init_c1_posit = [0.5*lane_width,dy_c1] # middle of right lane
                    init_c2_posit = [1.5*lane_width,dy_c2] # middle of right lane
                    init_c1_vel = 15
                    init_c2_vel = 15
                    init_c1_heading = math.pi/2
                    init_c2_heading = math.pi/2
                    init_c1_accel = 0
                    init_c2_accel = 0
                    init_c1_yaw_rate = 0
                    init_c2_yaw_rate = 0


                    #Adjust Destination for noise so that intended target is still in middle of lane
                    c1_init = np.array([*init_c1_posit,init_c1_vel,init_c1_heading]).reshape(4,1)
                    c2_init = np.array([*init_c2_posit,init_c2_vel,init_c2_heading]).reshape(4,1)
                
                    c1_dest = np.copy(c1_init)
                    c1_dest[0] += lane_width
                    c1_dest[2] += 0
                    
                    c2_dest = np.copy(c2_init)
                    c2_dest[0] += 0
                    c2_dest[2] += 0

                    c1_c2_dest = np.copy(c1_init)
                    c1_c2_dest[0] += lane_width
                    c1_c2_dest[2] += 0
                    
                    c2_c1_dest = np.copy(c2_init)
                    c2_c1_dest[0] += 0
                    c2_c1_dest[2] += 0

                    c1_dests = [c1_dest,c1_c2_dest]
                    c2_dests = [c2_dest,c2_c1_dest]

                    c1_c1_has_lead = c1_lead[c1_index] #if c1 thinks they are going to cut ahead
                    c1_c2_has_lead = c1_lead[c1_c2_index] #if c1 thinks c2 will give way
                    c2_c1_has_lead = c2_lead[c2_c1_index] #if c2 thinks c1 will give way
                    c2_c2_has_lead = c2_lead[c2_index] #if c2 thinks they are expected to continue

                    c1_leads = [c1_c1_has_lead,c1_c2_has_lead]
                    c2_leads = [c2_c2_has_lead,c2_c1_has_lead]

                    #####################################################################
                    # Run MPC
                    result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u =\
                    doMPC(num_timesteps,T,c1_init,c1_dests,c1_leads,c2_init,c2_dests,c2_leads,shift_tol)
                    #Legend:
                    #  - result = 1: converged to solution
                    #  - result = 0: failed to converge to solution solution
                    #  - result = -1: trajectory crashed

                    outcome = 0
                    if result is 1: #Converged to satisfactory solution
                        if c1_mpc_x[1,-1]>c2_mpc_x[1,-1]: outcome = 1
                        else: outcome = -1

                    #Legend:
                    #  - outcome = 1: c1 ended up ahead of c2
                    #  - outcome = 0: no solution was generated
                    #  - outcome = -1: c1 ended up behind c2

                    #####################################################################
                    #Record Results                
                    exp_file = open("{}-{}.txt".format(exp_name,start_time),"a")
                    exp_file.write("{}\n".format(RESULT_DIVIDER))
                    exp_file.write("Shift: C1: {}\tC2: {}\n".format(dy_c1,dy_c2))
                    #Result: 1 <- converged to solution, -1 <- crash 0 <- no solution
                    exp_file.write("Result: {}\tOutcome: {}\tT: {}\n".format(result,outcome,t))
                    # Did either car need to be shifted to reach solution
                    exp_file.write("Shift: {}\t{}\n".format(c1_has_shift,c2_has_shift))
                    exp_file.write("C1\nX:{}\tU:{}\n".format(c1_mpc_x,c1_mpc_u))
                    exp_file.write("C2\nX:{}\tU:{}\n".format(c2_mpc_x,c2_mpc_u))
                    exp_file.write("\n")
                    exp_file.close()

#####################################
