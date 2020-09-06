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


def makeJointIntersectionOptimiser(dt,horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range):
    #########################################################
    ##### Make Integrator ###################################
    F = makeIntegrator(dt,veh_length)
    ##########################################################
    ########### Initialise Optimisation Problem ##############

    N = int(horizon/dt)
    #x_low,x_high,speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
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

    opti.minimize(c1_min_traj_duration+c1_behind_mid+\
                   c2_min_traj_duration+c2_behind_mid+\
                   safety)

    for k in range(N):
        opti.subject_to(x1[:,k+1]==F(x1[:,k],u1[:,k]))
        opti.subject_to(x2[:,k+1]==F(x2[:,k],u2[:,k]))

    safety_constr = (((x1[0,:]-x2[0,:])/safety_params[1])**2 + ((x1[1,:]-x2[1,:])/safety_params[0])**2)
    opti.subject_to(safety_constr>=1)


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

###################################################################################################
####### Reward Grid Stuff #########################################################################
def makeVanillaAltRewardGrid(reward_grid,alt1,alt2):
    alt_reward = np.copy(reward_grid)
    alt_reward[:,:,0] = (1-alt1)*reward_grid[:,:,0] + alt1*reward_grid[:,:,1]
    alt_reward[:,:,1] = (1-alt2)*reward_grid[:,:,1] + alt2*reward_grid[:,:,0]

    return alt_reward

###################################################################################################
########## Other ##################################################################################
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
    heading_satisfied = heading_val<math.pi/35 # 5 degree tolerance
    return obj_satisfied and lane_satisfied and heading_satisfied
###################################################################################################
######### MPC Subroutine ##########################################################################
def doMPC(num_timesteps,T,c1_init,c1_init_lane,c1_dest_lane,c1_leads,c2_init,c2_init_lane,c2_dest_lane,c2_leads,shift_tol,c1_has_shift=False,c2_has_shift=False):
    c1_c1_is_lead,c1_c2_is_lead = c1_leads
    c2_c2_is_lead,c2_c1_is_lead = c2_leads
    c1_x,c2_x = np.copy(c1_init),np.copy(c2_init)
    #Recording trajectory generated by MPC loop
    c1_mpc_x,c2_mpc_x = np.array(c1_x),np.array(c2_x)
    c1_mpc_u,c2_mpc_u = np.array([0,0]).reshape(2,1),np.array([0,0]).reshape(2,1)

    t = 0
    c1_t,c2_t = None,None #time at which each car completed their true objective

    #print("T: {}\t D1: {}\t D2: {}".format(t,computeDistance(c1_x,c1_dest_lane),computeDistance(c2_x,c2_dest_lane)))
    #pdb.set_trace()
    while t<T and (c1_t is None or c2_t is None):
        #print("t is: {}\t T is: {}".format(t,T))
        ###########################################
        #### MPC for C1 ###########################
        c1_opt_x,c1_opt_u,c1_c2_opt_x,c1_c2_opt_u = optimiser(c1_x,c1_init_lane,c1_dest_lane,c1_c1_is_lead,c2_x,c2_init_lane,c2_dest_lane,c1_c2_is_lead)

        ############################################
        #### MPC for C2 ############################
        c2_opt_x,c2_opt_u,c2_c1_opt_x,c2_c1_opt_u = optimiser(c2_x,c2_init_lane,c2_dest_lane,c2_c2_is_lead,c1_x,c1_init_lane,c1_dest_lane,c2_c1_is_lead)

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
        #c1_check_val = math.cos(c1_dest_lane[2])*(c1_dest_lane[0]-c1_x[0]) + \
        #                   math.sin(c1_dest_lane[2])*(c1_dest_lane[1]-c1_x[1])
        c1_fin = finCheck(c1_x,c1_dest_lane,lane_width)
        #if c1_t is None and c1_check_val<0 and computeDistance(c1_x,c1_dest_lane)<5:
        if c1_t is None and c1_fin:
            c1_t = t #Time C1 satisfied trajectory
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
        #C2 has drifted from their objective, reset value.
        #elif c2_t is not None and c2_check_val>0: c2_t = None
        elif c2_t is not None and not c2_fin: c2_t = None

        if computeDistance(c1_x,c1_dest_lane)<computeDistance(c1_x,c1_init_lane):
            c1_init_lane = c1_dest_lane # c1 has left starting lane, shouldn't be pulled back
        if computeDistance(c2_x,c2_dest_lane)<computeDistance(c2_x,c2_init_lane):
            c2_init_lane = c2_dest_lane # c2 has left starting lane, shouldn't be pulled back

        #print("T: {}\t D1: {}\t D2: {}".format(t,computeDistance(c1_x,c1_dest_lane),computeDistance(c2_x,c2_dest_lane)))


    if c1_t is not None and c2_t is not None:
        #Didn't just time out
        result = 1
    else:
        #Timed out. Not a success
        result = 0

    #print("Returning successful results")
    return result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u 


def dynamicIntersectionPlotter(mpc_x1,mpc_x2,mdpt,lane_width):
    c1_plt_x = []
    c1_plt_y = []
    c2_plt_x = []
    c2_plt_y = []

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

    delta = 2

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

    shift_tol = .01*veh_length

    ###################################
    #Initialise Experiment File
    import datetime
    exp_name = "Intersection_MPC_Vary_init"
    start_time = datetime.datetime.now()
    exp_file = open("{}-{}.txt".format(exp_name,start_time),"w")
    exp_file.write("{}\n\n".format(CONTENT_DIVIDER))
    exp_file.write("axle_length: {}\ndt: {}\nepsilon: {}\tlane_width: {}\nT: {}\nlookahead_horizon: {}\nN: {}\nspeed_limit: {}\taccel_range: {}\tyaw_rate_range: {}\n".format(axle_length,dt,epsilon,lane_width,T,lookahead_horizon,N,speed_limit,accel_range,yaw_rate_range))
    exp_file.write("\n")

    optimiser = makeJointIntersectionOptimiser(dt,lookahead_horizon,veh_width,veh_length,lane_width,speed_limit,accel_range,yaw_rate_range)

    #Use float values or else numpy will round to int
    #reward_grid = np.array([[[-np.inf,-np.inf],[0,1]],[[1,0],[-np.inf,-np.inf]]])
    reward_grid = np.array([[[-1.0,-1.0],[1.0,0.0]],[[0.0,1.0],[-1.0,-1.0]]])

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
                    init_c1_posit = [veh_length+delta+lane_width/2,.5*veh_length-dy_c1] # middle of right lane
                    #No shift
                    init_c2_posit = [.5*veh_length-dy_c2,veh_length+delta+1.5*lane_width] # middle of right lane
                    #init_c2_posit = [.5*veh_length-dy_c2-lane_width,veh_length+delta+1.5*lane_width] # middle of right lane
                    init_c1_vel = 0
                    init_c2_vel = 0
                    init_c1_heading = math.pi/2
                    init_c2_heading = 0
                    init_c1_accel = 0
                    init_c2_accel = 0
                    init_c1_yaw_rate = 0
                    init_c2_yaw_rate = 0


                    #Adjust Destination for noise so that intended target is still in middle of lane
                    c1_init = np.array([*init_c1_posit,init_c1_vel,init_c1_heading]).reshape(4,1)
                    c2_init = np.array([*init_c2_posit,init_c2_vel,init_c2_heading]).reshape(4,1)
                
                    c1_c1_is_lead = c1_lead[c1_index] #if c1 thinks they are going to cut ahead
                    c1_c2_is_lead = c1_lead[c1_c2_index] #if c1 thinks c2 will give way
                    c2_c1_is_lead = c2_lead[c2_c1_index] #if c2 thinks c1 will give way
                    c2_c2_is_lead = c2_lead[c2_index] #if c2 thinks they are expected to continue

                    c1_leads = [c1_c1_is_lead,c1_c2_is_lead]
                    c2_leads = [c2_c2_is_lead,c2_c1_is_lead]

                    #####################################################################
                    # Run MPC
                    result,t,c1_has_shift,c2_has_shift,c1_mpc_x,c1_mpc_u,c2_mpc_x,c2_mpc_u =\
                    doMPC(num_timesteps,T,c1_init,c1_init_lane,c1_dest_lane,c1_leads,c2_init,c2_init_lane,c2_dest_lane,c2_leads,shift_tol)
                    #Legend:
                    #  - result = 1: converged to solution
                    #  - result = 0: failed to converge to solution/timed out
                    #  - result = -1: trajectory crashed

                    outcome = 0
                    if result is 1: #Converged to satisfactory solution
                        if c1_mpc_x[1,-1]>c2_mpc_x[1,-1] and c2_mpc_x[0,-1]>c1_mpc_x[0,-1]:
                            outcome = 1
                        else: outcome = -1

                    #Legend:
                    #  - outcome = 1: cars successfully crossed the intersection
                    #  - outcome = 0: no solution was generated
                    #  - outcome = -1: cars did not cross the intersection
                    

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
