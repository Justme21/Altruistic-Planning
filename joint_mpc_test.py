#Optimal Control problem using multiple-shooting
#Multiple-shooting: whole state, trajectory and control trajectory, are decision variables

from casadi import *
import math
import matplotlib.pyplot as plt # for the 'spy' function and plotting results
import numpy as np # to get teh size of matrices
import random # to add noise in mpc

import pdb

def optimiser():
    ########## Initialise Variables ##########################

    #2 agents (ego,other), each with 4-D state space 
    x = MX.sym('x',4) # state <- x,y,v,heading
    u = MX.sym('u',2) # control input <- a,yaw_rate

    ##########################################################
    ########### Define ODE/Dynamics model  ###################

    #computational graph definition of dynamics model
    #Bicycle model
    L = 4 # Length of vehicle
    ode = vertcat(x[2]*cos(x[3]+u[1]),x[2]*sin(x[3]+u[1]),u[0],(2*x[2]/L)*sin(u[1]))

    #f is a function that takes as input x and u and outputs the
    # state specified by the ode

    f = Function('f',[x,u],[ode],['x','u'],['ode']) # last 2 arguments name the inputs/outputs (Optional)
    #f([0.2,0.8],0.1) # to see sample output

    ##########################################################
    ########### Implementing the Integrator ##################
    dt = .2
    T = 4 # time horizon
    N = int(T*(1/dt)) # number of control intervals

    #Options for integrator to discretise the system
    # Options are optional
    intg_options = {}
    intg_options['tf'] = dt # intergrator runs for 1 timestep
    intg_options['simplify'] = True
    intg_options['number_of_finite_elements'] = 4 #number of intermediate steps to integration (?)

    #DAE problem structure/problem definition
    dae = {}
    dae['x'] = x  #What are states? 
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

    #########################################################
    ################# How to simulate whole trajectory ######

    sim = F.mapaccum(N)
    # F maps from (x[2],u) -> x_next[2]
    # sim maps from (x[2],u[1x20]) -> x_next[2x20]

    #########################################################
    ############ Symbolic Differentiation ###################

    #x0 = [0,1]

    #U = MX.sym('U',1,N)
    #X1 = sim(x0,U)[1,:] # Simulate using concrete x0 and symbolic input series
    #J = jacobian(X1,U) # Jacobian of first states wrt input series


    #print(J.shape) #NxN jacobian
    #Does not work on symbolic function
    #plt.spy(J) # plot the dependeneices on a grid

    #Jf = Function('JF',[U],[J])
    #to get numerical output
    #full(Jf(0)) # compute jacobian on 0 control
    #plt.imshow(Jf(0).full())
    #plt.show()

    ##########################################################
    ########### Initialise Optimisation Problem ##############

    num_cars = 2

    lane_width = 4
    speed_limit = 22.22

    safe_x_radius = 2
    safe_y_radius = 4

    #############################################
    #Define Other Attributes
    SVO_other = math.radians(85)

    other_dx = 0
    other_dy = -50 # generally y is weighted so as not to matter
    other_dv = 0

    other_x = 6
    other_y = 50
    other_vel = 5
    other_heading = math.pi/2

    #other_init = (other_x,other_y,other_vel,other_heading)
    other_dest = (other_x+other_dx,other_y+other_dy,other_vel+other_dv,other_heading)

    #############################################
    #Define Ego Attributes
    SVO_ego = math.radians(45)

    ego_dx = 0
    ego_dy = 50
    ego_dv = 0

    ego_x = 2
    ego_y = 0
    ego_vel = 5
    ego_heading = math.pi/2

    #ego_init = [ego_x,ego_y,ego_vel,ego_heading]
    ego_dest = [ego_x+ego_dx,ego_y+ego_dy,ego_vel+ego_dv,ego_heading]

    ##############################################
    #Define Optimisation features: variables, parameters
    state_bounds = [0,2*lane_width,0,speed_limit,0,math.pi]
    action_bounds = [-3,3,-math.pi/180,math.pi/180]

    opti = casadi.Opti()

    #2 cars, 4-D state, N+1 trajectory length
    x_ego = opti.variable(4,N+1) # Decision variables for state trajectory
    x_other = opti.variable(4,N+1) # Decision variables for state trajectory
    u_ego = opti.variable(2,N)
    u_other = opti.variable(2,N)

    #ego_init_state = np.array(ego_init).reshape(4,1)
    #other_init_state = np.array(other_init).reshape(4,1)

    ego_dest_state = np.array(ego_dest).reshape(4,1)
    other_dest_state = np.array(other_dest).reshape(4,1)

    p_ego = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x
    p_other = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x
    #opti.set_value(p_ego,ego_init_state) # set initial conditions (initial value for x)
    #opti.set_value(p_other,other_init_state) # set initial conditions (initial value for x)
    goal_ego = opti.parameter(4,1)
    goal_other = opti.parameter(4,1)
    opti.set_value(goal_ego,ego_dest_state)
    opti.set_value(goal_other,other_dest_state)

    #x_low,x_high, speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
    state_bnd = opti.parameter(6,1)
    opti.set_value(state_bnd,state_bounds)
    act_bnd = opti.parameter(4,1)
    opti.set_value(act_bnd,action_bounds)

    safety_params = opti.parameter(2,1)
    opti.set_value(safety_params,[safe_x_radius,safe_y_radius])

    ###########################################
    #Define reward function
    def reward(x,u,goal,bnd,wght):
        #0
        progress_x = wght[0]*(x[0,-1]-goal[0])**2 # final distance from goal x-position
        #1
        progress_y = wght[1]*(x[1,-1]-goal[1])**2 # final distance from goal y-position
        #2
        #desired_velocities = wght[2]*(x[2,-1]-goal[2])**2
        desired_velocities = wght[2]*(x[2,-1]-5)**2
        #3
        progress_heading = wght[3]*(x[3,-1]-goal[3])**2 # final distance from goal y-position

        #4
        comfort_accel = wght[4]*sumsqr(u[0,:])/u.shape[1] # minimise average squared yaw
        #5
        comfort_yaw = wght[5]*sumsqr(u[1,:])/u.shape[1]

        #pdb.set_trace()
        return sum1(vertcat(progress_x,progress_y,desired_velocities,progress_heading,comfort_accel,comfort_yaw))

    weight = opti.parameter(6,1)
    #                       0, 1, 2, 3, 4, 5,
    opti.set_value(weight,[30/lane_width, 0, 1/5, 1/math.pi, 1/3,1/math.pi])
    #opti.set_value(weight,[0, 0, 10000, 0, 0,0])
    #opti.set_value(weight,[ 2, 0, 1, 2, 1,20])

    svo_weight_ego = opti.parameter(2,1)
    opti.set_value(svo_weight_ego,(cos(SVO_ego),sin(SVO_ego)))

    svo_weight_other = opti.parameter(2,1)
    opti.set_value(svo_weight_other,(cos(SVO_other),sin(SVO_other)))

    r_ego = reward(x_ego,u_ego,goal_ego,state_bounds,weight)
    r_other = reward(x_other,u_other,goal_other,state_bounds,weight)

    G_ego = r_ego*svo_weight_ego[0] + r_other*svo_weight_ego[1]
    G_other = r_other*svo_weight_other[0] + r_ego*svo_weight_other[1]

    G_tot = G_ego + G_other

    G = Function('G',[x_ego,x_other,u_ego,u_other],[G_tot],['x_ego','x_other','u_ego','u_other'],['G-out'])

    ##########################################
    #Define Optimisation Problem
    opti.minimize(G(x_ego,x_other,u_ego,u_other))
    #This can also be done with functional programming (mapaccum)
    for x,u,p in zip([x_ego,x_other],[u_ego,u_other],[p_ego,p_other]):
        for k in range(N):
            opti.subject_to(x[:,k+1]==F(x[:,k],u[:,k]))

        #####Ego Constraints################
        #X-coord constraints <- stay on road constraint
        ego_x_constr_1 = state_bnd[0]-x[0,:]
        ego_x_constr_2 = x[0,:]-state_bnd[1]
        opti.subject_to(ego_x_constr_1<=0)
        opti.subject_to(ego_x_constr_2<=0)
        #Velocity Contraints
        ego_vel_constr_1 = state_bnd[2]-x[2,:]
        ego_vel_constr_2 = x[2,:]-state_bnd[3]
        opti.subject_to(ego_vel_constr_1<=0)
        opti.subject_to(ego_vel_constr_2<=0)
        #Heading Constraints
        #ego_heading_constr_1 = state_bnd[4]-x[3,:]
        #ego_heading_constr_2 = x[3,:]-state_bnd[5]
        #opti.subject_to(ego_heading_constr_1<=0)
        #opti.subject_to(ego_heading_constr_2<=0)
        #Accel Constraints
        ego_accel_constr_1 = act_bnd[0]-u[0,:]
        ego_accel_constr_2 = u[0,:]-act_bnd[1]
        opti.subject_to(ego_accel_constr_1<=0)
        opti.subject_to(ego_accel_constr_2<=0)
        #Yaw Rate Constraints
        ego_yaw_constr_1 = act_bnd[2]-u[1,:]
        ego_yaw_constr_2 = u[1,:]-act_bnd[3]
        opti.subject_to(ego_yaw_constr_1<=0)
        opti.subject_to(ego_yaw_constr_2<=0)
        #Initial position contraints
        opti.subject_to(x[:,0]-p==0) #Initial state

    ##Safety Constraints
    safety_constr = 1-(((x_ego[0,:]-x_other[0,:])**2)/(safety_params[0]**2) + ((x_ego[1,:]-x_other[1,:])**2)/(safety_params[1]**2))
    opti.subject_to(safety_constr<=0)


    opti

    ###########################################################
    ########### Define Optimizer ##############################

    #Choose a solver
    test1 = {}
    test1['qpsol'] = 'qrqp'
    opti.solver('sqpmethod',test1)

    #sol = opti.solve() # result of calling solve is a solution object

    #sol.value(x) # <- print optimal values for x, similarly for u

    #Make the solver silent
    opts = {}
    opts['qpsol'] = 'qrqp'# same as above
    opts['print_header'] = False
    opts['print_iteration'] = False
    opts['print_time'] = False

    qpsol_options = {}
    qpsol_options['print_iter'] = False
    qpsol_options['print_header'] = False
    qpsol_options['print_info'] = False
    opts['qpsol_options'] = qpsol_options

    opti.solver('ipopt')

    #sol = opti.solve() #result of calling solve is a solution object

    #ego_state_traj = sol.value(x_ego)
    #other_state_traj = sol.value(x_other)
    #ego_u = sol.value(u_ego)
    #other_u = sol.value(u_other)
    #
    #ego_x = []
    #ego_y = []
    #other_x = []
    #other_y = []
    #import time
    #plt.ion()
    #plt.figure()
    #plt.xlim(0,4)
    #
    #for i in range(ego_state_traj.shape[1]):
    #    #pdb.set_trace()
    #    ego_x.append(ego_state_traj[0,i])
    #    ego_y.append(ego_state_traj[1,i])
    #    other_x.append(other_state_traj[0,i])
    #    other_y.append(other_state_traj[1,i])
    #    plt.plot(ego_x,ego_y,'g-')
    #    plt.plot(other_x,other_y,'r-')
    #    plt.draw()
    #    plt.pause(1e-17)
    #    time.sleep(dt)
    #pdb.set_trace()

    #Turn optimisation to CasADi function
    #Mapping from initial state (p) to optimal control action (u)

    M = opti.to_function('M',[p_ego,p_other],[u_ego[:,1],u_other[:,1]],['p_ego','p_other'],\
                              ['u_ego_opt','u_other_opt'])

    return M

#M contains SQP method, which maps to a QP solver, all contained in a single, differentiable,
#computational graph

####################################
######## MPC Loop ##################
X_log = []
U_log = []

#ego_dx = 10
#ego_dy = 5
#ego_dv = 5

ego_x = 2
ego_y = 5
ego_vel = 5
ego_heading = math.pi/2

x_ego = np.array([ego_x,ego_y,ego_vel,ego_heading]).reshape(4,1)

#other_dx = 0
#other_dy = -50 # generally y is weighted so as not to matter
#other_dv = 0

other_x = 6
other_y = 5
other_vel = 5
other_heading = math.pi/2

x_other = np.array([other_x,other_y,other_vel,other_heading]).reshape(4,1)

M = optimiser()
N = int(4/.2)

#x = np.array([0,1]).reshape(2,1) # reshape here to make this the same shape as output of F
for i in range(4*N):
    #u_ego,u_other = M(x_ego,x_other).full()
    u_ego,u_other = M(x_ego,x_other)

    pdb.set_trace()

    #U_log.append(u_ego)
    #X_log.append(x_ego)

    # simulate system
    x_ego = F(x_ego,u_ego).full() + np.array([random.random()*.02,random.random()*.02,0,0]).reshape(4,1) # adding some noise
    x_other = F(x_other,u_other).full() + np.array([random.random()*.02,random.random()*.02,0,0]).reshape(4,1) # adding some noise
    #x = F(x,u).full()

pdb.set_trace()

#####################################
