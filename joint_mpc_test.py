#Optimal Control problem using multiple-shooting
#Multiple-shooting: whole state, trajectory and control trajectory, are decision variables

from casadi import *
import math
import matplotlib.pyplot as plt # for the 'spy' function and plotting results
import numpy as np # to get teh size of matrices
import random # to add noise in mpc

import pdb

########## Initialise Variables ##########################

#2 agents (ego,other), each with 4-D state space 
x = MX.sym('x',8) # state <- x,y,v,heading
u0 = MX.sym('u0',2) # control input <- a,yaw_rate
u1 = MX.sym('u1',2) # control input <- a,yaw_rate

##########################################################
########### Define ODE/Dynamics model  ###################

#computational graph definition of dynamics model
#Bicycle model
L = 4 # Length of vehicle
ode0 = vertcat(x[2]*cos(x[3]+u0[1]),x[2]*sin(x[3]+u0[1]),u0[0],(2*x[2]/L)*sin(u0[1]))
ode1 = vertcat(x[4+2]*cos(x[4+3]+u1[1]),x[4+2]*sin(x[4+3]+u1[1]),u1[0],(2*x[4+2]/L)*sin(u1[1]))
#ode1 = vertcat(x[4+2]*cos(x[4+3]+u1[1]),x[4+2]*sin(x[4+3]+u1[1]),u1[0],(2*x[4+2]/L)*sin(u1[1]))
ode = vertcat(ode0,ode1)

#f is a function that takes as input x and u and outputs the
# state specified by the ode

#f = Function('f',[x,u,u1],[ode],['x','u','u1'],['ode']) # last 2 arguments name the inputs/outputs (Optional)
f = Function('f',[x,u0,u1],[ode],['x','u0','u1'],['ode']) # last 2 arguments name the inputs/outputs (Optional)
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
dae['p'] = vertcat(u0,u1)  # What are parameters (fixed during integration horizon)
#dae['p'] = vertcat(u,u1)  # What are parameters (fixed during integration horizon)
#dae['ode'] = f(x,u,u1) # Expression for x_dot = f(x,u)
dae['ode'] = f(x,u0,u1) # Expression for x_dot = f(x,u)

# Integrating using Runga-Kutte integration method
intg = integrator('intg','rk',dae,intg_options) #function object over CasADi symbols

#Sample output from integrator
#res = intg(x0=[0,1],p=0) # include object labels to make it easier to identify inputs
#res['xf'] #print the final value of x at the end of the integration

#Can call integrator function symbolically
res = intg(x0=x,p=vertcat(u0,u1)) # no numbers give, just CasADi symbols
x_next = res['xf']

#This allows us to simplify API
# Maps straight from inital state x to final state xf, given control input u
F = Function('F',[x,u0,u1],[x_next],['x','u0','u1'],['x_next'])

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

other_x = 2
other_y = 50
other_vel = 5
other_heading = -math.pi/2

other_init = (other_x,other_y,other_vel,other_heading)
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

ego_init = [ego_x,ego_y,ego_vel,ego_heading]
ego_dest = [ego_x+ego_dx,ego_y+ego_dy,ego_vel+ego_dv,ego_heading]

##############################################
#Define Optimisation features: variables, parameters
state_bounds = [0,2*lane_width,0,speed_limit,0,math.pi]
action_bounds = [-3,3,-math.pi/180,math.pi/180]

opti = casadi.Opti()

#2 cars, 4-D state, N+1 trajectory length
x = opti.variable((num_cars*4),N+1) # Decision variables for state trajectory
#u = opti.variable((num_cars*2),N)
u_ego = opti.variable(2,N)
u_other = opti.variable(2,N)

init_state = np.array([ego_init,other_init]).reshape(8,1)
dest_state = np.array([ego_dest,other_dest]).reshape(8,1)

p = opti.parameter(num_cars*4,1) # Parameter (not optimized over) Initial value for x
opti.set_value(p,init_state) # set initial conditions (initial value for x)
goal = opti.parameter(num_cars*4,1)
opti.set_value(goal,dest_state)
#x_low,x_high, speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
state_bnd = opti.parameter(6,1)
opti.set_value(state_bnd,state_bounds)
act_bnd = opti.parameter(4,1)
opti.set_value(act_bnd,action_bounds)

safety_params = opti.parameter(2,1)
opti.set_value(safety_params,[safe_x_radius,safe_y_radius])

###########################################
#Define reward function
def reward(i,x,u,goal,bnd,weight):
    adjust = i*(num_cars-1)
    #0
    progress_x = (x[adjust*4+0,-1]-goal[adjust*4+0])**2 # final distance from goal x-position
    #1
    progress_y = (x[adjust*4+1,-1]-goal[adjust*4+1])**2 # final distance from goal y-position
    #2
    desired_velocities = (x[adjust*4+2,-1]-goal[adjust*4+2])**2
    #3
    progress_heading = (x[adjust*4+3,-1]-goal[adjust*4+3])**2 # final distance from goal y-position

    #4
    comfort_accel = sumsqr(u[0,:])/u.shape[1] # minimise average squared yaw
    #5
    comfort_yaw = sumsqr(u[1,:])/u.shape[1]

    ##Safety Constraints
    #6
    #anti_collision = sum2(-gt(((x[0,:]-x[4+0,:])**2)/(safety_params[0]**2) + ((x[1,:]-x[4+1,:])**2)/(safety_params[1]**2),1))
    #X-coord constraints <- stay on road constraint
    #7
    #x_constr_1 = sum2(bnd[0]-x[adjust*4+0,:])
    #8
    #x_constr_2 = sum2(x[adjust*4+0,:]-bnd[1])
    #Velocity Contraints
    #9
    #vel_constr_1 = sum2(bnd[2]-x[adjust*4+2,:])
    #10
    #vel_constr_2 = sum2(x[adjust*4+2,:]-bnd[3])
    #Heading Constraints
    #11
    #heading_constr_1 = sum2(bnd[4]-x[adjust*4+3,:])
    #12
    #heading_constr_2 = sum2(x[adjust*4+3,:]-bnd[5])
   
    #no_tailgating = sum2(exp(-100*((x[0,:]-obs_posit[0])**2 + (x[1,:]-obs_posit[1])**2)))

    #weight = opti.parameter(6,1)
    #opti.set_value(weight,[2,0,1,2,1,20])

    #return sum1(vertcat(progress_x,progress_y,desired_velocities,progress_heading,comfort_accel,comfort_yaw,anti_collision,x_constr_1,x_constr_2,vel_constr_1,vel_constr_2,heading_constr_1,heading_constr_2)*weight)
    return sum1(vertcat(progress_x,progress_y,desired_velocities,progress_heading,comfort_accel,comfort_yaw)*weight)

#g_out = sum1(vertcat(progress_x,progress_y,desired_velocities,comfort,no_tailgating)*weight)
#g_out = sum1(vertcat(progress_x,progress_y,desired_velocities,progress_heading,comfort_accel,comfort_yaw)*weight)

weight = opti.parameter(6,1)
#                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12
#opti.set_value(weight,[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
opti.set_value(weight,[1, 1, 0, 0, 0, 20])
#opti.set_value(weight,[ 2, 0, 1, 2, 1,20])

svo_weight_ego = opti.parameter(2,1)
opti.set_value(svo_weight_ego,(cos(SVO_ego),sin(SVO_ego)))

svo_weight_other = opti.parameter(2,1)
opti.set_value(svo_weight_other,(cos(SVO_other),sin(SVO_other)))

r_ego = reward(0,x,u_ego,goal,state_bounds,weight)
r_other = reward(1,x,u_other,goal,state_bounds,weight)

G_ego = r_ego*svo_weight_ego[0] + r_other*svo_weight_ego[1]
G_other = r_other*svo_weight_other[0] + r_ego*svo_weight_other[1]

G_tot = G_ego + G_other

G = Function('G',[x,u_ego,u_other],[G_tot],['x','u_ego','u_other'],['G-out'])

##########################################
#Define Optimisation Problem
opti.minimize(G(x,u_ego,u_other))
#This can also be done with functional programming (mapaccum)
for k in range(N):
    opti.subject_to(x[:,k+1]==F(x[:,k],u_ego[:,k],u_other[:,k]))


##########################################
##Safety Constraints
safety_constr = 1-(((x[0,:]-x[4+0,:])**2)/(safety_params[0]**2) + ((x[1,:]-x[4+1,:])**2)/(safety_params[1]**2))
opti.subject_to(safety_constr<=0)

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
ego_accel_constr_1 = act_bnd[0]-u_ego[0,:]
ego_accel_constr_2 = u_ego[0,:]-act_bnd[1]
opti.subject_to(ego_accel_constr_1<=0)
opti.subject_to(ego_accel_constr_2<=0)
#Yaw Rate Constraints
ego_yaw_constr_1 = act_bnd[2]-u_ego[1,:]
ego_yaw_constr_2 = u_ego[1,:]-act_bnd[3]
opti.subject_to(ego_yaw_constr_1<=0)
opti.subject_to(ego_yaw_constr_2<=0)
#Initial position contraints
opti.subject_to(x[:,0]-p==0) #Initial state

#Ego Laplacian Constraints
#u_ego are the only variables we can set the value to.
# u_ego does not arise in the position constraints, so the k_ego_state variables aren't actually
#      used, so they are not counted in the variable count. Leaving here just for completeness 
#k_ego_state = opti.variable(7*(N+1)) #N+1 states, 7 constraints for each state
#k_ego_action = opti.variable(4*N)
##ego_state_constr_list = vertcat(ego_safety_constr,ego_x_constr_1,ego_x_constr_2,ego_vel_constr_1,\
##                        ego_vel_constr_2,ego_heading_constr_1,ego_heading_constr_2)
#ego_action_constr_list = vertcat(ego_accel_constr_1,ego_accel_constr_2,ego_yaw_constr_1,\
#                         ego_yaw_constr_2)
#
##ego_state_constr_jac = jacobian(ego_state_constr_list,u_ego)
##ego_state_constr_laplacian = mtimes(transpose(k_ego_state),ego_state_constr_jac)
#ego_action_constr_jac = jacobian(ego_action_constr_list,u_ego)
#ego_action_constr_laplacian = mtimes(transpose(k_ego_action),ego_action_constr_jac)
#
#jac_G_ego = jacobian(G_ego,u_ego)
##opti.subject_to(jac_G_ego-(ego_state_constr_laplacian+ego_action_constr_laplacian) == 0)
#opti.subject_to(jac_G_ego-(ego_action_constr_laplacian) == 0)
#
##opti.subject_to(k_ego_state*ego_state_constr_list.reshape((7*(N+1),1)) == 0)
#opti.subject_to(k_ego_action*ego_action_constr_list.reshape((4*N,1)) == 0)
#
##opti.subject_to(k_ego_state>=0)
#opti.subject_to(k_ego_action>=0)

##########################################
##Other Constraints
#X-coord constraints <- stay on road constraint
other_x_constr_1 = state_bnd[0]-x[4+0,:]
other_x_constr_2 = x[4+0,:]-state_bnd[1]
opti.subject_to(other_x_constr_1<=0)
opti.subject_to(other_x_constr_2<=0)
#Velocity Contraints
other_vel_constr_1 = state_bnd[2]-x[4+2,:]
other_vel_constr_2 = x[4+2,:]-state_bnd[3]
opti.subject_to(other_vel_constr_1<=0)
opti.subject_to(other_vel_constr_2<=0)
#Heading Constraints
#other_heading_constr_1 = state_bnd[4]-x[4+3,:]
#other_heading_constr_2 = x[4+3,:]-state_bnd[5]
#opti.subject_to(other_heading_constr_1<=0)
#opti.subject_to(other_heading_constr_2<=0)
#Accel Constraints
other_accel_constr_1 = act_bnd[0]-u_other[0,:]
other_accel_constr_2 = u_other[0,:]-act_bnd[1]
opti.subject_to(other_accel_constr_1<=0)
opti.subject_to(other_accel_constr_2<=0)
#Yaw Rate Constraints
other_yaw_constr_1 = act_bnd[2]-u_other[1,:]
other_yaw_constr_2 = u_other[1,:]-act_bnd[3]
opti.subject_to(other_yaw_constr_1<=0)
opti.subject_to(other_yaw_constr_2<=0)
#Initial position contraints
#opti.subject_to(x[:,0]-p==0) #Initial state
#
##Other Laplacian Constraints
#k_other_state = opti.variable(7*(N+1)) #N+1 states, 7 constraints for each state
#k_other_action = opti.variable(4*N)
##other_state_constr_list = vertcat(other_safety_constr,other_x_constr_1,other_x_constr_2,\
##                        other_vel_constr_1,other_vel_constr_2,other_heading_constr_1,\
##                        other_heading_constr_2)
#other_action_constr_list = vertcat(other_accel_constr_1,other_accel_constr_2,other_yaw_constr_1,\
#                         other_yaw_constr_2)
#
##other_state_constr_jac = jacobian(other_state_constr_list,u_other)
##other_state_constr_laplacian = mtimes(transpose(k_other_state),other_state_constr_jac)
#other_action_constr_jac = jacobian(other_action_constr_list,u_other)
#other_action_constr_laplacian = mtimes(transpose(k_other_action),other_action_constr_jac)
#
#jac_G_other = jacobian(G_other,u_other)
##opti.subject_to(jac_G_other-(other_state_constr_laplacian+other_action_constr_laplacian) == 0)
#opti.subject_to(jac_G_other-(other_action_constr_laplacian) == 0)
#
##opti.subject_to(k_other_state*other_state_constr_list.reshape((7*(N+1),1)) == 0)
#opti.subject_to(k_other_action*other_action_constr_list.reshape((4*N,1)) == 0)
#
##opti.subject_to(k_other_state>=0)
#opti.subject_to(k_other_action>=0)

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

sol = opti.solve() #result of calling solve is a solution object

t1 = sol.value(x)
ego_x = []
ego_y = []
other_x = []
other_y = []
import time
plt.ion()
plt.figure()
plt.xlim(0,4)

#axes = plt.gca()
#ego_line, = axes.plot([],[],'g-')
#other_line, = axes.plot([],[],'r-')

for i in range(t1.shape[1]):
    #pdb.set_trace()
    ego_x.append(t1[0,i])
    ego_y.append(t1[1,i])
    other_x.append(t1[4,i])
    other_y.append(t1[5,i])
    #ego_line.set_xdata(ego_x)
    #ego_line.set_ydata(ego_y)
    #other_line.set_xdata(other_x)
    #other_line.set_ydata(other_y)
    plt.plot(ego_x,ego_y,'g-')
    plt.plot(other_x,other_y,'r-')
    plt.draw()
    plt.pause(1e-17)
    time.sleep(dt)
pdb.set_trace()
#sol.value(x)

#Turn optimisation to CasADi function
#Mapping from initial state (p) to optimal control action (u)

M = opti.to_function('M',[p],[u[:,1]],['p'],['u_opt'])

#M contains SQP method, which maps to a QP solver, all contained in a single, differentiable,
#computational graph

####################################
######## MPC Loop ##################
X_log = []
U_log = []

x = np.array([0,1]).reshape(2,1) # reshape here to make this the same shape as output of F
for i in range(4*N):
    u = M(x).full()

    U_log.append(u)
    X_log.append(x)

    # simulate system
    x = F(x,u).full() + np.array([0,random.random()*.02]).reshape(2,1) # adding some noise
    #x = F(x,u).full()

pdb.set_trace()

#####################################
