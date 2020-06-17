#Optimal Control problem using multiple-shooting
#Multiple-shooting: whole state, trajectory and control trajectory, are decision variables

from casadi import *
import math
import matplotlib.pyplot as plt # for the 'spy' function and plotting results
import numpy as np # to get teh size of matrices
import random # to add noise in mpc

import pdb

##########################################################
########## Initialise Variables ##########################

#2-D state 
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
T = 10 # time horizon
N = int(T*(1/dt)) # number of control intervals

#Options for integrator to discretise the system
# Options are optional
intg_options = {}
intg_options['tf'] = T/N # timesteps
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

lane_width = 4
speed_limit = 22.22

init_x = lane_width/2
init_y = 5
init_vel = 15
init_heading = math.pi/2
init_state = [init_x,init_y,init_vel,init_heading]

dest_x = 3*lane_width/2
dest_y = init_y + 100
dest_vel = 0
dest_heading = math.pi/2
dest_state = [dest_x,dest_y,dest_vel,dest_heading]

bounds = [0,2*lane_width,0,speed_limit,0,math.pi,-3,3,-math.pi/18,math.pi/18]

opti = casadi.Opti()

x = opti.variable(4,N+1) # Decision variables for state trajectory
u = opti.variable(2,N)
p = opti.parameter(4,1) # Parameter (not optimized over) Initial value for x
goal = opti.parameter(4,1)
#x_low,x_high, speed_low,speed_high,heading_low,heading_high,accel_low,accel_high,yaw_low,yaw_high
bnd = opti.parameter(10,1)

weight = opti.parameter(4,1)
opti.set_value(weight,[1,0,1,1])

# minimize sum_{k=1}^{N+1} x_{k}^{T}x_{k} + sum_{k=1}^{N+1} u_{k}^{T}u_{k}
opti.minimize(sumsqr((x[:,-1]-goal)*weight) + sumsqr(u[1,:])) # Distance to destination
#opti.minimize(sumsqr((x[:,-1]-goal)*weight)) # Distance to destination
#opti.minimize(sumsqr(x-goal) + sumsqr(u)) # Distance to destination
#opti.minimize(sumsqr(x)+sumsqr(u))

#This can also be done with functional programming (mapaccum)
for k in range(N):
    opti.subject_to(x[:,k+1]==F(x[:,k],u[:,k]))

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
opti.subject_to(x[:,0]==p) #Initial state

opti

###########################################################
########### Define Optimizer ##############################

#Choose a solver
test1 = {}
test1['qpsol'] = 'qrqp'
opti.solver('sqpmethod',test1)

#Choose a concrete value for p
opti.set_value(p,init_state) # set initial conditions (initial value for x)
opti.set_value(goal,dest_state)
opti.set_value(bnd,bounds)
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
