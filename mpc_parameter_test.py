import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
import scipy.optimize
import numpy as np
import time

from itertools import tee

import math
import matplotlib.pyplot as plt

def extract(var):
    return th.function([], var, mode=th.compile.Mode(linker='py'))()

def shape(var):
    """Returns function mapping [] to a numpy array containing the shape of var"""
    return extract(var.shape)

def vector(n):
    return th.shared(np.zeros(n))

def matrix(n, m):
    return th.shared(np.zeros((n, m)))

def grad(f, x, constants=[]):
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs='warn')
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret

def jacobian(f, x, constants=[]):
    #import pdb
    #pdb.set_trace()
    #sz = shape(f) #this produced a bug
    #sz = shape(f)[0] #alternative formulation found later in code, should get the same result
    #Commented out these two lines.
    sz = int(shape(f)) #put in in response to bug. This seems to work
    return tt.stacklists([grad(f[i], x) for i in range(sz)])
    ret = th.gradient.jacobian(f, x, consider_constant=constants)
    if isinstance(ret, list):
        ret = tt.concatenate(ret, axis=1)
    return ret

def hessian(f, x, constants=[]):
    #import pdb
    #pdb.set_trace()
    t1 = grad(f,x,constants=constants)
    t2 = jacobian(t1,x,constants=constants)
    return jacobian(grad(f, x, constants=constants), x, constants=constants)

class NestedMaximizer(object):
    #def __init__(self, f1, vs1, f2, vs2):
    def __init__(self, f1, vs1, f2, vs2,bounds={}):
        self.bounds = bounds

        self.f1 = f1
        self.f2 = f2
        self.vs1 = vs1
        self.vs2 = vs2
        self.sz1 = [shape(v)[0] for v in self.vs1]
        self.sz2 = [shape(v)[0] for v in self.vs2]
        for i in range(1, len(self.sz1)):
            self.sz1[i] += self.sz1[i-1]
        self.sz1 = [(0 if i==0 else self.sz1[i-1], self.sz1[i]) for i in range(len(self.sz1))]
        for i in range(1, len(self.sz2)):
            self.sz2[i] += self.sz2[i-1]
        self.sz2 = [(0 if i==0 else self.sz2[i-1], self.sz2[i]) for i in range(len(self.sz2))]
        self.df1 = grad(self.f1, vs1)
        self.new_vs1 = [tt.vector() for v in self.vs1]
        self.func1 = th.function(self.new_vs1, [-self.f1, -self.df1], givens=list(zip(self.vs1, self.new_vs1)))
        def f1_and_df1(x0):
            return self.func1(*[x0[a:b] for a, b in self.sz1])
        self.f1_and_df1 = f1_and_df1
        J = jacobian(grad(f1, vs2), vs1)
        H = hessian(f1, vs1)
        g = grad(f2, vs1)
        self.df2 = -tt.dot(J, ts.solve(H, g))+grad(f2, vs2)
        self.func2 = th.function([], [-self.f2, -self.df2])
        def f2_and_df2(x0):
            for v, (a, b) in zip(self.vs2, self.sz2):
                v.set_value(x0[a:b])
            self.maximize1()
            return self.func2()
        self.f2_and_df2 = f2_and_df2
    def maximize1(self):
        ####################################################
        #Added bounds on vs1 parameter
        B = []
        for v, (a, b) in zip(self.vs1, self.sz1):
            if v in self.bounds:
                B += self.bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([v.get_value() for v in self.vs1])
        ####################################################
        #opt = scipy.optimize.fmin_l_bfgs_b(self.f1_and_df1, x0=x0)[0]
        opt = scipy.optimize.fmin_l_bfgs_b(self.f1_and_df1, x0=x0, bounds=B)[0]
        
        for v, (a, b) in zip(self.vs1, self.sz1):
            v.set_value(opt[a:b])
    def maximize(self, bounds={}):
        t0 = time.time()
        #if not isinstance(bounds, dict):
        if not isinstance(self.bounds, dict):
            #bounds = {v: bounds for v in self.vs2}
        ###########################################
        #Added bounds on vs1 parameters
        #if not isinstance(bounds, dict):
            self.bounds = {v: self.bounds for v in self.vs1+self.vs2}
        ###########################################
        B = []
        bound_count = 0
        for v, (a, b) in zip(self.vs2, self.sz2):
            #if v in bounds:
            if v in self.bounds:
                bound_count += 1
                #B += bounds[v]
                B += self.bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([v.get_value() for v in self.vs2])
        def f(x0):
            #if time.time()-t0>60:
             #   raise Exception('Too long')
            return self.f2_and_df2(x0)
        opt = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=B)
        diag = opt[2]['task']
        opt = opt[0]
        for v, (a, b) in zip(self.vs2, self.sz2):
            v.set_value(opt[a:b])
        self.maximize1()

########################################################################
###### Trajectory Stuff ################################################
def dynamics(x,u,L):
   return tt.stacklists([x[2]*tt.cos(x[3]+u[1]),
                           x[2]*tt.sin(x[3]+u[1]),
                           u[0],
                           (2*x[2]/L)*tt.sin(u[1])
                          ])


def makeTrajectory(dt,L):
    x = tt.dvector()
    u = tt.dvector()
    x_plus = x + dt*dynamics(x,u,L)
    f = th.function([x,u],x_plus)
    return f


def costFunction(features):
    def f(xE,uE,wE,xNE,uNE):
        return sum([wE[j]*c(xE,uE,xNE,uNE) for j,c in enumerate(features)])
    return f


def entropy(distr):
    return -tt.sum(distr*tt.log(distr))/tt.sum(distr)


def returnCost(costFunc):
    xE = tt.dvector('xE')
    uE = tt.dvector('uE')
    wE = tt.dvector('wE')    

    xNE = tt.dvector('xNE')
    uNE = tt.dvector('uNE')

    zE = costFunc(xE,uE,wE,xNE,uNE)

    f = th.function([xE,uE,wE,xNE,uNE],zE,on_unused_input='ignore')

    return f


def dynamicPlotter(mpc_x1,mpc_x2,lane_width=5):
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

########################################################################
############ Cost Function Stuff #######################################
def xPositCost(target_x_posit):
    def f(x,u,*args):
        return 1-tt.exp(1.5*(x[0]-target_x_posit)**2)

    return f


def yPositCost(target_y_posit):
    def f(x,u,*args):
        return 1-tt.exp((x[1]-target_y_posit)**2)

    return f


def velocityCost(target_vel):
    def f(x,u,*args):
        return 1-tt.exp((x[2]-target_vel)**2)

    return f


def headingCost(target_heading):
    def f(x,u,*args):
        return 1-tt.exp(50*(x[3]-target_heading)**2)

    return f


def collisionAvoidanceCost(lat_radius,long_radius):
    def f(x1,u1,x2,u2):
        #Ellipse centred on x2's position.
        del_x = x1[0]-x2[0]
        del_y = x1[1]-x2[1]
        #return -(1-tt.clip(((del_x/lat_radius)**2 + (del_y/long_radius)**2),0,1))
        #https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
        #Ellipse oriented according to x2's orientation. x1, the one incurring the cost, does not want to enter x2's ellipse
        return -(1-tt.clip((((del_x*tt.cos(x2[3])+del_y*tt.sin(x2[3]))/lat_radius)**2 + ((del_x*tt.sin(x2[3])-del_y*tt.cos(x2[3]))/long_radius)**2),0,1))

    return f


def keepAheadCost(veh_length):
    #Penalise x1 car being beehind x2
    def f(x1,u1,x2,u2):
        return tt.exp(.3*(tt.clip(x1[1]-x2[1],-veh_length-1,veh_length+1)))
        # return tt.tanh(.4*(tt.clip(x1[1]-x2[1],-veh_length,veh_length)))

    return f

#####################################################################

if __name__ == '__main__':
    cost_scale_coef = 1
    dt = .2
    num_timesteps = 1 #number of timesteps to execute per iteration
    N = 5 #lookahead horizon
    veh_width = 2
    veh_length = 4
    lane_width = 5

    info_gain_coef = 50

    #########################################
    #Define Features and feature weights
    num_features = 3
    rfeatures = [xPositCost(2+lane_width),velocityCost(15),headingCost(math.pi/2),\
                 collisionAvoidanceCost(veh_width+2,veh_length+4),keepAheadCost(veh_length)]
    hfeatures = [xPositCost(2+lane_width),velocityCost(15),headingCost(math.pi/2),\
                 collisionAvoidanceCost(veh_width+2,veh_length+4),keepAheadCost(veh_length)]

    rcost = costFunction(rfeatures)
    hcost = costFunction(hfeatures)

    #Original parameter values from sadigh implementation
    #r_type_1 = np.array([.01,.01,.005,.8,.6])

    r_type_get_ahead = np.array([.01,.01,.005,.5,1.6])
    r_type_get_behind = np.array([.01,.01,.005,.8,-.6])
    r_types = [r_type_get_ahead,r_type_get_behind]
    r_type_true = 0

    #Original parameter values from Sadigh implementation
    #h_type_stay_ahead = np.array([.01,.008,.0001,0,0])
    #h_type_keep_behind = np.array([.01,.008,.0001,.8,0])
    
    h_type_yield = np.array([.01,.008,.0001,.8,-.3])
    h_type_no_yield = np.array([.01,.008,.0001,.5,.3])
    h_types = [h_type_yield,h_type_no_yield]
    h_type_true = 1

    wr = vector(num_features)
    wh = vector(num_features)

    ########################################
    #Define Vehicle variables
    xr = vector(4)
    ur = [vector(2) for i in range(N)]
    
    xh = vector(4)
    uh = [vector(2) for i in range(N)]
     
    #R's belief over H's type
    b_t = vector(len(h_types))
    b_t.set_value([1/len(h_types) for _ in range(len(h_types))])

    #Initialise vehicle variables
    xr.set_value([2,0.01,15,math.pi/2])
    xh.set_value([2+lane_width,0.01,15,math.pi/2])

    for i in range(N):
        ur[i].set_value([.01,.01])
        uh[i].set_value([.01,.01])

    #Control input bounds
    bounds = [(-2,2),(-math.pi/18,math.pi/18)]
    
    #######################################
    #Define computation graph
    zr,zh = 0,0
    xr_temp = xr
    xh_temp = xh
    for i in range(N):
        zr_temp = rcost(xr_temp,ur[i],wr,xh_temp,uh[i])
        zr += zr_temp
        xr_temp += dt*dynamics(xr_temp,ur[i],veh_length)

        zh_temp = hcost(xh_temp,uh[i],wh,xr_temp,ur[i])
        zh += zh_temp
        xh_temp += dt*dynamics(xh_temp,uh[i],veh_length)

        #Include the information gain component
        ###Value overflow here if not clipped: clip above 1e4 results in overflow
        #   Cost value on the order of +-16 means exponential is e^16. which quickly explodes
        costs_temp = tt.clip(tt.exp(cost_scale_coef*tt.stacklists([hcost(xh_temp,uh[i],w,xr_temp,ur[i]) for w in h_types])),1e-4,1e4)
        b_temp = b_t*costs_temp
        b_temp = b_temp/tt.sum(b_temp)

        #info_gain = info_gain_coef*(entropy(b_t)-entropy(b_temp))
        #zr += info_gain #information gain term
        b_t = b_temp #update belief

    zr += rcost(xr_temp,None,wr,xh_temp,None)
    zh += hcost(xh_temp,None,wh,xr_temp,None)

    optimizer = NestedMaximizer(zh, uh, zr, ur,bounds=bounds)

    #########################################
    #Perform MPC
    r_trajectory = [xr.get_value()]
    h_trajectory = [xh.get_value()]

    traj_func = makeTrajectory(dt,veh_length)
    hcost_eval = returnCost(hcost)
   
    wr.set_value(r_types[r_type_true])
    wh.set_value(h_types[h_type_true])

    h_trajectory_distr = [1/len(h_types) for _ in range(len(h_types))]

    print("{}: {}".format(0,[round(x,2) for x in h_trajectory_distr]))

    for j in range(6*N):
        optimizer.maximize()

        xr_temp = xr.get_value()
        xh_temp = xh.get_value()

        for i in range(num_timesteps):
            p_uh = [np.clip(np.exp(cost_scale_coef*hcost_eval(xh_temp,uh[i].get_value(),w,xr_temp,ur[i].get_value())),1e-8,1e8) for w in h_types]
            h_trajectory_distr = [b*p for b,p in zip(h_trajectory_distr,p_uh)]
            h_trajectory_distr = [x/sum(h_trajectory_distr) for x in h_trajectory_distr]
            if True in [math.isnan(x) for x in h_trajectory_distr]:
                import pdb
                pdb.set_trace()

            print("{}: {}".format((j+i+1)*dt,[round(x,2) for x in h_trajectory_distr]))
            #print("\t{}".format([cost_scale_coef*hcost_eval(xh_temp,uh[i].get_value(),w,xr_temp,ur[i].get_value()) for w in h_types]))
            #print("\t{}".format([np.clip(np.exp(cost_scale_coef*hcost_eval(xh_temp,uh[i].get_value(),w,xr_temp,ur[i].get_value())),1e-4,1e4) for w in h_types]))
            #print("\t{}".format(info_gain_coef*sum([x*math.log(x)/sum(h_trajectory_distr) for x in h_trajectory_distr])))
            xr_temp = traj_func(xr_temp,ur[i].get_value())
            r_trajectory.append(xr_temp)      

            xh_temp = traj_func(xh_temp,uh[i].get_value())
            h_trajectory.append(xh_temp)

        #The computation graph is defined with xr,xh as the roots. So changing these values changes the optimisation
        xr.set_value(xr_temp)
        xh.set_value(xh_temp)

    print("\n\n")

    #################################################
    #Print Results
    print("\t\tR\t\t\tH")
    for i,(r_t,h_t) in enumerate(zip(r_trajectory,h_trajectory)):
        #print("{}: ({},{}) {}\t ({},{}) {}".format(i*dt,round(r_t[0],1),round(r_t[1],1),round(r_t[2],1),round(h_t[0],1),round(h_t[1],1),round(h_t[2],1)))
        print("{}: ({},{}) {}\t ({},{}) {}\t{}".format(i*dt,round(r_t[0],1),round(r_t[1],1),round(r_t[2],1),round(h_t[0],1),round(h_t[1],1),round(h_t[2],1),math.sqrt((r_t[0]-h_t[0])**2 + (r_t[1]-h_t[1])**2)))

    #################################################
    #Plot results
    r_trajectory = np.array(r_trajectory).transpose()
    h_trajectory = np.array(h_trajectory).transpose()

    import pdb
    pdb.set_trace()

    dynamicPlotter(r_trajectory,h_trajectory,lane_width)

    import pdb
    pdb.set_trace()
