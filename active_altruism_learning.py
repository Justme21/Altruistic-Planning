import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
import scipy.optimize
import numpy as np
import time

from itertools import tee

import math
import numpy as np

EPS = .01 #Used to determine preferred action in the reward matrix intersection check

#################################################################################################
##### Adapted from https://github.com/dsadigh/driving-interactions ##############################
######################################################################
#Misc Code for performing bilevel optimisation
def extract(var):
    return th.function([], var, mode=th.compile.Mode(linker='py'))()

def shape(var):
    """Returns function mapping [] to a numpy array containing the shape of var"""
    return extract(var.shape)

def vector(n):
    return th.shared(np.zeros(n))

def matrix(n, m):
    return tt.shared(np.zeros((n, m)))

def grad(f, x, constants=[]):
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs='warn')
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret

def jacobian(f, x, constants=[]):
    sz = int(shape(f))
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

#####################################################################
# Code for performing the bilevel optimisation
class NestedMaximizer(object):
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
        bound_count = 0
        for v, (a, b) in zip(self.vs1, self.sz1):
            if v in self.bounds:
                bound_count += 1
                B += self.bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([v.get_value() for v in self.vs1])
        ####################################################
        opt = scipy.optimize.fmin_l_bfgs_b(self.f1_and_df1, x0=x0, bounds=B)[0]

        for v, (a, b) in zip(self.vs1, self.sz1):
            v.set_value(opt[a:b])
    def maximize(self, bounds={}):
        t0 = time.time()
        if not isinstance(self.bounds, dict):
        ###########################################
        #Added bounds on vs1 parameters
            self.bounds = {v: self.bounds for v in self.vs1+self.vs2}
        ###########################################
        B = []
        bound_count = 0
        for v, (a, b) in zip(self.vs2, self.sz2):
            if v in self.bounds:
                bound_count += 1
                B += self.bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([v.get_value() for v in self.vs2])
        def f(x0):
            return self.f2_and_df2(x0)
        opt = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=B)
        diag = opt[2]['task']
        opt = opt[0]
        for v, (a, b) in zip(self.vs2, self.sz2):
            v.set_value(opt[a:b])
        self.maximize1()

###################################################################
#Optimizer used to perform base level optimisation
class Maximizer(object):
    def __init__(self, f, vs, g={}, pre=None, gen=None, method='bfgs', eps=1, iters=100000, debug=False, inf_ignore=np.inf,bounds={}):
        self.bounds = bounds

        self.inf_ignore = inf_ignore
        self.debug = debug
        self.iters = iters
        self.eps = eps
        self.method = method
        def one_gen():
            yield
        self.gen = gen #Generator for the data; sets trajectories for all the cars
        if self.gen is None:
            self.gen = one_gen
        self.pre = pre
        self.f = f
        self.vs = vs #List containing vector of weights. List only has a single vector in it
        #counts the number of entries in each vector of weights (in our setting this is just the number of weights)
        self.sz = [shape(v)[0] for v in self.vs] #e.g. [4] if there are 4 weights
        #In the case there are multiple weights (layers, hierarchies?) this converts them all into a single list and stores the indices
        #Not really relevant here.
        for i in range(1,len(self.sz)): #in our setting len(self.sz)==1
            self.sz[i] += self.sz[i-1] #Cumulative value?
        #Pairs of consecutive cumulative sz values (the ranges corresponding to each layer.
        #In this case it is [(0,n)] as we have one layer and n weeights
        self.sz = [(0 if i==0 else self.sz[i-1], self.sz[i]) for i in range(len(self.sz))]
        if isinstance(g, dict): #in default case g is {}
            #df/d_weight computed here. df is theano tensor of grad(f,v)
            #This will be one vector containing all the derivatives in a long line (concattenated together)
            self.df = tt.concatenate([g[v] if v in g else grad(f, v) for v in self.vs])
        else:
            self.df = g
        self.new_vs = [tt.vector() for v in self.vs] #should be a list with 1 vector in it
        #defining a synbolic function mapping weights to negative f and negative df
        #givens are list of pairs Variables (Var1,Var2) that are substituted in the computation graph (Var2 replaces Var1)
        self.func = th.function(self.new_vs, [-self.f, -self.df], givens=list(zip(self.vs, self.new_vs)))
        def f_and_df(x0):
            if self.debug:
                print(x0)
            s = None
            N = 0
            for _ in self.gen(): #trajectory values are set
                if self.pre:
                    for v, (a, b) in zip(self.vs, self.sz):
                        v.set_value(x0[a:b])
                    self.pre()
                res = self.func(*[x0[a:b] for a, b in self.sz])
                #Catch case when f or df go to NaN
                if np.isnan(res[0]).any() or np.isnan(res[1]).any() or (np.abs(res[0])>self.inf_ignore).any() or (np.abs(res[1])>self.inf_ignore).any():
                    continue
                if s is None:
                    s = res
                    N = 1
                else:
                    s[0] += res[0]
                    s[1] += res[1]
                    N += 1
            s[0]/=N
            s[1]/=N
            return s
        self.f_and_df = f_and_df
    def argmax(self, vals={}):
        if not isinstance(self.bounds,dict):
            self.bounds = {v:self.bounds for v in self.vs}
        B = []
        for v, (a, b) in zip(self.vs, self.sz): #v=vector of weights (theta), a=0, b=4 (if 4 weights are in reward)
            if v in bounds:
                B += self.bounds[v]
            else:
                B += [(None, None)]*(b-a)
        #x0 here now refers to an numpy vector initial values of the weights; the vector theta
        x0 = np.hstack([np.asarray(vals[v]) if v in vals else v.get_value() for v in self.vs])
        if self.method=='bfgs':
            opt = scipy.optimize.fmin_l_bfgs_b(self.f_and_df, x0=x0, bounds=B)[0]
        elif self.method=='gd':
            opt = x0
            for i in range(self.iters):
                print("Iteration {}: \t Opt: {}".format(i,opt))
                opt -= self.f_and_df(opt)[1]*self.eps
        else:
            opt = scipy.optimize.minimize(self.f_and_df, x0=x0, method=self.method, jac=True).x
        return {v: opt[a:b] for v, (a, b) in zip(self.vs, self.sz)}
    def maximize(self, *args, **vargs):
        result = self.argmax(*args, **vargs)
        for v, res in list(result.items()):
            v.set_value(res)

##########################################################################################################
###### Trajectory Stuff ##################################################################################
def dynamics(x,u,L):
   """Kinematic Bicycle dynamics model"""
   return tt.stacklists([x[2]*tt.cos(x[3]+u[1]),
                           x[2]*tt.sin(x[3]+u[1]),
                           u[0],
                           (2*x[2]/L)*tt.sin(u[1])
                          ])


def makeTrajectory(dt,L):
    """Returns a function that maps a (state, action) theano vector pair to the next state"""
    x = tt.dvector()
    u = tt.dvector()
    x_plus = x + dt*dynamics(x,u,L)
    f = th.function([x,u],x_plus)
    return f


def costFunction(features):
    """Returns a function that returns the cost the ego agent incurs for the given trajectories"""
    #All inputs to f are theano vectors
    #xE/NE = vector of ego/non-ego states
    #uE/NE = vector of ego/non-ego actions
    #wE = vector of weights in the ego cost function
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
############ MPC Cost Function Stuff ###################################
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
        #https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
        #Ellipse oriented according to x2's orientation. x1, the one incurring the cost, does not want to enter x2's ellipse
        return -(1-tt.clip((((del_x*tt.cos(x2[3])+del_y*tt.sin(x2[3]))/lat_radius)**2 + ((del_x*tt.sin(x2[3])-del_y*tt.cos(x2[3]))/long_radius)**2),0,1))

    return f


def keepAheadCost(veh_length):
    #Penalise x1 car being beehind x2
    def f(x1,u1,x2,u2):
        return tt.tanh((tt.clip(x1[1]-x2[1],-veh_length,veh_length)))

    return f


#####################################################################
############# Game Theory Reward Matrix Stuff #######################
def computeReward(rewards,alpha,i):
    """Given reward; (reward to row player, reward to column player),
       as well as the altrusim coefficient alpha, and i indexing the player (i=0 for row, i=1 for column), return player's reward"""
    return (1-alpha)*rewards[i]+alpha*rewards[1-i]


def computePreferredActions(reward_matrix_row):
    """For a given row of a reward matrix, returns a dict mapping from ranges of altruism values in [0,1] and the preferred action in that range,
       according to the rewards in the row"""
    intersections = {}
    for i,col1 in enumerate(reward_matrix_row):
        for j,col2 in enumerate(reward_matrix_row[i+1:]):
            #Computing the value of altruism where the rewards intersect, as this is the value where the preferred action changes  
            num = col1[1] - col2[1]
            denom = col2[0]-col1[0]+col1[1]-col2[1]
            if denom ==0:
                print("Error, denominator is 0")
                import pdb
                pdb.set_trace()
            else:
                intersection_point = num/denom
                #Identify which action was preferred for alpha<intersection and alpha>intersection
                if computeReward(col1,intersection_point-EPS,1)>computeReward(col2,intersection_point-EPS,1):
                    alt_below_preferred_action_index = i
                    alt_above_preferred_action_index = i+1+j
                else:
                    alt_below_preferred_action_index = i+1+j
                    alt_above_preferred_action_index = i
                    preferred_action = i+1+j
                intersections[intersection_point] = (alt_below_preferred_action_index,alt_above_preferred_action_index)

    range_to_action = {}
    low_range_val = 0
    #Sort intersections from low to high
    ordered_intersection_vals = sorted(intersections.keys())
    for i,intersect_val in enumerate(ordered_intersection_vals):
        #Don't count empty ranges
        if intersect_val != low_range_val:
            if intersect_val>1:
                range_to_action[(low_range_val,1)] = intersections[intersect_val][0]
            else:
                range_to_action[(low_range_val,intersect_val)] = intersections[intersect_val][0]
            cur_index = i-1
            #cur_index indexes intersection values (so it also indexes ranges, since the ranges will be created in order)
            #while there are other indices and the current action would receive a higher reward at a previous index (range)
            #  than the action currently associated with that interval, we replace the action at that interval with the current action
            while cur_index >=0 and \
               computeReward(reward_matrix_row[intersection[intersect_val][0]],ordered_intersection_vals[cur_index]-EPS,1)>\
               computeReward(reward_matrix_row[intersection[ordered_intersection_vals[cur_index]][0]],ordered_intersection_vals[cur_index]-EPS,1):
                range_to_action[list(range_to_action.keys())[cur_index]] = intersections[intersect_val][0]
                cur_index -= 1
        low_range_val = intersect_val

        if intersect_val>1: break

    if intersect_val<1:
        #We are exploiting the fact that the intersect_val value will persist after the for loop
        range_to_action[(intersect_val,1)] = intersections[intersect_val][1]

    return range_to_action


def getProb(intersection_ranges,action_index,c_alpha_distr):
    """Return the probability of C choosing action 'action_index' given the row of the reward matrix chosen by R
       and the distribution over the possible values for C's altruism value"""
    prob = 0
    #Ranges of altruism for which the specified action is preferred
    possible_ranges = [x for x in intersection_ranges if intersection_ranges[x]==action_index]
    for alpha_range in c_alpha_distr:
        for int_range in possible_ranges:
            #We presume the intersection ranges are in order. Once they no longer ovelap with the distr, they won't again
            if int_range[0]>alpha_range[1]: break
            #Probability from a uniform distribution is the proportion the distribution occupied by the specified range
            try:
                #Probability of intersection is probability of intersecting with range times probability of range
                prob += c_alpha_distr[alpha_range]*(min(alpha_range[1],max(alpha_range[0],int_range[1])) - max(alpha_range[0],min(alpha_range[1],int_range[0])))/(alpha_range[1]-alpha_range[0])
            except ZeroDivisionError:
                print("0-length range while in getProb")
                import pdb
                pdb.set_trace()
    
    return prob


def expectedValue(matrix_row,intersection_ranges,r_alpha,c_alpha_distr):
    """Compute the expected value of a given row from a reward matrix, and the ranges of altruism in which each action is preferred"""
    e_reward = 0
    #Prob is the probability of C choosing action 0, if the given row of the game matrix were chosen by R,
    # given the distribution over C's altruism value.
    for i,col in enumerate(matrix_row):
        prob = getProb(intersection_ranges,i,c_alpha_distr)
        e_reward += prob*computeReward(col,r_alpha,0)

    return e_reward


def rewardMatrixExpectedValues(reward_matrix,preferred_actions,r_alpha,c_alpha_distr):
    """Compute the expected value of every row of the reward matrix"""
    expected_values = []
    for row,intersection_ranges in zip(reward_matrix,preferred_actions):
        expected_values.append(expectedValue(row,intersection_ranges,r_alpha,c_alpha_distr))

    return expected_values


def updateDistr(cur_distr,intersection_ranges,action_distr):
    """Use Bayes Rule to update the belief over the value of \alpha"""
    distr_copy = dict(cur_distr)
    for inter_range in intersection_ranges:
        temp_distr = {}
        for distr_range in cur_distr:
            #The two distributions overlap.
            if inter_range[0]<distr_range[1] and inter_range[1]>distr_range[0]:
                #We have no information about the probability of values outside the intersection range
                if inter_range[0]>distr_range[0]:
                    temp_distr[(distr_range[0],inter_range[0])] = cur_distr[distr_range]*(inter_range[0]-distr_range[0])/(distr_range[1]-distr_range[0])
                if inter_range[1]<distr_range[1]:
                    temp_distr[(inter_range[1],distr_range[1])] = cur_distr[distr_range]*(distr_range[1]-inter_range[1])/(distr_range[1]-distr_range[0])
                #Lower and Upper bounds for the region we have information about
                low_bound = max(inter_range[0],distr_range[0])
                high_bound = min(inter_range[1],distr_range[1])
                #Bayes Theorem: P(alt \in [low,high]|obs) \propto \sum_{action}P(action|obs)*P(action|alt \in [low,high])*P(alt \in [low,high])
                #distr provides the prior
                # within the inter_range the probability of the preferred action = 1
                # P(action|obs) given by action_distr
                temp_distr[(low_bound,high_bound)] = cur_distr[distr_range]*((high_bound-low_bound)/(distr_range[1]-distr_range[0]))*action_distr[intersection_ranges[inter_range]]
            else:
                temp_distr[distr_range] = cur_distr[distr_range]
        cur_distr = temp_distr

    #Normalise the resulting distribution
    normaliser = sum(cur_distr.values())
    if normaliser != 0:
        for x in cur_distr: cur_distr[x]/=normaliser
    else:
        #If the probability of choosing any action is 0, then all actions are chosen equally
        #So uniform distribution
        for x in cur_distr: cur_distr[x] = x[1]-x[0]
    return cur_distr


def augmentRewardMatrix(reward_matrix,preferred_actions,r_alpha,c_alpha_distr):
    """Compute R+J where J is Expected Reward Gain"""
    new_reward_matrix = []
    expected_value_gain = 0
    expected_values = rewardMatrixExpectedValues(reward_matrix,preferred_actions,r_alpha,c_alpha_distr)
    for i,(row,intersection_ranges) in enumerate(zip(reward_matrix,preferred_actions)):
        new_row = []
        intersection_ranges = preferred_actions[i] #This is unnecessary
        for j,col in enumerate(row):
            #If the turning point is outside of the range we already know, then there is no information gained
            temp_action_distr = [0 for _ in range(len(row))]
            temp_action_distr[j] = 1 #Pretending we have perfect observability
            temp_c_alpha_distr = updateDistr(c_alpha_distr,intersection_ranges,temp_action_distr)
            
            new_expected_values = rewardMatrixExpectedValues(reward_matrix,preferred_actions,r_alpha,temp_c_alpha_distr)
            expected_value_gain = sum([abs(new_val-old_val) for old_val,new_val in zip(expected_values,new_expected_values)])
            
            new_row.append((col[0]+expected_value_gain,col[1]))

        new_reward_matrix.append(list(new_row))

    return new_reward_matrix


def augmentRewardMatrixInfoGain(reward_matrix,preferred_actions,r_alpha,c_alpha_distr):
    """Compute R+J where J is Information Gain"""
    new_reward_matrix = []
    expected_value_gain = 0
    expected_values = rewardMatrixExpectedValues(reward_matrix,preferred_actions,r_alpha,c_alpha_distr)
    for i,(row,intersection_ranges) in enumerate(zip(reward_matrix,preferred_actions)):
        new_row = []
        intersection_ranges = preferred_actions[i]
        for j,col in enumerate(row):
            #If the turning point is outside of the range we already know, then there is no information gained
            temp_action_distr = [0 for _ in range(len(row))]
            temp_action_distr[j] = 1 #Pretending we have perfect observability
            temp_c_alpha_distr = updateDistr(c_alpha_distr,intersection_ranges,temp_action_distr)
            
            #new_expected_values = rewardMatrixExpectedValues(reward_matrix,preferred_actions,r_alpha,temp_c_alpha_distr)
            #expected_value_gain = sum([abs(new_val-old_val) for old_val,new_val in zip(expected_values,new_expected_values)])
            init_info = -sum([c_alpha_distr[k]*math.log(1/(k[1]-k[0])) for k in c_alpha_distr if c_alpha_distr[k]>0])
            final_info = -sum([temp_c_alpha_distr[k]*math.log(1/(k[1]-k[0])) for k in temp_c_alpha_distr if temp_c_alpha_distr[k]>0])
            information_gain = init_info-final_info

            #new_row.append((col[0]+expected_value_gain,col[1]))
            new_row.append((col[0]+information_gain,col[1]))

        new_reward_matrix.append(list(new_row))

    return new_reward_matrix


def noUpdate(matrix,*args):
    """Returns the matrix given as argument unchanged"""
    #At the end of each iteration of MPC, the reward matrix is updated. This is so that the augmented reward matrix, 
    # which adds on the expected value based on the belief of the type of column player, can be updated.
    # The baseline case does not need updating. 
    return matrix

########################################################################

if __name__ == '__main__':
    #########################################
    ##### Game Theory Reasoning #############
    r_alpha = 0 #Known
    c_alpha = .9 #Unknown
 
    ##### Defining the reward matrix ################
    #Values used in Active Learning Experiment
    r_values = [[3,-5],[-1,1],[-1,2]]
    c_values = [[0,7],[2,1],[2,2]]

    #Values used in Information Sufficiency experiment
    #r_values = [[5,-2],[1,0]]
    #c_values = [[-4,1],[-4,1]]

    #Testing Reward Matrix Values
    #r_values = [[3,-10],[0,1],[2,-1]]
    #c_values = [[-2,3],[-2,3],[0,3]]

    reward_matrix = []
    for r,c in zip(r_values,c_values):
        temp_val = []
        for r_val,c_val in zip(r,c):
            temp_val.append((r_val,c_val))
        reward_matrix.append(list(temp_val))

    #Preferred actions is a list of dictionaries mapping ranges of altruism values to the index of the action that 
    # is preferred in this range, if the corresponding row of the reward matrix is selected
    preferred_actions = []
    for row in reward_matrix:
        preferred_actions.append(computePreferredActions(row))

    #Distribution over column player's altruism coefficient
    # Range of values -> probability of altruism coefficient being in this range
    c_alpha_distr = {(0,1):1} 
    #c_alpha_distr = {(5/12,1):1} 
    #c_alpha_distr = {(0,.5):.5,(.5,1):.5} 

    #Vanilla Expected Value Computation
    actions_expected_value = rewardMatrixExpectedValues(reward_matrix,preferred_actions,r_alpha,c_alpha_distr)

    #Augment Reward Matrix to include Information Gain
    augmented_reward_matrix = augmentRewardMatrix(reward_matrix,preferred_actions,r_alpha,c_alpha_distr)
    augmented_reward_matrix_alt = augmentRewardMatrixInfoGain(reward_matrix,preferred_actions,r_alpha,c_alpha_distr)

    actions_aug_expected_value = rewardMatrixExpectedValues(augmented_reward_matrix,preferred_actions,r_alpha,c_alpha_distr)
    actions_info_expected_value = rewardMatrixExpectedValues(augmented_reward_matrix_alt,preferred_actions,r_alpha,c_alpha_distr)

    ###############################################
    #Setting Font Size for Plots
    #import matplotlib
    #font = {'family' : 'normal',
    #        'size'   : 25}
    #matplotlib.rc('font', **font)
    ###############################################
    #Plotting the expected values
    #plot_vals = actions_expected_value
    #import matplotlib.pyplot as plt
    #colours = [None for _ in range(3)]
    #vals = list(plot_vals)
    #for val,col in zip(sorted(vals),["red","black","green"]):
    #    colours[vals.index(val)] = col
    #plt.bar(["A1","A2","A3"],plot_vals,color=colours)
    #plt.xticks(np.arange(3),["A1","A2","A3"])
    ##plt.title("Action Expected Reward Gain")
    #plt.title("Action Expected Reward")
    #plt.xlabel("Action")
    #plt.ylabel("Expected Reward")
    #
    #plt.show()
    ##############################################
    #Plotting the relative expectde values
    #import matplotlib.pyplot as plt
    #old_plot_vals = [0,0]
    #for plot_distr in [{(0,1):1},{(round(5/12,2),1):1}]:
    #    #Vanilla Expected Value Computation
    #    plot_actions_expected_value = rewardMatrixExpectedValues(reward_matrix,preferred_actions,r_alpha,plot_distr)

    #    #Augment Reward Matrix to include Information Gain
    #    plot_augmented_reward_matrix = augmentRewardMatrix(reward_matrix,preferred_actions,r_alpha,plot_distr)
    #    plot_augmented_reward_matrix_alt = augmentRewardMatrixInfoGain(reward_matrix,preferred_actions,r_alpha,plot_distr)

    #    plot_actions_aug_expected_value = rewardMatrixExpectedValues(plot_augmented_reward_matrix,preferred_actions,r_alpha,plot_distr)
    #    plot_actions_info_expected_value = rewardMatrixExpectedValues(plot_augmented_reward_matrix_alt,preferred_actions,r_alpha,plot_distr)

    #    plot_vals = [x[1]-plot_actions_expected_value[1] for x in [plot_actions_info_expected_value,plot_actions_aug_expected_value]]

    #    plt.bar(["Info Gain","Exp Reward Gain"],plot_vals,label="b=U{}".format(list(plot_distr.keys())[0]),bottom=old_plot_vals)
    #    old_plot_vals = plot_vals
    #plt.xticks(np.arange(2),["Info Gain","Exp Reward Gain"])
    #plt.title("Values of J")
    #plt.xlabel("Action")
    #plt.ylabel("J")
    #plt.legend()

    #plt.show()
    import pdb
    pdb.set_trace()
    ##############################################
    #MPC Parameters
    cost_scale_coef = 1
    dt = .2
    num_timesteps = 1 #number of timesteps to execute per iteration
    N = 5 #lookahead horizon
    veh_width = 2
    veh_length = 4
    lane_width = 5

    info_gain_coef = 50

    #########################################
    #Define Features and feature weights for MPC cost functions
    num_features = 5
    rfeatures = [xPositCost(2),xPositCost(2+lane_width),velocityCost(15),headingCost(math.pi/2),\
                 collisionAvoidanceCost(veh_width+.5,veh_length+1),keepAheadCost(veh_length)]
    hfeatures = [xPositCost(2+lane_width),velocityCost(15),headingCost(math.pi/2),\
                 collisionAvoidanceCost(veh_width+.5,veh_length+1),keepAheadCost(veh_length)]

    rcost = costFunction(rfeatures)
    hcost = costFunction(hfeatures)

    r_type_get_ahead = np.array([0,.01,.1,.005,.8,.6])
    r_type_get_behind = np.array([0,.01,.01,.005,.8,-.6])
    r_type_explore = np.array([.1,0,.01,.005,0,1])
    r_types = [r_type_get_ahead,r_type_get_behind,r_type_explore]
    r_type_true = 0

    #How human might behave if robot attempts merge ahead
    h_type_yield = np.array([.01,.001,.005,0,-1])
    h_type_no_yield = np.array([.01,.001,.005,0,1])
    h_type_r1_resp_1 = [h_type_yield,h_type_no_yield]

    #How human might behave if robot attempts merge behind
    h_type_let_go_behind = np.array([.01,.1,.005,.8,0])
    h_type_r1_resp_2 = [h_type_let_go_behind,h_type_let_go_behind]

    #How human might behave if robot does exploratory action
    h_type_give_way = np.array([.01,.001,.005,0,-1])
    h_type_carry_on = np.array([.01,.001,.005,0,1])
    h_type_r1_resp_3 = [h_type_give_way,h_type_carry_on]

    h_types = [h_type_r1_resp_1,h_type_r1_resp_2,h_type_r1_resp_3]
    h_type_true = 0

    wr = vector(num_features)
    wh = vector(num_features)

   ########################################
    #Define Vehicle variables
    xr = vector(4)
    ur = [vector(2) for i in range(N)]

    xh = vector(4)
    uh = [vector(2) for i in range(N)]


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
        #Standard Computation Graph
        zr_temp = rcost(xr_temp,ur[i],wr,xh_temp,uh[i])
        zr += zr_temp
        xr_temp += dt*dynamics(xr_temp,ur[i],veh_length)

        zh_temp = hcost(xh_temp,uh[i],wh,xr_temp,ur[i])
        zh += zh_temp
        xh_temp += dt*dynamics(xh_temp,uh[i],veh_length)

        #Information Gain
        #costs_temp = tt.clip(tt.exp(cost_scale_coef*tt.stacklists([hcost(xh_temp,uh[i],w,xr_temp,ur[i]) for w in h_types])),1e-4,1e4)
        #b_temp = b_t*costs_temp
        #b_temp = b_temp/tt.sum(b_temp)

        #info_gain = info_gain_coef*(entropy(b_t)-entropy(b_temp))
        #zr += info_gain #information gain term
        #b_t = b_temp #update belief


    zr += rcost(xr_temp,None,wr,xh_temp,None)
    zh += hcost(xh_temp,None,wh,xr_temp,None)

    optimizer = NestedMaximizer(zh, uh, zr, ur,bounds=bounds)

    ##############################################
    ###### Perform MPC ###########################
    r_act_trajectory = [-1] #Default initial value, since no action is associated with initial state
    h_act_trajectory = [-1] #Default initial value, since no action is associaed with initial state
    r_trajectory = [xr.get_value()]
    h_trajectory = [xh.get_value()]

    c_alpha_distr_record = [dict(c_alpha_distr)]

    traj_func = makeTrajectory(dt,veh_length)
    hcost_eval = returnCost(hcost)

    prev_act = None
    prev_distr = None

    ### Choose Cost Function by Solving Game Matrix
    matrix = reward_matrix
    updateMatrix = noUpdate
    #matrix = augmented_reward_matrix
    #updateMatrix = augmentRewardMatrix
    #matrix = augmented_reward_matrix_alt
    #updateMatrix = augmentRewardMatrixInfoGain
    for n in range(int(6*N)):
        #For evaluating Game Matrix values
        expected_values = rewardMatrixExpectedValues(matrix,preferred_actions,r_alpha,c_alpha_distr)
   
        #Compute the Stackelberg equilibrium with row player as leader
        r_act = np.argmax(np.array(expected_values))
        h_act = np.argmax(np.array([computeReward(col,c_alpha,1) for col in reward_matrix[r_act]]))

        #For evaluating parameter values
        #r_act = r_type_true
        #h_act = h_type_true

        cur_h_types = h_types[r_act]
        if prev_act != r_act:
            print("Resetting trajectory distr")
            h_trajectory_distr = [1/len(cur_h_types) for _ in range(len(cur_h_types))]
        
        wr.set_value(r_types[r_act])
        wh.set_value(cur_h_types[h_act])

        optimizer.maximize()

        xr_temp = xr.get_value()
        xh_temp = xh.get_value()

        for i in range(num_timesteps):
            #Update Trajectory Values
            r_act_trajectory.append(r_act)
            h_act_trajectory.append(h_act)

            xr_temp = traj_func(xr_temp,ur[i].get_value())
            r_trajectory.append(xr_temp)

            xh_temp = traj_func(xh_temp,uh[i].get_value())
            h_trajectory.append(xh_temp)

            #Compute distribution over actions
            action_distr = [np.clip(np.exp(cost_scale_coef*hcost_eval(xh_temp,uh[i].get_value(),w,xr_temp,ur[i].get_value())),1e-8,1e8) for w in cur_h_types]

            h_trajectory_distr = [b*p for b,p in zip(h_trajectory_distr,action_distr)]
            h_trajectory_distr = [x/sum(h_trajectory_distr) for x in h_trajectory_distr]
            if True in [math.isnan(x) for x in h_trajectory_distr]:
                import pdb
                pdb.set_trace()

            print("{}: {} R: {}\tH: {}".format((n+i+1)*dt,[round(x,2) for x in h_trajectory_distr],[round(x,2) for x in xr_temp],[round(x,2) for x in xh_temp]))

            #Update c_alpha_range based on observed equilibrium
            intersection_ranges = preferred_actions[r_act]
            c_alpha_distr = updateDistr(c_alpha_distr,intersection_ranges,action_distr)

            #Record new value of distribution of c-alpha values
            c_alpha_distr_record.append(dict(c_alpha_distr))

        #################################################
        #For testing the case with perfect observability
        #intersection_ranges = preferred_actions[r_act]
        #action_distr = [0 for _ in h_types]
        #action_distr[h_act] = 1
        #c_alpha_distr = updateDistr(c_alpha_distr,intersection_ranges,action_distr)
        #################################################

        matrix = updateMatrix(reward_matrix,preferred_actions,r_alpha,c_alpha_distr)   
        if prev_act != None and prev_act != r_act:
            print("\nAction Change: {}-{}\nDistr: {}-{}\nExpected_Values: {}".format(prev_act,r_act,prev_distr,c_alpha_distr,expected_values))
        prev_act = r_act
        prev_distr = dict(c_alpha_distr)

        print("\n N: {}".format(n))
        print("R_act: {}\tH_act: {}".format(r_act,h_act))
        print("Distr: {}".format(c_alpha_distr))
        print("Expected Values: {}".format(expected_values))
        print("R-Reward: {}\tH-Reward: {}".format(matrix[r_act][h_act][0],matrix[r_act][h_act][1]))
        ################################################

        #The computation graph is defined with xr,xh as the roots. So changing these values changes the optimisation
        xr.set_value(xr_temp)
        xh.set_value(xh_temp)


    print("\t\tR\t\t\tH")
    for i,(r_t,r_t_act,h_t,h_t_act) in enumerate(zip(r_trajectory,r_act_trajectory,h_trajectory,h_act_trajectory)):
        print("{}: ({},{}) {}\t ({},{}) {}\t{}\t{}:{}".format(i*dt,round(r_t[0],1),round(r_t[1],1),round(r_t[2],1),round(h_t[0],1),round(h_t[1],1),round(h_t[2],1),math.sqrt((r_t[0]-h_t[0])**2 + (r_t[1]-h_t[1])**2),r_t_act,h_t_act))

    #Write results to file
    filename = updateMatrix.__name__
    results_file = open("{}_{}.txt".format(filename,str(c_alpha)[2:]),'w')
    for i,(r_t,r_t_act,h_t,h_t_act) in enumerate(zip(r_trajectory,r_act_trajectory,h_trajectory,h_act_trajectory)):
        results_file.write("{} {} {} {} {} {} {} {}\n".format(round(r_t[0],1),round(r_t[1],1),round(r_t[2],1),round(h_t[0],1),round(h_t[1],1),round(h_t[2],1),r_t_act,h_t_act))
    results_file.close()

    #Record how the belief changed over time
    distr_file = open("{}_{}_distr.txt".format(filename,str(c_alpha)[2:]),'w')
    for distr in c_alpha_distr_record:
        for key in distr:
            distr_file.write("{}:{}\t".format(key,distr[key]))
        distr_file.write("\n")

    distr_file.close()
