import math
import numbers

class Line():
    def __init__(self,*args):
        args = list(args)
        args.reverse()
        self.coefs = [(x,i) for i,x in enumerate(args)]


    def dot(self,coefs=None):
        if coefs is None: coefs = list(self.coefs)
        else: coefs = list(coefs)

        for i in range(len(coefs)):
            if coefs[i][1] == 0:
                coefs[i] = (0,0)
            else:
                coefs[i] = (coefs[i][0]*coefs[i][1],coefs[i][1]-1)
        return coefs


class Trajectory():
    def __init__(self,init_state,dest_state,time_len,label=None):
        if init_state is None or dest_state is None:
            print("Error, default values for states are invalid")
            exit(-1)

        self.traj_len_t = time_len
        #In this implementation coordinates are in Frenet Coordinate frame
        # So x is lateral displacement and y longitudinal.
        # Since application is on straight road we can treat this as analogous to x,y
        self.line_x,self.line_y = defineTrajectory(init_state,dest_state,time_len)
        self.computeDerivatives()

        #Init and Dest state stored for evaluation purposes only
        self.init_state = init_state
        self.dest_state = self.state(time_len)

        self.label = label


    def computeDerivatives(self):
        self.x = self.line_x.coefs
        self.y = self.line_y.coefs
        self.x_dot = self.line_x.dot()
        self.y_dot = self.line_y.dot()
        self.x_dot_dot = self.line_x.dot(self.x_dot)
        self.y_dot_dot = self.line_y.dot(self.y_dot)


    def action(self,t,axle_length):
        if t>self.traj_len_t:
            return 0,0
        else:
            x = evaluate(t,self.x)
            y = evaluate(t,self.y)

            x_dot = evaluate(t,self.x_dot)
            y_dot = evaluate(t,self.y_dot)
            x_dot_dot = evaluate(t,self.x_dot_dot)
            y_dot_dot = evaluate(t,self.y_dot_dot)

            denom_a = math.sqrt(x_dot**2 + y_dot**2)
            denom_yaw = denom_a**3

            acceleration = ((x_dot*x_dot_dot)+(y_dot*y_dot_dot))/denom_a
            #I think due to the flipping of the y-axis yaw rate needs to be computed with negatives of y-associated values
            # This works but am not sure. Original is commented out below
            #yaw_rate = math.degrees(math.atan(((x_dot*y_dot_dot)-(y_dot*x_dot_dot))*axle_length/denom_yaw))
            yaw_rate = math.degrees(math.atan(((x_dot*-y_dot_dot)-(-y_dot*x_dot_dot))*axle_length/denom_yaw))

        return acceleration,yaw_rate


    def position(self,t):
        return (evaluate(t,self.x),evaluate(t,self.y))


    #def velocity(self,t):
    #    return math.sqrt(evaluate(t,self.x_dot)**2 + evaluate(t,self.y_dot)**2)

    def velocity(self,t):
        """Returns velocity parametrised into x,y coodinates v_x,v_y"""
        return (evaluate(t,self.x_dot),evaluate(t,self.y_dot))


    def acceleration(self,t):
        """Returns acceleration parametrised into (x,y) coordinates"""
        # Returns (x_dot_dot(t),y_dot_dot(t))
        return (evaluate(t,self.x_dot_dot),evaluate(t,self.y_dot_dot))


    def heading(self,t):
        x_dot = evaluate(t,self.x_dot)
        #minus here to capture the axis flip
        #y_dot = -evaluate(t,self.y_dot)
        y_dot = evaluate(t,self.y_dot)

        if round(x_dot,2) == 0:
            if round(y_dot,2)>=0: heading = 90
            else:
                #print("In heading now for x_dot = 0")
                #import pdb
                #pdb.set_trace()
                heading = 270

        else:
            heading = math.degrees(math.atan(y_dot/x_dot))
            if x_dot<0: heading = (heading+180)%360#atan has domain (-90,90)
            #if heading == 270:
                #print("In heading now for x_dot != 0")
                #import pdb
                #pdb.set_trace()

        heading%=360
        return heading


    def state(self,t,axle_length=None):
        """Returns the estimated state at a known timepoint along the trajectory.
           ACtion omitted as this would require vehicle axle length"""
        posit = self.position(t)
        vel = math.sqrt(sum([x**2 for x in self.velocity(t)]))
        heading = self.heading(t)

        if axle_length is not None:
            acceleration,yaw_rate = self.action(t,axle_length)
        else:
            acceleration,yaw_rate = None,None

        #Acceleration and Yaw Rate are as inputs to Driving Simulator. Parametrised Acceleration is useful for building trajectories
        state = {"position":posit,"velocity":vel,"heading":heading,"acceleration":acceleration,"yaw_rate":yaw_rate,"parametrised_acceleration": self.acceleration(t)}
        return state


    def completePositionList(self,dt=.1):
        t = 0
        position_list = []
        while round(t,2)<=self.traj_len_t:
            position_list.append(self.position(round(t,2)))
            t += dt

        return position_list


    def completeHeadingList(self,dt=.1):
        t = 0
        heading_list = []
        while round(t,2)<=self.traj_len_t:
            #Weird instabilities, round to 2 decimal places to minimise this effect
            heading_list.append(self.heading(round(t,2)))
            #if self.heading(round(t,2)) > 180.0:
            #    import pdb
            #    pdb.set_trace()
            t += dt

        #if 270.0 in heading_list:
        #    print("Jesus")
        #    import pdb
        #    pdb.set_trace()
        return heading_list


    def completeVelocityList(self,dt=.1):
        t = 0
        velocity_list = []
        while round(t,2)<=self.traj_len_t:
            velocity_list.append(math.sqrt(sum([x**2 for x in self.velocity(round(t,2))])))
            t += dt

        return velocity_list


    def completeActionList(self,axle_length,dt=.1):
        t = 0
        action_list = []
        while round(t,2)<=self.traj_len_t:
            action_list.append(self.action(round(t,2),axle_length))
            t += dt

        return action_list


def defineTrajectory(init_state,dest_state,T):
    #Assumptions
    # Given: initial lateral and longitudinal (x,y) position; initial longitudinal velocity; final (lat,long) position,final longitudinal velocity

    init_heading = init_state["heading"]
    (d_0,s_0) = init_state["position"]
    #By relieving the constraint that initial velocity must be in the direction of lane we can generate trajectories that
    # Start partway through the intended manoeuvre
    init_v_x = init_state["velocity"]*math.cos(math.radians(init_heading))
    init_v_y = init_state["velocity"]*math.sin(math.radians(init_heading)) 
    s_dot_0 = init_v_y
    d_dot_0 = init_v_x
    d_dot_dot_0,s_dot_dot_0 = init_state["parametrised_acceleration"]

    dest_heading = dest_state["heading"] #This is dangerous as it means we have to specify the direction we want to end up facing
    (d_T,_) = dest_state["position"]
    dest_v_x = dest_state["velocity"]*math.cos(math.radians(dest_heading))
    dest_v_y = dest_state["velocity"]*math.sin(math.radians(dest_heading))
    d_dot_T = dest_v_x
    s_dot_T = dest_v_y
    d_dot_dot_T,s_dot_dot_T = dest_state["parametrised_acceleration"]

    sig_0 = s_0
    sig_1 = s_dot_0
    sig_2 = s_dot_dot_0/2
    sig_3 = (1/(T**2))*(s_dot_T-sig_1-((4/3)*sig_2*T))
    sig_4 = (-1/(2*(T**3)))*(s_dot_T-sig_1) + (1/(2*(T**2)))*sig_2

    line_s = Line(sig_4,sig_3,sig_2,sig_1,sig_0)

    delta_0 = d_0
    delta_1 = d_dot_0
    delta_2 = d_dot_dot_0/2
    delta_3 = (10/(T**3))*((-3/10)*delta_2*(T**2) - (3/5)*delta_1*T +(d_T-d_0))
    delta_4 = (1/(T**3))*(3*delta_2*T + 8*delta_1 - (15/T)*(d_T-delta_0))
    #delta_5 = (1/(10*(T**3)))*(-28*delta_2 - (30/(T))*delta_1 + (60/(T**2))*(d_T-delta_0))
    delta_5 = (-1/(T**3))*delta_2 - (3/(T**4))*delta_1 + (6/(T**5))*(d_T-delta_0)

    line_d = Line(delta_5,delta_4,delta_3,delta_2,delta_1,delta_0)
    ########################################################################

    return line_d,line_s


#############################################################################################################
#This is the specific case of the general derivation above. If the assumptions below are satisfied in the above
# case it should generate the same trajectories
def defineTrajectoryNonGeneral(init_state,dest_state,T):
    #Assumptions
    # Given: initial lateral and longitudinal (x,y) position; initial longitudinal velocity; final (lat,long) position,final longitudinal velocity
    # Assume initial lateral velocity is 0, final lateral velocity is 0
    # Assume initial acceleration (lat and long) is 0
    # Assume final acceleration (lat and long) is 0

    (d_0,s_0) = init_state["position"]
    s_dot_0 = init_state["velocity"] #we assume this is longitudinal velocity
    d_dot_0 = 0
    d_dot_dot_0,s_dot_dot_0 = 0,0 

    (d_T,_) = dest_state["position"]
    s_dot_T = dest_state["velocity"] #we assume this is longitudinal velocity
    d_dot_T = 0
    d_dot_dot_T,s_dot_dot_T = 0,0

    ########################################################################
    #All these derivations explicitly follow from the above assumptions. Cannot be genralised
    sig_0 = s_0
    sig_1 = s_dot_0
    sig_2 = s_dot_dot_0/2
    sig_3 = (s_dot_T-s_dot_0)/(T**2)
    sig_4 = (-1/(2*(T**3)))*(s_dot_T-s_dot_0)

    line_s = Line(sig_4,sig_3,sig_2,sig_1,sig_0)

    delta_0 = d_0
    delta_1 = d_dot_0
    delta_2 = d_dot_dot_0/2
    delta_3 = (10/(T**3))*(d_T-d_0)
    delta_4 = (-15/(T**4))*(d_T-d_0)
    delta_5 = (6/(T**5))*(d_T-d_0)

    line_d = Line(delta_5,delta_4,delta_3,delta_2,delta_1,delta_0)
    ########################################################################

    return line_d,line_s
##########################################################################################################

def putCarOnTraj(car,traj,time):
    posit = traj.position(time)
    v_x,v_y = traj.velocity(time)
    velocity = math.sqrt(v_x**2 + v_y**2)
    if v_x != 0:
        # - here to account for the fact that y-axis is inverted but angles are not
        heading = math.degrees(math.atan(-v_y/v_x))
        if v_x<0: heading += 180 #tan has domain [-90.90], which constrains output of atan to left half-domain
        heading%=360
    else:
        if v_y>0: heading = 270
        else: heading = 90

    car.setMotionParams(posit,heading,velocity)
    car.sense()


def evaluate(t,coefs):
    return sum([entry[0]*(t**entry[1]) for entry in coefs])
