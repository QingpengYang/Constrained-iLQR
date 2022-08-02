import numpy as np
import pdb

# Add lambda functions
cos = lambda a : np.cos(a)
sin = lambda a : np.sin(a)
tan = lambda a : np.tan(a)

class Model:
    """
    A vehicle model with 4 dof. 
    State - [x, y, vel, theta]
    Control - [acc, yaw_rate]
    """
    def __init__(self, args):
        self.wheelbase = args.wheelbase
        self.steer_min = args.steer_angle_limits[0]
        self.steer_max = args.steer_angle_limits[1]
        self.accel_min = args.acc_limits[0]
        self.accel_max = args.acc_limits[1]
        self.max_speed = args.max_speed
        self.const_speed = args.const_speed
        self.tractor_l = args.tractor_l
        self.trailer_d = args.trailer_d
        self.Ts = args.timestep
        self.N = args.horizon
        self.z = np.zeros((self.N))
        self.o = np.ones((self.N))
        
    def forward_simulate(self, state, control):
        """
        Find the next state of the vehicle given the current state and control input
        """
        # Clips the controller values between min and max accel and steer values
        control = np.clip(control, self.steer_min, self.steer_max)
        # control[1] = np.clip(control[1], state[2]*tan(self.steer_min)/self.wheelbase, state[2]*tan(self.steer_max)/self.wheelbase)
        next_state = np.array([state[0] + self.const_speed * cos(state[2]) * self.Ts,
                               state[1] + self.const_speed * sin(state[2]) * self.Ts,
                               state[2] + self.const_speed / self.tractor_l * tan(control[0]) * self.Ts,
                               state[3] + self.const_speed / self.trailer_d * sin(state[3] - state[2]) * self.Ts]) 

        # next_state = np.array([state[0] + cos(state[3])*(state[2]*self.Ts + (control[0]*self.Ts**2)/2),
        #                        state[1] + sin(state[3])*(state[2]*self.Ts + (control[0]*self.Ts**2)/2),
        #                        np.clip(state[2] + control[0]*self.Ts, 0.0, self.max_speed),
        #                        state[3] + control[1]*self.Ts])  # wrap angles between 0 and 2*pi - Gave me error
        return next_state

    def get_A_matrix(self, velocity_vals, theta, acceleration_vals):
        """
        Returns the linearized 'A' matrix of the ego vehicle 
        model for all states in backward pass. 
        """
        v = velocity_vals
        v_dot = acceleration_vals
          
        A = np.array([[self.o, self.z, cos(theta)*self.Ts, -(v*self.Ts + (v_dot*self.Ts**2)/2)*sin(theta)],
                      [self.z, self.o, sin(theta)*self.Ts,  (v*self.Ts + (v_dot*self.Ts**2)/2)*cos(theta)],
                      [self.z, self.z,             self.o,                                         self.z],
                      [self.z, self.z,             self.z,                                         self.o]])
        return A
    
    def get_A_matrix_1(self, theta, hitch):
        """
        Returns the linearized 'A' matrix of the ego vehicle 
        model for all states in backward pass. 
        """

        A = np.array([[self.o, self.z,                      -sin(theta)*self.const_speed*self.Ts,                                                          self.z],
                      [self.z, self.o,                       cos(theta)*self.const_speed*self.Ts,                                                          self.z],
                      [self.z, self.z,                                                    self.o,                                                          self.z],
                      [self.z, self.z, -self.const_speed/self.trailer_d*self.Ts*cos(hitch-theta), self.o+self.const_speed/self.trailer_d*self.Ts*cos(hitch-theta)]])

        return A

    def get_B_matrix(self, theta):
        """
        Returns the linearized 'B' matrix of the ego vehicle 
        model for all states in backward pass. 
        """
        B = np.array([[self.Ts**2*cos(theta)/2,         self.z],
                      [self.Ts**2*sin(theta)/2,         self.z],
                      [         self.Ts*self.o,         self.z],
                      [                 self.z, self.Ts*self.o]])
        return B
    
    def get_B_matrix_1(self, steer):
        """
        Returns the linearized 'B' matrix of the ego vehicle 
        model for all states in backward pass. 
        """
        B = np.array([[                                                 self.z],
                      [                                                 self.z],
                      [self.Ts*self.const_speed/self.tractor_l/(cos(steer)**2)],
                      [                                                 self.z]])
        return B
