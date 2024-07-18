import numpy as np
import pickle

class PIDController:
    '''
    Class to store values for the PID Controller,
    and to determine output force. 
    '''
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.previous_error = 0
        self.time = 0

    def update(self, theta_big, dt):
        '''
        Determines the output force accoring to PID
        '''
        # # Normalize the angle (-2pi to 2pi)
        # theta = (theta_big%(np.sign(theta_big)*2*np.pi))
        theta = theta_big
        
        # Proportional error
        error = theta

        # Integral error
        self.integral_error += error * dt

        # Derivative error
        derivative_error = (error - self.previous_error) / dt

        # Calculate the output force
        output_force = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)

        # Update variables
        self.previous_error = error
        self.time += dt

        # Write data to files
        with open('anglePID.pkl', 'ab') as f:
            pickle.dump(theta, f)

        with open('timePID.pkl', 'ab') as f:
            pickle.dump(self.time, f)  

        return output_force

# Create PID object
pid_controller = PIDController(Kp=40, Ki=10, Kd=0.8)

def controller_call(angle, dt):
    '''
    Method that MATLAB calls for PID
    '''
    force = pid_controller.update(angle, dt)
    return force

# # Additional correction stuff for spinning pend
# if (theta > np.pi):
#     theta = -(2*np.pi-theta)
# elif (theta < -np.pi):
#     theta = 2*np.pi+theta
# if (-np.pi/2 < theta < np.pi/2):
# else:
#     output_force = 0