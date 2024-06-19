import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.previous_error = 0
        self.time = 0 # nice to have

    def update(self, rad_big, dt):

        # Normalize the angle (-2pi to 2pi)
        rad = (rad_big%(np.sign(rad_big)*2*np.pi))

        # # Additional correction stuff for spinning pend
        # if (rad > np.pi):
        #     rad = -(2*np.pi-rad)
        # elif (rad < -np.pi):
        #     rad = 2*np.pi+rad
        # if (-np.pi/2 < rad < np.pi/2):
        # else:
        #     output_force = 0
        
        # Proportional error
        error = rad

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

        return output_force
    
# Global Instance of PID controller
pid_controller = PIDController(Kp=40, Ki=10, Kd=0.8)

def PIDRun(rad, dt):
    global pid_controller
    force = pid_controller.update(rad, dt)
    return force