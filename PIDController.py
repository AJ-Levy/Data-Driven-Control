class PIDController:
    '''
    Class to store values for the PID Controller,
    and to determine output force. 
    '''
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0
        self.previous_error = 0

    def update(self, theta):
        '''
        Determines the output force accoring to PID.
        '''
        # Proportional error
        error = theta

        # Integral error
        self.integral_error += error * self.dt

        # Derivative error
        derivative_error = (error - self.previous_error) / self.dt

        # Calculate the output force
        output_force = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)

        # Update variables
        self.previous_error = error

        return output_force

# Create PID object
pid_controller = PIDController(Kp=40, Ki=5, Kd=10, dt=0.005)

def controller_call(angle):
    '''
    Method that MATLAB calls for PID.
    '''
    force = pid_controller.update(angle)
    return force