class pendPIDcontroller:
    '''
    Class to store values for the PID Controller,
    and to determine output pulse. 
    '''
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0
        self.previous_error = 0

    def update(self, voltage):
        '''
        Determines the output pulse accoring to PID.
        '''
        # Proportional error
        error = voltage

        # Integral error
        self.integral_error += error * self.dt

        # Derivative error
        derivative_error = (error - self.previous_error) / self.dt

        # Calculate the output pulss
        output_pulse = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)

        # Update variables
        self.previous_error = error

        return output_pulse

# Create PID object
pid_controller = pendPIDcontroller(Kp=40, Ki=5, Kd=10, dt=0.001)

def controller_call(angle):
    '''
    Method that MATLAB calls for PID.
    '''
    pulse = pid_controller.update(angle)
    return pulse