class PIDController:
    '''
    Class to store values for the PID Controller,
    and to determine output signal. 
    '''
    def __init__(self, Kp, Ki, Kd, dt=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0
        self.previous_error = 0
        self.previous_time = None
        self.dt = None
        self.scale_factor = 1/350000

    def update(self, voltage, time):
        '''
        Determines the output force according to PID.
        '''
        # Proportional error
        error = voltage

        if self.previous_time is None:
            self.previous_time = time
            # return intial duty cycle of 0.5
            return 0.5

        # Calculate the time difference
        self.dt = time - self.previous_time

        # Integral error
        self.integral_error += error * self.dt

        # Derivative error
        derivative_error = (error - self.previous_error) / self.dt

        # Calculate the output
        output = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)

        # Update variables
        self.previous_error = error
        self.previous_time = time

        return output * self.scale_factor

# Create PID object
pid_controller = PIDController(Kp=100, Ki=10, Kd=1)

def controller_call(voltage, time):
    '''
    Method that MATLAB calls for PID.
    '''
    signal = pid_controller.update(voltage, time)
    return signal
