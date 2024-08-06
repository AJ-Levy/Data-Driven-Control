class PIDController:
    '''
    Class to store values for the PID Controller,
    and to determine output signal. 
    '''
    def __init__(self, Kp, Ki, Kd, dt=None, saturation_min = 0.1, saturation_max = 0.9):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0.1
        self.previous_error = 0
        self.previous_time = None
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max
        self.in_saturation = False

    def saturation(self, out, saturation_min, saturation_max):
        '''
        Ensures the output remains between the saturation limits.
        '''
        if out > saturation_max:
            self.in_saturation = True
            return saturation_max
        elif out < saturation_min:
            self.in_saturation = True
            return saturation_min
        else:
            self.in_saturation = False
        return out

    def update(self, voltage, time):
        '''
        Determines the output force according to PID.
        '''
        # Proportional error
        error = voltage

        if self.previous_time is None:
            self.previous_time = time
            # return intial duty cycle
            intial_output = self.Kp * error + self.Ki * self.integral_error
            return self.saturation(intial_output, self.saturation_min, self.saturation_max)

        # Calculate the time difference
        self.dt = time - self.previous_time

        # integral error (anti-windup measures included)
        if not (self.in_saturation):
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

        saturated_output = self.saturation(output, self.saturation_min, self.saturation_max)
        
        return saturated_output

# Create PID object
pid_controller = PIDController(Kp=0.01, Ki=5, Kd=0)

def controller_call(voltage, time):
    '''
    Method that MATLAB calls for PID.
    '''
    global pid_controller
    signal = pid_controller.update(voltage, time)
    return signal
