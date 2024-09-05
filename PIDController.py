class PIDController:
    '''
    A class that implements simple a Proportional-Integral-Derivative (PID) controller.
    
    Attributes:
        Kp (float): Proportional gain coefficient.
        Ki (float): Integral gain coefficient.
        Kd (float): Derivative gain coefficient.
        dt (float): Step size.
        integral_error (float): Accumulated integral of the error.
        previous_error (float): Error value from the previous update.
    '''

    def __init__(self, Kp, Ki, Kd, dt):
        '''
        Initialises the PID controller with the specified gain coefficients and step siz.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            dt (float): Step size.
        '''
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0
        self.previous_error = 0

    def update(self, setpoint, measurement):
        '''
        Computes the control output using the PID control equation.

        Args:
            setpoint (float): Desired value.
            measurement (float): Current value.

        Returns:
            float: Output force.
        '''

        error = measurement - setpoint

        self.integral_error += error * self.dt

        derivative_error = (error - self.previous_error) / self.dt

        output_force = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)

        self.previous_error = error

        return output_force

# Instantiate PIDController object.
pid_controller = PIDController(Kp=44, Ki=80, Kd=6, dt=0.001)

def controller_call(setpoint, measurement):
    '''
    Calls the PID controller to compute the control signal.

    Args:
        setpoint (float): Desired value.
        measurement (float): Current value.

    Returns:
        float: Output force.
    '''
    force = pid_controller.update(setpoint, measurement)
    return force