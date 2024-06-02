class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.proportional_error = 0
        self.integral_error = 0
        self.derivative_error = 0
        self.previous_error = 0
        # dependant on sampling rate
        self.previous_time = None

    def update(self, rad, time):
        # calculate the time difference
        if self.previous_time is None:
            dt = 0
        else:
            dt = time - self.previous_time

        # Calculate the error
        error = self.setpoint - rad

        # Proportional error
        self.proportional_error = error

        # Integral error
        self.integral_error += error * dt

        # Derivative error
        if self.previous_time is not None:
            self.derivative_error = (error - self.previous_error) / dt
        else:
            self.derivative_error = 0

        # Calculate the output force
        output_force = (self.Kp * self.proportional_error +
                        self.Ki * self.integral_error +
                        self.Kd * self.derivative_error)

        # Update the previous error and time
        self.previous_error = error
        self.previous_time = time

        return output_force
    
# Global Instance of PID controller
pid_controller = PIDController(Kp=10, Ki=1, Kd=5)

def PIDRun(rad, time):
    global pid_controller
    force = pid_controller.update(rad, time)
    return force