class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral_error = 0
        self.previous_error = 0

    def update(self, rad, dt):
        # Calculate the error
        error = self.setpoint - rad

        # Integral error
        self.integral_error += error * dt

        # Derivative error
        derivative_error = (error - self.previous_error) / dt

        # Calculate the output force
        output_force = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)

        # Update the previous error and time
        self.previous_error = error

        return output_force
    
# Global Instance of PID controller
pid_controller = PIDController(Kp=10, Ki=1, Kd=5)

def PIDRun(rad, dt):
    # dt depends on sampling rate
    global pid_controller
    force = pid_controller.update(rad, dt)
    return force