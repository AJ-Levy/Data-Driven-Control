import numpy as np


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

    def update(self, rad):
        theta = rad*(180/np.pi)
        # Calculate the error
        error = self.setpoint - theta

        # Proportional error
        self.proportional_error = error

        # Integral error
        self.integral_error += error

        # Derivative error
        self.derivative_error = error - self.previous_error

        # Calculate the output force
        output_force = (self.Kp * self.proportional_error +
                        self.Ki * self.integral_error +
                        self.Kd * self.derivative_error)

        # Update the previous error
        self.previous_error = error

        return -output_force

def PIDRun(rad):
    PID = PIDController(Kp=0.2, Ki=0.1, Kd=0.05)
    force = PID.update(rad)
    return force

