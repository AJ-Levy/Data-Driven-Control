import numpy as np
import matplotlib.pyplot as plt

# PID Controller Class
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.previous_time = None
    
    def compute(self, current_value, time):
        dt = 0.02
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

# Simulation Environment
class InvertedPendulum:
    def __init__(self, angle=0, length=1, mass=1, gravity=9.81):
        self.angle = angle
        self.angular_velocity = 0
        self.length = length
        self.mass = mass
        self.gravity = gravity

    def update(self, force, dt):
        # Equation of motion for the inverted pendulum
        torque = force * self.length
        angular_acceleration = torque / (self.mass * self.length**2) - (self.gravity / self.length) * np.sin(self.angle)
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt
        return self.angle

# Control loop
def simulate_pendulum(pid, pendulum, time, dt):
    angles = []
    for t in np.arange(0, time, dt):
        force = pid.compute(pendulum.angle, dt)
        angle = pendulum.update(force, dt)
        angles.append(angle)
    return angles

# Parameters
Kp = 100
Ki = .1
Kd = 20
setpoint = 0
initial_angle = np.pi / 6  # 30 degrees
simulation_time = 10  # seconds
dt = 0.02  # time step

# Initialize PID controller and pendulum
pid = PIDController(Kp, Ki, Kd, setpoint)
pendulum = InvertedPendulum(angle=initial_angle)

# Run simulation
angles = simulate_pendulum(pid, pendulum, simulation_time, dt)

# Plot results
time = np.arange(0, simulation_time, dt)
plt.plot(time, angles)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Inverted Pendulum Angle vs Time')
plt.grid(True)
plt.show()
