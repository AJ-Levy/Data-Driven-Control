import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

# Load convergence data
controllerModel = 'pendSimQController'
cartPoleSubsystem = 'Pendulum and Cart'
eng = matlab.engine.start_matlab()
eng.load_system(controllerModel, nargout=0)
#count = 10


def check_stabilization(signal, margin=0.04, required_iterations=1000):
    """
    Check if the signal has stabilized around zero.
    
    Args:
    signal (array-like): The signal data to check.
    margin (float): The margin of error around zero to consider as stabilized.
    required_iterations (int): The number of consecutive iterations within margin to consider as stabilized.
    
    Returns:
    bool: True if the signal is stabilized, False otherwise.
    int: The index at which stabilization starts, or -1 if not stabilized.
    """
    signal = np.array(signal)
    n = len(signal)
    
    if required_iterations > n:
        return False, -1
    
    for i in range(n - required_iterations + 1):
        # Check if all values in the range are within the margin
        if np.all(np.abs(signal[i:i + required_iterations]) < margin):
            return True, i
    
    return False, -1

'''
for angle_it in range(-55, 65, 10):
    ang = np.deg2rad(angle_it)
    eng.set_param(f'{controllerModel}/Noise', 'Cov', str([0]), nargout=0)
    eng.set_param(f'{controllerModel}/Noise_v', 'Cov', str([0]), nargout=0)
    eng.set_param(f'{controllerModel}/{cartPoleSubsystem}', 'init', str(ang), nargout=0)
    eng.eval(f"out = sim('{controllerModel}');", nargout=0)

    # Get angles
    angle_2d = eng.eval("out.angle")
    angle_lst = []
    for a in angle_2d:
        angle_lst.append(a[0])

    # Get time
    time_2d = eng.eval("out.time")
    time_lst = []
    for t in time_2d:
        time_lst.append(t[0])

    with open(f"{angle_it}_{count}.txt", "w") as f:
        for i in range(len(time_lst)):
            f.write(f"{angle_lst[i]}#{time_lst[i]}\n")

'''
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    # Split each line at '#' and convert to float
    data = [line.strip().split('#') for line in data]
    data = [(float(x), float(y)) for x, y in data]
    return np.array(data)

# Read data from files
plt.figure()
for angle_it in range(-55, 65, 10):
    ang = np.deg2rad(angle_it)
    angle_array = []
    time_array = []
    for count in range(1,11):
        data = read_data(f"{angle_it}_{count}.txt")
        angle_array.append(data[:, 0])
        time_array.append(data[:, 1])

    angle_avg = np.mean(angle_array, axis=0)
    print(angle_avg)
    angle_unc = np.sqrt(np.var(angle_array, axis=0))
    time_avg = np.mean(time_array, axis=0)

    stabilises, index = check_stabilization(angle_avg)
    if stabilises:
        print(f"{angle_it} degrees: {time_avg[index]} s")
    else:
        print(f"{angle_it} degrees doesn't stabilise")

    plt.plot(time_array[0], angle_avg, label = f"Initial Angle: {ang:.2f} rad", color = "purple")
    stabilisation_precision = 0.05
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.2f} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("θ (rad)")
    plt.xlim(0,5)
    plt.ylim(-np.pi/2, np.pi/2)
    #plt.title("Angle of pendulum over time")
    plt.legend(loc = 'upper right')
    plt.show()


stabilisation_precision = 0.05
plt.axhline(y=stabilisation_precision, color='k', linestyle='--')
plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.2f} rad')
plt.xlabel("Time (s)")
plt.ylabel("θ (rad)")
plt.xlim(0,5)
plt.ylim(-np.pi/2, np.pi/2)
#plt.title("Angle of pendulum over time")
plt.legend(loc = 'upper right')
plt.show()

stab_times = [0.676, 0.532, 0.38, 0.222, 0.166, 0.285, 0.131, 0.39, 0.46]
print(f"Min Time: {min(stab_times):.2f} s")
print(f"Avg Time: {np.mean(stab_times):.2f} +/- {np.sqrt(np.var(stab_times)):.2f} s")
print(f"Max Time: {max(stab_times):.2f}:.2f s")
