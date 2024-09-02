import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

## Setting Font

import matplotlib as mpl
from matplotlib import font_manager

font_paths = ['/Users/ariellevy/Library/Fonts/LinLibertine_R.otf']  # Update with the path to your Libertine font file
for font_path in font_paths:
    font_manager.fontManager.addfont(font_path)

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Linux Libertine O']
mpl.rcParams['font.size'] = 14

##############################

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

'''
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
'''

'''
data_PID = read_data("PID.txt")
data_Q = read_data("qlearn.txt")

angle_PID, time_PID = data_PID[:, 0], data_PID[:,1]
angle_Q, time_Q = data_Q[:, 0], data_Q[:,1]

with open("angles.txt", 'r') as file:
    angle_NEAT = file.readlines()
angle_NEAT = [float(line.strip()) for line in angle_NEAT]

with open("time.txt", 'r') as file:
    time_NEAT = file.readlines()
time_NEAT = [float(line.strip()) for line in time_NEAT]

plt.figure(figsize=(8,6))
plt.plot(time_PID, angle_PID, label = "PID")
plt.plot(time_Q, angle_Q, label = "Q-Learning")
plt.plot(time_NEAT, angle_NEAT, label = "NEAT")
'''
'''
angle_noise_2, time_noise_2 = data_noise_2[:, 0], data_noise_2[:,1]
angle_noise_3, time_noise_3 = data_noise_3[:, 0], data_noise_2[:,1]
angle_noise_4, time_noise_4 = data_noise_4[:, 0], data_noise_4[:,1]
angle_noise_5, time_noise_5 = data_noise_5[:, 0], data_noise_5[:,1]

# Mean Deviaiton due to noise and SNR
max_dists = []
P_signal = 0
n = len(time_no_noise)
P_noise = 0
for i in range(n):
    # deviaiton 
    d1 = (angle_no_noise[i] - angle_noise_1[i])**2
    d2 = (angle_no_noise[i] - angle_noise_2[i])**2
    d3 = (angle_no_noise[i] - angle_noise_3[i])**2
    d4 = (angle_no_noise[i] - angle_noise_4[i])**2
    d5 = (angle_no_noise[i] - angle_noise_5[i])**2
    max_dists.append(np.sqrt(max(d1,d2,d3,d4,d5)))

print(f"Distance from signal: {np.mean(max_dists):.3f} +/- {np.std(max_dists):.4f}")

plt.plot(time_noise_1, angle_noise_1, color = "grey", label = f"Noise", alpha=0.7)
plt.plot(time_noise_2, angle_noise_2, color = "grey", alpha=0.7)
plt.plot(time_noise_3, angle_noise_3, color = "grey", alpha=0.7)
plt.plot(time_noise_4, angle_noise_4, color = "grey", alpha=0.7)
plt.plot(time_noise_5, angle_noise_5, color = "grey", alpha=0.7)
plt.plot(time_no_noise, angle_no_noise, color = "blue", label = f"Ideal")
'''

'''
stabilisation_precision = 0.05
plt.axhline(y=stabilisation_precision, color='k', linestyle='--')
plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.2f} rad')
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.xlim(0,0.5)
plt.ylim(-1, 1)
#plt.title("Angle of pendulum over time")
plt.legend()
plt.savefig('plot.pdf', format='pdf')
plt.show()
'''

data = {
    'PID': {'mean': 0.28, 'std_dev': 0.35, 'min_val': 0.02, 'max_val': 0.78},
    'Q-Learning': {'mean': 0.36, 'std_dev': 15, 'min_val': 40, 'max_val': 90},
    'NEAT': {'mean': 55, 'std_dev': 12, 'min_val': 35, 'max_val': 75},
}


means = np.array([26.1, 3.9, 0.0])
mins = np.array([25.5, 3.4, 0.0])
maxs = np.array([26.7, 4.7, 0.0])
stds = np.array([0.4, 0.5, 0.0])

x = np.array([1,2,3])

plt.errorbar(x, means, yerr=stds, fmt='o', color='darkorange', ecolor = "orange", label='Mean ± SD')

# Plot the min and max values as filled areas
for i in range(len(x)):
    plt.fill_between([x[i] - 0.01, x[i] + 0.01], mins[i], maxs[i], color='lightgrey', alpha=0.5)

# Plot min and max values
plt.scatter(x, maxs, color='teal', marker='o', label='Max.', zorder=5)
plt.scatter(x, mins, color='purple', marker='o', label='Min.', zorder=5)

for i in range(len(x)):
    offset = 0.1  # Offset to move text to the right
    # Annotate mean
    plt.text(x[i] + offset, means[i], f'{means[i]:.2f} ± {stds[i]:.2f}', ha='left', va='center', color='k', fontsize=12)
    
    # Annotate min
    plt.text(x[i] + offset, mins[i], f'{mins[i]:.2f}', ha='left', va='center', color='k', fontsize=12)
    
    # Annotate max
    plt.text(x[i] + offset, maxs[i], f'{maxs[i]:.2f}', ha='left', va='center', color='k', fontsize=12)

plt.ylim(0,30)
plt.ylabel("Stabilisation Time (ms)")
plt.xticks([0, 1, 2, 3, 4], ['','PID', 'Q-Learning', 'NEAT', ''])
plt.xlabel("Control Methods")
plt.legend()
plt.savefig("plot.pdf", format="pdf")


# Show the plot
plt.show()
