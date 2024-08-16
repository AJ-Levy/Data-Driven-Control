import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

def check_stabilization(signal, margin=0.05, required_iterations=2000):
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

def setNoise(eng, model, noise):
    '''
    Sets appropriate amount of noise if required
    '''
    if noise:
        eng.set_param(f'{model}/Noise', 'Cov', str([0.00001]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)

def main(noise = False,
         model = 'pendSimPID',
         mask = 'Pendulum and Cart',
         stabilisation_precision = 0.05,
         delta = np.pi/3):
    '''
    Main method to set up MATLAB, simulink,
    and handle data aquisition/plotting.
    '''
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()

    eng.load_system(model, nargout=0)


    setNoise(eng, model, noise)

    # Set random initial angle
    eng.set_param(f'{model}/{mask}', 'init', str(ang), nargout=0)

    print("Running simulation...")
    eng.eval(f"out = sim('{model}');", nargout=0)
    print("Simulation complete")

    # Get angles
    angle_2d = eng.eval("out.angle")
    angle_lst = []
    for angle in angle_2d:
        angle_lst.append(angle[0])

    # Get time
    time_2d = eng.eval("out.time")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0])

    stabilises, index = check_stabilization(angle_lst)
    print(f"{ang} deg: stabilises at {time_lst[index]} s")
    plt.plot(time_lst, angle_lst, label = f"{ang} deg")
    
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--', label=f'{stabilisation_precision} rad')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'-{stabilisation_precision} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.title("Angle of pendulum over time")
    plt.xlim(0,max(time_lst))
    plt.ylim(-np.pi/2,np.pi/2)
    plt.legend()
    plt.show()

    eng.quit()

if __name__ == '__main__':
    main(noise=False)