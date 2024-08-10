import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

def main(model = 'pendSimPID',
         mask = 'Pendulum and Cart',
         stabilisation_precision = 0.05,
         delta = 0.5):
    '''
    Main method to set up MATLAB, simulink,
    and handle data aquisition/plotting.
    '''
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)
    
    # Set random intial angle
    intial_angle = np.random.uniform(-delta, delta)
    while intial_angle <= stabilisation_precision and intial_angle >= -stabilisation_precision:
        intial_angle = np.random.uniform(-delta, delta)
    eng.set_param(f'{model}/{mask}', 'init', str(intial_angle), nargout=0)

    # Set random noise seed
    noise_seed = str(np.random.randint(1, high=40000))
    eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)

    print("Running simulation...")
    eng.eval("out = sim('pendSimPID.slx');", nargout=0)
    print("Simulation complete")

    # Get angles
    angle_2d = eng.eval("out.angle")
    angle_lst = []
    for angle in angle_2d:
        angle_lst.append(angle[0])

    # Get time
    time_2d = eng.eval("out.tout")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0])

    # Plot data
    plt.plot(time_lst, angle_lst, label = 'Output Signal')
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--', label=f'{stabilisation_precision} rad')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'-{stabilisation_precision} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.title("Angle of pendulum over time")
    plt.xlim(0,max(time_lst))
    plt.ylim(-0.5,0.5)
    plt.legend()
    plt.show()

    eng.quit()

if __name__ == '__main__':
    main()