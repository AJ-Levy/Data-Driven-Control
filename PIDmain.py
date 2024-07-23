import matlab.engine
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def genIntialAngle(delta=0.2):
    '''
    Generates a random intial angle about 0 radians.
    The angles lies in the range [-delta, delta) radians.
    '''
    return np.random.uniform(-delta, delta)

def main(model = 'pendSimPID',
         mask = 'Pendulum and Cart',
         stabilisation_precision = 0.05):

    # Remove old data
    if os.path.exists("anglePID.pkl"):
        os.remove("anglePID.pkl")
    if os.path.exists("timePID.pkl"):
        os.remove("timePID.pkl")
    
    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)
    # set random intial angle
    intial_angle = genIntialAngle()
    while intial_angle <= stabilisation_precision and intial_angle >= -stabilisation_precision:
        intial_angle = genIntialAngle()
    eng.set_param(f'{model}/{mask}', 'init', str(intial_angle), nargout=0)
    print("Running simulation...")
    eng.sim(model)
    print("Simulation complete")

    # Load angles
    angles = []
    with open('anglePID.pkl', 'rb') as f:
        try:
            while True:
                angles.append(pickle.load(f))
        except EOFError:
            pass
        
    # Load time
    time = []
    with open('timePID.pkl', 'rb') as f:
        try:
            while True:
                time.append(pickle.load(f))
        except EOFError:
            pass

    # Plot data
    plt.plot(time, angles, label = 'Output Signal')
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--', label=f'{stabilisation_precision} rad')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'-{stabilisation_precision} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.title("Angle of pendulum over time")
    plt.xlim(0,max(time))
    plt.ylim(-0.5,0.5)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

# # Should work but doesnt
# answer = eng.workspace['out.angle']
# print(answer)