import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

def setNoise(eng, model, noise):
    '''
    Sets appropriate amount of noise if required
    '''
    if noise:
        eng.set_param(f'{model}/Noise', 'Cov', str([0.0000008]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)

def main(noise = False,
         desired_voltage = 30,
         source_voltage = 48,
         model = 'bcSimPID',
         stabilisation_precision = 0.5):
    '''
    Main method to set up MATLAB, simulink,
    and handle data aquisition/plotting.
    '''
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()

    eng.load_system(model, nargout=0)
    
    setNoise(eng, model, noise)
    eng.set_param(f'{model}/finalVoltage', 'Value', str(desired_voltage), nargout=0)
        
    # Set initial voltage
    eng.set_param(f'{model}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)

    print("Running simulation...")
    eng.eval(f"out = sim('{model}');", nargout=0)
    print("Simulation complete")

    # Get voltages
    voltage_2d = eng.eval("voltage")
    voltage_lst = []
    for v in voltage_2d:
        voltage_lst.append(v[0])

    # Get time
    time_2d = eng.eval("time")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0])

    plt.plot(time_lst, voltage_lst, label = f'{desired_voltage} V')
    
    eng.quit()

    plt.axhline(y=stabilisation_precision + (desired_voltage), color='k', linestyle='--', label=f'{stabilisation_precision + (desired_voltage)} V')
    plt.axhline(y=-stabilisation_precision + (desired_voltage), color='k', linestyle='--', label=f'{-stabilisation_precision + (desired_voltage)} V')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    #plt.title("Voltage Progression over Time")
    plt.xlim(0,max(time_lst))
    #view_range = 0.2*desired_voltage
    #plt.ylim(desired_voltage - view_range,desired_voltage + view_range)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(noise=True)