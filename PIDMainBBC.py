import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

def main(desired_voltage = 30,
         source_voltage = 48,
         model = 'pibuckconverter',
         block = 'input_voltage',
         stabilisation_precision = 0.5):
    '''
    Main method to set up MATLAB, simulink,
    and handle data aquisition/plotting.
    '''
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)

    # Set desired output voltage
    eng.set_param(f'{model}/finalVoltage', 'Value', str(desired_voltage), nargout=0)
    
    # Set initial voltage
    eng.set_param(f'{model}/{block}', 'Amplitude', str(source_voltage), nargout=0)

    print("Running simulation...")
    eng.eval(f"out = sim('{model}');", nargout=0)
    print("Simulation complete")

    # Get force data
    # Get angles
    pulse_2d = eng.eval("pulse")
    pulse_lst = []
    for p in pulse_2d:
        pulse_lst.append(p[0])
    print(f"Average Duty Cycle: {np.median(pulse_lst):.3f}")

    # Get angles
    voltage_2d = eng.eval("Vout")
    voltage_lst = []
    for v in voltage_2d:
        voltage_lst.append(v[0])

    # Get time
    time_2d = eng.eval("time")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0])

    # Plot data
    plt.plot(time_lst, voltage_lst, label = 'Output Signal', color = "red")
    plt.axhline(y=stabilisation_precision + desired_voltage, color='k', linestyle='--', label=f'{stabilisation_precision + desired_voltage} V')
    plt.axhline(y=-stabilisation_precision + desired_voltage, color='k', linestyle='--', label=f'{-stabilisation_precision + desired_voltage} V')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage Progression over Time")
    plt.xlim(0,max(time_lst))
    view_range = 0.2*desired_voltage
    plt.ylim(desired_voltage - view_range,desired_voltage + view_range)
    plt.legend()
    plt.show()

    eng.quit()

if __name__ == '__main__':
    main()