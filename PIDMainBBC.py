import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

def main(desired_voltage = 70,
         source_voltage = 50,
         model = 'bbcSimPID',
         mask = 'BBC',
         block = 'V_source_value',
         stabilisation_precision = 0.05):
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
    #eng.set_param(f'{model}/{mask}/{block}', 'Amplitude', str(source_voltage), nargout=0)

    print("Running simulation...")
    eng.eval(f"out = sim('{model}');", nargout=0)
    print("Simulation complete")

    # Get angles
    voltage_2d = eng.eval("out.Vout")
    voltage_lst = []
    for v in voltage_2d:
        voltage_lst.append(v[0])

    # Get time
    time_2d = eng.eval("out.time")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0])

    # Plot data
    plt.plot(time_lst, voltage_lst, label = 'Output Signal')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage Progression over Time")
    plt.xlim(0,max(time_lst))
    plt.legend()
    plt.show()
    print(voltage_lst)

    eng.quit()

if __name__ == '__main__':
    main()