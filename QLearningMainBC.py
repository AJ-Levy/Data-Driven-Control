import matlab.engine
import matplotlib.pyplot as plt
import os
import numpy as np
import time

def viewTable(qtable_file='qtable_BC.npy'):
    ''' 
    View QTable as an array
    '''
    qtable = np.load(qtable_file)

    # Print the Q-table
    print("Q-table:")
    print(qtable)

# reset Q table
def reset(num_states=24, num_actions=5, qtable_file='qtable_BC.npy'):
    '''
    Reset QTable to 2D array of zeros of size
    num_states x num_actions.
    '''
    qtable = np.zeros((num_states, num_actions))
    np.save(qtable_file, qtable)

def select_reference_voltage(v_ref, variation = 0.05):
    '''
    Randomly returns one of the reference voltage
    options for training.
    '''
    variation_range = v_ref * variation
    return v_ref + np.random.uniform(-variation_range, variation_range)

def train(eng, model, mask, convergence_data_file, source_voltage, v_ref, num_episodes=1250, count=0):
    '''
    Train QLearning Agent
    '''
    # clean up algorithm convergence file
    if os.path.exists(convergence_data_file):
        os.remove(convergence_data_file)

    reset()
    for episode in range(1, num_episodes+1):
        # pass in current episode
        eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)
        # pass in initial parameters (source voltage and reference voltage)
        eng.set_param(f'{model}/finalVoltage', 'Value', str(select_reference_voltage(v_ref)), nargout=0)
        eng.set_param(f'{model}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)

        eng.sim(model)
        if episode % (num_episodes//10) == 0:
            count += 1 
            print(f"{count*10}%")

def setNoise(eng, model, noise):
    if noise:
        eng.set_param(f'{model}/Noise', 'Cov', str([0.0000004]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)

def main(trainModel = True, 
         noise = False,
         trainingModel = 'bcSimQTraining', 
         controllerModel = 'bcSimQController',
         mask = 'BBC',
         convergence_data_file = 'qconverge_BC.txt',
         stabilisation_precision = 0.5,
         source_voltage = 48,
         desired_voltage = 30):

    global time

    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel, nargout=0)
    start_time = time.time()
    if trainModel:
        print("Training model...")
        train(eng, trainingModel, mask, convergence_data_file, source_voltage, v_ref)
    print("Running simulation...")
    eng.load_system(controllerModel, nargout=0)
    
    viewTable()
    
    setNoise(eng, controllerModel, noise)
    # pass in initial parameters (source voltage and reference voltage)
    eng.set_param(f'{controllerModel}/finalVoltage', 'Value', str(desired_voltage), nargout=0)
    eng.set_param(f'{controllerModel}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)
    eng.eval(f"out = sim('{controllerModel}');", nargout=0)

    ## Data Presentation
    # Get voltages
    voltage_2d = eng.eval("voltage")
    voltage_lst = []
    for v in voltage_2d:
        voltage_lst.append(v[0])

    # Get time
    time_2d = eng.eval("time")
    time_lst = []
    for t in time_2d:
        time_lst.append(t[0])
    
    plt.plot(time_lst, voltage_lst, label = f"Ref: {desired_voltage} V")
        
    # configure plot
    plt.axhline(y=desired_voltage + stabilisation_precision, color='k', linestyle='--')
    plt.axhline(y=desired_voltage - stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.1f} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.xlim(0,0.1)
    plt.legend(loc = 'upper right')
    plt.show()

    duration = time.time() - start_time
    print(f"Simulation complete in {duration:.1f} secs")

    # Load convergence data
    episodes = []
    rewards = []
    with open(convergence_data_file, 'r') as f:
        for line in f:
            ep, reward = line.strip().split('#')
            episodes.append(float(ep))
            rewards.append(float(reward))

    plt.plot(episodes, rewards, color = 'darkorange')
    plt.plot()
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.savefig('buckQConvergence.pdf', format = "pdf")
    plt.show()
    

    eng.quit()

if __name__ == '__main__':
    main(trainModel=True, noise=False)