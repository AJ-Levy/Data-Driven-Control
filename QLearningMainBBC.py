import matlab.engine
import matplotlib.pyplot as plt
import os
import numpy as np
import time

def viewTable(qtable_file='qtable_BBC.npy'):
    ''' 
    View QTable as an array
    '''
    qtable = np.load(qtable_file)

    # Print the Q-table
    print("Q-table:")
    print(qtable)

# reset Q table
def reset(num_states=60, num_actions=2, qtable_file='qtable_BBC.npy'):
    '''
    Reset QTable to 2D array of zeros of size
    num_states x num_actions.
    '''
    qtable = np.zeros((num_states, num_actions))
    np.save(qtable_file, qtable)

def train(eng, model, mask, convergence_data_file, num_episodes=1000, count=0):
    '''
    Train QLearning Agent
    '''
    # clean up algorithm convergence file
    if os.path.exists(convergence_data_file):
        os.remove(convergence_data_file)

    reset()
    for episode in range(1,num_episodes+1):
        # pass in current episode
        eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)
        # pass in initial parameters (source voltage and reference voltage)
        source_voltage = 48
        desired_voltage = 30
        eng.set_param(f'{model}/finalVoltage', 'Value', str(desired_voltage), nargout=0)
        eng.set_param(f'{model}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)

        eng.sim(model)
        if episode % (num_episodes//10) == 0:
            count += 1 
            print(f"{count*10}%")

def setNoise(eng, model, noise):
    if noise:
        eng.set_param(f'{model}/Noise', 'Cov', str([0.00001]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)

def main(trainModel = True, 
         noise = False,
         trainingModel = 'bbcSimQTraining', 
         controllerModel = 'bbcSimQController',
         mask = 'BBC',
         convergence_data_file = 'qconverge_BBC.txt',
         stabilisation_precision = 0.05):

    global time

    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel, nargout=0)
    start_time = time.time()
    ## Comment Out Once Model is Trained ##
    if trainModel:
        print("Training model...")
        train(eng, trainingModel, mask, convergence_data_file)
    #######################################
    print("Running simulation...")
    eng.load_system(controllerModel, nargout=0)
    setNoise(eng, controllerModel, noise)
   
    ## Data Presentation
    # Get voltages
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
    plt.plot(time_lst, voltage_lst, label = "Output Signal")
        
    # configure plot
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.xlim(0,max(time_lst))
    plt.title("Voltage Progression over time")
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

    plt.plot(episodes, rewards, color = 'r')
    plt.plot()
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward per Time Step')
    plt.title('Convergence of Q-learning')
    plt.show()

    eng.quit()

if __name__ == '__main__':
    main(trainModel=False, noise=True)