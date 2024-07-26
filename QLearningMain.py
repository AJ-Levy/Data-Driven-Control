import matlab.engine
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import time

def viewTable(qtable_file='qtable.npy'):
    ''' 
    View QTable as an array
    '''
    qtable = np.load(qtable_file)

    # Print the Q-table
    print("Q-table:")
    print(qtable)

# reset Q table
def reset(num_states=18, num_actions=2, qtable_file='qtable.npy'):
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
        intial_angle = genIntialAngle()
        # pass in current episode
        eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)
        # pass in random intial angular offset
        eng.set_param(f'{model}/{mask}', 'init', str(intial_angle), nargout=0)
        eng.sim(model)
        if episode % (num_episodes//10) == 0:
            count += 1 
            print(f"{count*10}%")

def genIntialAngle(delta=0.2):
    '''
    Generates a random intial angle about 0 radians.
    The angles lies in the range [-delta, delta) radians.
    '''
    return np.random.uniform(-delta, delta)

def main(trainModel = True, 
         trainingModel = 'pendSimQTraining', 
         controllerModel = 'pendSimQController',
         cartPoleSubsystem = 'Pendulum and Cart',
         angle_data_file = 'angleQ.pkl',
         time_data_file = 'timeQ.pkl',
         convergence_data_file = 'qconverge.txt',
         stabilisation_precision = 0.05):

    global time

    # clean up old data
    if os.path.exists(angle_data_file):
        os.remove(angle_data_file)
    if os.path.exists(time_data_file):
        os.remove(time_data_file)
    
    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel, nargout=0)
    start_time = time.time()
    ## Comment Out Once Model is Trained ##
    if trainModel:
        print("Training model...")
        train(eng, trainingModel, cartPoleSubsystem, convergence_data_file)
    #######################################
    print("Running simulation...")
    eng.load_system(controllerModel, nargout=0)
    intial_angle = genIntialAngle()
    while intial_angle <= stabilisation_precision and intial_angle >= -stabilisation_precision:
        intial_angle = genIntialAngle()
    # pass in random intial angular offset
    eng.set_param(f'{controllerModel}/{cartPoleSubsystem}', 'init', str(intial_angle), nargout=0)
    eng.sim(controllerModel)
    print("Final QTable")
    viewTable()
    duration = time.time() - start_time
    print(f"Simulation complete in {duration:.1f} secs")

    # Data Presentation
    # Load angles
    angles = []
    with open(angle_data_file, 'rb') as f:
        try:
            while True:
                angles.append(pickle.load(f))
        except EOFError:
            pass
        
    # Load time
    times = []
    with open(time_data_file, 'rb') as f:
        try:
            while True:
                times.append(pickle.load(f))
        except EOFError:
            pass

    # Plot data
    plt.plot(times, angles, label = "Output Signal")
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--', label=f'{stabilisation_precision} rad')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'-{stabilisation_precision} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.xlim(0,max(times))
    plt.ylim(-0.5,0.5)
    plt.title("Angle of pendulum over time")
    plt.legend()
    plt.show()

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
    plt.ylabel('Cumulative Reward')
    plt.title('Convergence of Q-learning')
    plt.show()

    eng.quit()

if __name__ == '__main__':
    main(trainModel=True)