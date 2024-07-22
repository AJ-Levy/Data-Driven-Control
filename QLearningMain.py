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
    num_states x num_actions
    '''
    qtable = np.zeros((num_states, num_actions))
    np.save(qtable_file, qtable)

def train(eng, model, num_episodes=1000, count=0):
    '''
    Train QLearning Agent
    '''
    reset()
    for episode in range(1,num_episodes+1):
        eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)
        eng.sim(model)
        if episode % (num_episodes//10) == 0:
            count += 1 
            print(f"{count*10}%")

def main(trainingModel = 'pendSimQTraining', 
         controllerModel = 'pendSimQController',
         angle_data_file = 'angleQ.pkl',
         time_data_file = 'timeQ.pkl',
         convergence_data_file = 'qconverge.txt'):

    global time

    # Remove old data
    if os.path.exists(angle_data_file):
        os.remove(angle_data_file)
    if os.path.exists(time_data_file):
        os.remove(time_data_file)
    if os.path.exists(convergence_data_file):
        os.remove(convergence_data_file)
    
    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel, nargout=0)
    start_time = time.time()
    ## Comment Out Once Model is Trained ##
    print("Training model...")
    train(eng, trainingModel)
    #######################################
    print("Running simulation...")
    eng.load_system(controllerModel, nargout=0)
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
    time = []
    with open(time_data_file, 'rb') as f:
        try:
            while True:
                time.append(pickle.load(f))
        except EOFError:
            pass

    # Plot data
    plt.plot(time, angles)
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.ylim(-0.5,0.5)
    plt.title("Angle of pendulum over time")
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
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Convergence of Q-learning')
    plt.show()

if __name__ == '__main__':
    main()