import matlab.engine
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def viewTable(qtable_file='qtable.npy'):
    ''' 
    View QTable as an array
    '''
    qtable = np.load(qtable_file)

    # Print the Q-table
    print("Q-table:")
    print(qtable)

# reset Q table
def reset(num_states=162, num_actions=2, qtable_file='qtable.npy'):
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
        #eng.set_param(f'{model}/episode_num', 'Value', str(episode), nargout=0)
        eng.sim(model)
        if episode % (num_episodes//10) == 0:
            count += 1 
            print(f"{count*10}%")

def main(trainingModel = 'pendSimQTraining', 
         controllerModel = 'pendSimQController'):

    # Remove old data
    if os.path.exists("angleQ.pkl"):
        os.remove("angleQ.pkl")
    if os.path.exists("timeQ.pkl"):
        os.remove("timeQ.pkl")
    
    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel)
    ## Comment Out Once Model is Trained ##
    #print("Training model...")
    #train(eng, trainingModel)
    #######################################
    print("Running simulation...")
    eng.load_system(controllerModel)
    eng.sim(controllerModel)
    print("Final QTable")
    viewTable()
    print("Simulation complete")

    # Load angles
    angles = []
    with open('angleQ.pkl', 'rb') as f:
        try:
            while True:
                angles.append(pickle.load(f))
        except EOFError:
            pass
        
    # Load time
    time = []
    with open('timeQ.pkl', 'rb') as f:
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

if __name__ == '__main__':
    main()