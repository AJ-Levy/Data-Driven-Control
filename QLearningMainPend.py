import matlab.engine
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
def reset(num_states=144, num_actions=4, qtable_file='qtable.npy'):
    '''
    Reset QTable to 2D array of zeros of size
    num_states x num_actions.
    '''
    qtable = np.zeros((num_states, num_actions))
    np.save(qtable_file, qtable)

def train(eng, model, mask, convergence_data_file, num_episodes=1500, count=0):
    '''
    Train QLearning Agent
    '''
    # clean up algorithm convergence file
    if os.path.exists(convergence_data_file):
        os.remove(convergence_data_file)

    reset()
    for episode in range(1,num_episodes+1):
        initial_angle = genIntialAngle()
        # pass in current episode
        eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)
        # pass in random intial angular offset
        eng.set_param(f'{model}/{mask}', 'init', str(initial_angle), nargout=0)
        eng.sim(model)
        if episode % (num_episodes//10) == 0:
            count += 1 
            print(f"{count*10}%")

def setNoise(eng, model, noise):
    if noise:
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'Cov', str([0.00001]), nargout=0)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise_v', 'Cov', str([0.001]), nargout=0)
        eng.set_param(f'{model}/Noise_v', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)
        eng.set_param(f'{model}/Noise_v', 'Cov', str([0]), nargout=0)

def genIntialAngle(delta=np.pi/3):
    '''
    Generates a random intial angle about 0 radians.
    The angles lies in the range [-delta, delta) radians.
    '''
    return np.random.uniform(-delta, delta)

def main(trainModel = True, 
         noise = False,
         trainingModel = 'pendSimQTraining', 
         controllerModel = 'pendSimQController',
         cartPoleSubsystem = 'Pendulum and Cart',
         convergence_data_file = 'qconverge.txt',
         stabilisation_precision = 0.05):

    global time

    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel, nargout=0)
    start_time = time.time()
    if trainModel:
        print("Training model...")
        train(eng, trainingModel, cartPoleSubsystem, convergence_data_file)
    print("Running simulation...")
    eng.load_system(controllerModel, nargout=0)
   
    # show all angles in specified range do acutally stabilise
    for i in range(1):
        # pass in angular offset
        ang = 0.6
        eng.set_param(f'{controllerModel}/{cartPoleSubsystem}', 'init', str(ang), nargout=0)
        setNoise(eng, controllerModel, noise)
        eng.eval(f"out = sim('{controllerModel}');", nargout=0)
        #print("Final QTable")
        #viewTable()
        #print(f"Initial Angle: {np.rad2deg(initial_angle):.1f} degrees")

        ## Data Presentation
        # Get angles
        angle_2d = eng.eval("out.angle")
        angle_lst = []
        for a in angle_2d:
            angle_lst.append(a[0])

        # Get time
        time_2d = eng.eval("out.time")
        time_lst = []
        for t in time_2d:
            time_lst.append(t[0])

        # Plot data
        plt.plot(time_lst, angle_lst, label = f"{ang:.2f} rad")
        
    # configure plot
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.2f} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("θ (rad)")
    plt.xlim(0,3)
    plt.ylim(-np.pi/2, np.pi/2)
    plt.legend(loc = 'upper right', fontsize = "11")
    plt.savefig('plot.pdf', format='pdf')
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

    plt.plot(episodes, rewards, color = 'blue')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.savefig('pendQConvergence.pdf', format='pdf')
    plt.show()

    eng.quit()

if __name__ == '__main__':
    main(trainModel=False, noise=False)