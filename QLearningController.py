import numpy as np
import pickle

class QLearningController:

    def __init__(self, num_actions=2):
        # qtable file
        self.qfile = 'qtable.npy'
        # defining q-table
        self.qtable = np.load(self.qfile)
        # last state and action
        self.last_state = None
        self.last_action = None
        # forces to be applied to cart
        self.forces = [5.0, -5.0]
        # state parameters
        self.num_actions = num_actions
        self.num_states = 18 # 0 - 17
        self.fail_state = -1
        # time keeping
        self.time = 0
        self.dt = 0.005

    def get_state(self, theta, theta_dot):
        '''
        Convert continous parameters into discrete states
        (source: https://pages.cs.wisc.edu/~finton/qcontroller.html)
        '''
        theta = np.rad2deg(theta)
        theta_dot = np.rad2deg(theta_dot)
        box = 0

        # Failure state
        if theta < -12 or theta > 12:
            return self.fail_state
        
        # angles
        if (theta < -6): box = 0
        elif (theta < -1): box = 1
        elif (theta < 0): box = 2
        elif (theta < 1): box = 3
        elif (theta < 6): box = 4
        else: box = 5

        # angular velocities
        if (theta_dot < -50): pass
        elif (theta_dot < 50):  box += 6
        else: box += 12

        return box

    def select_action(self, state):
        '''
        Selects next action using purely greedy strategy,
        returning the index of the action i.e. 0 or 1
        '''
        return np.argmax(self.qtable[state, :]) 

    def get_force(self, theta, theta_dot):
        '''
        Applies QLearning algorithm to select an action
        and then apply a force to the cart
        '''
        state = self.get_state(theta, theta_dot)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        force = self.forces[action]

        # time progresses
        self.time += self.dt

        # Write data to files
        with open('angleQ.pkl', 'ab') as f:
            pickle.dump(theta, f)

        with open('timeQ.pkl', 'ab') as f:
            pickle.dump(self.time, f) 

        return force
        
# QLearning controller
controller = QLearningController()     

def controller_call(rad_big, theta_dot):
    '''
    Method that MATLAB calls for QLearning
    '''
    global controller
    # Normalize the angle (between -pi and pi)
    theta = (rad_big%(np.sign(rad_big)*2*np.pi))
    if theta >= np.pi:
        theta -= 2 * np.pi
    force = controller.get_force(theta, theta_dot)
    return force