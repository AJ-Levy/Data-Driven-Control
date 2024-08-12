import numpy as np

class QLearningController:

    def __init__(self, num_actions=4):
        # qtable file
        self.qfile = 'qtable_10.npy'
        # defining q-table
        self.qtable = np.load(self.qfile)
        # last state and action
        self.last_action = None
        # forces to be applied to cart
        self.forces = [10.0, 30.0, -10.0, -30.0]
        # state parameters
        self.num_actions = num_actions
        self.num_states = 144 # 0 - 143
        self.fail_state = -1

    def get_state(self, theta, theta_dot):
        '''
        Convert continous parameters into discrete states
        (source: https://pages.cs.wisc.edu/~finton/qcontroller.html)
        '''
        theta = np.rad2deg(theta)
        theta_dot = np.rad2deg(theta_dot)
        box = 0

        # Failure state
        if theta < -60 or theta > 60:
            return self.fail_state
        
        # angles
        if (theta < -51): box = 0
        elif(theta < -46): box = 1
        elif (theta < -41): box = 2
        elif (theta < -36): box = 3
        elif (theta < -31): box = 4
        elif(theta < -26): box = 5
        elif (theta < -21): box = 6
        elif(theta < -16): box = 7
        elif (theta < -11): box = 8
        elif (theta < -6): box = 9
        elif (theta < -1): box = 10
        elif (theta < 0): box = 11
        elif (theta < 1): box = 12
        elif (theta < 6): box = 13
        elif (theta < 11): box = 14
        elif(theta < 16): box = 15
        elif (theta < 21): box = 16
        elif (theta < 26): box = 17
        elif(theta < 31): box = 18
        elif (theta < 36): box = 19
        elif (theta < 41): box = 20
        elif (theta < 46): box = 21
        elif(theta < 51): box = 22
        else: box = 23

        # angular velocities
        if (theta_dot < -50): pass
        elif (theta_dot < -25): box += 24
        elif (theta_dot < 0): box += 48
        elif (theta_dot < 25): box += 72
        elif (theta_dot < 50):  box += 96
        else: box += 120

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