import numpy as np

class QLearningAgent:

    def __init__(self, num_actions=2):
        # qtable file
        self.qfile = 'qtable.npy'
        # learning rate
        self.alpha = 0.4
        # discount factor
        self.gamma = 0.99
        # epsilon-greedy threshold
        self.epsilon = 0.2
        # defining q-table
        self.qtable = np.load(self.qfile)
        # last state and action
        self.last_state = None
        self.last_action = None
        # forces to be applied to cart
        self.forces = [3.0, -3.0]
        # state parameters
        self.fail_state = -1
        self.num_actions = num_actions
        self.num_states = 162 # 0 - 161


    def get_state(self, x, x_dot, theta, theta_dot):
        '''
        Convert continous parameters into discrete states
        (source: https://pages.cs.wisc.edu/~finton/qcontroller.html)
        '''
        theta = np.rad2deg(theta)
        theta_dot = np.rad2deg(theta_dot)
        box = 0

        # Failure state
        if x < -2.4 or x > 2.4 or theta < -12 or theta > 12:
            return self.fail_state
        
        # positions
        if x < -0.8: box = 0
        elif x > 0.8: box = 1
        else: box = 2

        # velocities
        if (x_dot < -0.5): pass
        elif (x_dot < 0.5): box += 3
        else: box += 6

        # angles
        if (theta < -6): pass
        elif (theta < -1): box += 9
        elif (theta < 0): box += 18
        elif (theta < 1): box += 27
        elif (theta < 6): box += 36
        else: box += 45

        # angular velocities
        if (theta_dot < -50): pass
        elif (theta_dot < 50):  box += 54
        else: box += 108

        return box
            

    def select_action(self, state):
        '''
        Selects next action using epsilon greedy strategy,
        returning the index of the action i.e. 0 or 1
        '''
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.qtable[state, :]) 

    
    def update(self, state, action, reward, next_state):
        '''
        Implementation of QLearning governing equation
        '''
        q_old = self.qtable[state, action]
        q_new = reward + self.gamma * np.max(self.qtable[next_state])
        self.qtable[state, action] = q_old + self.alpha * (q_new - q_old)
        # save updated qtable
        np.save(self.qfile, self.qtable)

    def get_force(self, theta, theta_dot, x, x_dot):
        '''
        Applies QLearning algorithm to select an action
        and then apply a force to the cart
        '''
        state = self.get_state(x, x_dot, theta, theta_dot)
        if self.last_state is not None:
            reward = self.reward_function(state)
            self.update(self.last_state, self.last_action, reward, state)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        force = self.forces[action] 

        return force
        
    # simple for now
    def reward_function(self, state):
        if state == self.fail_state:
            return -1
        return 0
        
# QLearning agent
agent = QLearningAgent()     

def controller_call(rad_big, theta_dot, x, x_dot):
    '''
    Method that MATLAB calls for QLearning
    '''
    global agent
    # Normalize the angle (between -pi and pi)
    theta = (rad_big%(np.sign(rad_big)*2*np.pi))
    if theta >= np.pi:
        theta -= 2 * np.pi
    force = agent.get_force(theta, theta_dot, x, x_dot)
    return force