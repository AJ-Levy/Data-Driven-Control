import numpy as np

class QLearningController:

    def __init__(self, num_actions=5):
        # qtable file
        self.qfile = 'qtable_BC.npy'
        # defining q-table
        self.qtable = np.load(self.qfile)
        # last state and action
        self.last_action = None
        # actions to be applied (duty cycle adjustments)
        self.actions = [0.1, 0.3, 0.5, 0.7, 0.9]
        # state parameters
        self.num_actions = num_actions
        self.num_states = 24 # 0 - 71
        self.fail_state = -1

    def get_state(self, voltage):
        '''
        Convert continous parameters into discrete states
        '''
        if (voltage < -100 or voltage > 100):
            return self.fail_state
        
        # voltages
        if voltage < -60: box = 0
        elif voltage < -40: box = 1
        elif voltage < -30: box = 2
        elif voltage < -23: box = 3
        elif voltage < -16: box = 4
        elif voltage < -13: box = 5
        elif voltage < -10: box = 6
        elif voltage < -8: box = 7
        elif voltage < -5: box = 8
        elif voltage < -3: box = 9
        elif voltage < -1: box = 10
        elif voltage < 0: box = 11
        elif voltage < 1: box = 12
        elif voltage < 3: box = 13
        elif voltage < 5: box = 14
        elif voltage < 8: box = 15
        elif voltage < 10: box = 16
        elif voltage < 13: box = 17
        elif voltage < 16: box = 18
        elif voltage < 23: box = 19
        elif voltage < 30: box = 20
        elif voltage < 40: box = 21
        elif voltage < 60: box = 22
        else: box = 23

        return box
    
    def select_action(self, state):
        '''
        Selects next action using purely greedy strategy,
        returning the index of the action i.e. 0 or 1
        '''
        return np.argmax(self.qtable[state, :]) 

    def get_output(self, voltage):
        '''
        Applies QLearning algorithm to select an action
        and then apply a force to the cart
        '''
        state = self.get_state(voltage)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        duty_cycle = self.actions[action]
        return duty_cycle
        
# QLearning controller
controller = QLearningController()   

def controller_call(voltage):
    '''
    Method that MATLAB calls for QLearning
    '''
    global controller

    output = controller.get_output(voltage)
    return output