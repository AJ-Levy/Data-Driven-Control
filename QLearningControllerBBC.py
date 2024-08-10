import numpy as np

class QLearningController:

    def __init__(self, num_actions=2):
        # qtable file
        self.qfile = 'qtable_BBC.npy'
        # defining q-table
        self.qtable = np.load(self.qfile)
        # last state and action
        self.last_action = None
        # forces to be applied to cart
        self.duty_cycles = [0,1]
        # state parameters
        self.num_actions = num_actions
        self.num_states = 10 # 0 - 17
        self.fail_state = -1

    def get_state(self, voltage):
        '''
        Convert continous parameters into discrete states
        '''
        box = 0

        if (voltage < -13 or voltage > 13):
            return self.fail_state
        
        # voltages
        '''
        if voltage < -100: box = 0
        if voltage < -80: box = 1
        elif voltage < -60: box = 2
        elif voltage < -40: box = 3
        elif voltage < -30: box = 4
        elif voltage < -23: box = 5
        elif voltage < -16: box = 6
       
        '''
        if voltage < -10: box = 0
        elif voltage < -5: box = 1
        elif voltage < -3: box = 2
        elif voltage < -1: box = 3
        elif voltage < 0: box = 4
        elif voltage < 1: box = 5
        elif voltage < 3: box = 6
        elif voltage < 5: box = 7
        elif voltage < 10: box = 8
        else: box = 9
        '''
        elif voltage < 16: box = 16
        elif voltage < 23: box = 17
        elif voltage < 30: box = 18
        elif voltage < 40: box = 19
        elif voltage < 60: box = 20
        elif voltage < 80: box = 21
        elif voltage < 100: box = 22
        '''

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
        duty_cycle = self.duty_cycles[action]

        return duty_cycle
        
# QLearning controller
controller = QLearningController()   

def controller_call(voltage):
    '''
    Method that MATLAB calls for QLearning
    '''
    global controller

    duty_cycle = controller.get_output(voltage)
    return duty_cycle