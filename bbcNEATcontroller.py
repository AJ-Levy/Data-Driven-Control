import dill
import os
import neat
import numpy as np


class bbcNEATcontroller:
    '''
    Class to store ANN and previous values.
    '''
    def __init__(self):
        self.prev_error = 0
        self.net = None

    def update_prev(self, error):
        '''
        Updates previous values.
        '''
        self.prev_error = error

    def update_ANN(self, net):
        '''
        Updates ANN.
        '''
        self.net = net


net_obj = bbcNEATcontroller()

def controller_call(goal, v_out, v_in, time):
    '''
    Method that MATLAB calls for NEAT.
    '''    
    # At start get new ANN and reset pulse
    if time == 0.0:
        with open('network.dill', 'rb') as f:
            net_obj.update_ANN(dill.load(f))
        os.remove("network.dill")
        net_obj.update_prev(0)

    error = (v_out-goal)
    error_dt = (error - net_obj.prev_error) / 5e-6

    # Return output pulse and update
    inputs = [(v_out), error, error_dt]
    pulse = (net_obj.net.activate(inputs)[0])
    net_obj.update_prev(error)

    # Saturation
    if goal == -30:
        if pulse >= 0.5:
            pulse = 0.5
        elif pulse <= 0.1:
            pulse = 0.1
    else:
        if pulse >= 0.9:
            pulse = 0.9
        elif pulse <= 0.5:
            pulse = 0.5

    return pulse