import dill
import os
import neat
import numpy as np


class bbcNEATcontroller:
    '''
    Class to store ANN and previous values.
    '''
    def __init__(self):
        self.prev_pulse = 0
        self.prev_volt = 0
        self.net = None

    def update_ANN(self, net):
        '''
        Updates ANN.
        '''
        self.net = net

    def update_prev(self, pulse, volt):
        '''
        Updates previous values.
        '''
        self.prev_pulse = pulse
        self.prev_volt = volt


net_obj = bbcNEATcontroller()

def controller_call(volt, time):
    '''
    Method that MATLAB calls for NEAT.
    '''    
    # At start get new ANN and reset pulse
    if time == 0.0:
        with open('network.dill', 'rb') as f:
            net_obj.update_ANN(dill.load(f))
        os.remove("network.dill")
        net_obj.update_prev(0, 0)

    # Return output pulse and update
    inputs = [volt, net_obj.prev_pulse]
    pulse = (net_obj.net.activate(inputs)[0])
    net_obj.update_prev(pulse, volt)
    return pulse