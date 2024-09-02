import dill
import os
import neat
import pendNEATactivations as activations
import numpy as np


class pendNEATcontroller:
    '''
    Class to store ANN and previous values.
    '''
    def __init__(self):
        self.prev_force = 0
        self.prev_theta = np.pi/6
        self.net = None

    def update_ANN(self, net):
        '''
        Updates ANN.
        '''
        self.net = net

    def update_prev(self, force, theta):
        '''
        Updates previous values.
        '''
        self.prev_force = force
        self.prev_theta = theta


net_obj = pendNEATcontroller()

def controller_call(theta, theta_v, time):
    '''
    Method that MATLAB calls for NEAT.
    '''    
    # At start get new ANN and reset force
    if time == 0.0:
        with open('network.dill', 'rb') as f:
            net_obj.update_ANN(dill.load(f))
        os.remove("network.dill")
        net_obj.update_prev(0, np.pi/6)

    # Return output force and update
    inputs = [theta, theta_v]
    force = (net_obj.net.activate(inputs)[0])
    net_obj.update_prev(force, theta)
    return force