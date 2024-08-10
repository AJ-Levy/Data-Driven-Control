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
    inputs = [theta, theta_v, net_obj.prev_force]
    force = (net_obj.net.activate(inputs)[0])
    net_obj.update_prev(force, theta)
    return force

# def controller_call(theta, theta_v, time):
#     '''
#     Uses the winner ANN.
#     Run from MATLAB.
#     ''' 
#     # Get winner ANN
#     with open('winnerANN.dill', 'rb') as f:
#         winner = dill.load(f)
    
#     # Load configuration
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                                 "pendConfig.txt")
    
#     # Add my own activation functions
#     activation_functions = activations.get_functions()
#     for name, function in activation_functions:
#         config.genome_config.add_activation(name, function)

#     net = neat.nn.FeedForwardNetwork.create(winner, config)

#     inputs = [theta, theta_v, net_obj.prev_force]
#     force = net.activate(inputs)[0]
#     net_obj.update_prev(force, theta)
#     return force