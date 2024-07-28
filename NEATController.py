import pickle
import os
import neat

class NEATController:
    '''
    Class to store ANN and previous force.
    '''
    def __init__(self):
        self.prev_force = 0
        self.net = None

    def update_ANN(self, net):
        '''
        Updates ANN.
        '''
        self.net = net

    def update_force(self, force):
        '''
        Updates previous force.
        '''
        self.prev_force = force


net_obj = NEATController()

def controller_call(theta, theta_v, time):
    '''
    Method that MATLAB calls for NEAT.
    '''    
    # Get new ANN at the start of the simulation
    # and reset previous force
    if time == 0.0:
        with open('network.pkl', 'rb') as f:
            net_obj.update_ANN(pickle.load(f))
        os.remove("network.pkl")
        net_obj.update_force(0)

    # Return output force and update
    inputs = [theta, theta_v, net_obj.prev_force]
    force = 30*(net_obj.net.activate(inputs)[0])
    net_obj.update_force(force/30)
    return force


# def controller_call(theta, theta_v):
#     '''
#     DEPRECIATED Method that MATLAB calls for NEAT (3.6x slower).
#     '''
#     # Read in ANN
#     with open('network.pkl', 'rb') as f:
#         net = pickle.load(f)

#     # Write angles to file
#     with open('angleNEAT.pkl', 'ab') as f:
#         pickle.dump(theta, f)
#     with open('angle_vNEAT.pkl', 'ab') as f:
#         pickle.dump(theta_v, f)

#     # Return output force
#     inputs = [theta]
#     force = net.activate(inputs)[0]
#     return force


# def controller_call(theta, theta_v, time):
#     '''
#     Uses the winner ANN.
#     Run from MATLAB.
#     '''
#     with open('winnerANN.pkl', 'rb') as f:
#         winner = pickle.load(f)

#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                             "config.txt")

#     net = neat.nn.FeedForwardNetwork.create(winner, config)

#     inputs = [theta, theta_v, net_obj.prev_force]
#     force = 30*net.activate(inputs)[0]
#     net_obj.update_force(force/30)
#     return force