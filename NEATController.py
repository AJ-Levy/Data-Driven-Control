import pickle
import numpy as np
import neat
    
def controller_call(theta, theta_v):
    '''
    Method that MATLAB calls for NEAT.
    '''
    # Read in ANN
    with open('network.pkl', 'rb') as f:
        net = pickle.load(f)

    # Write angles to file
    with open('dataNEAT.pkl', 'ab') as f:
        pickle.dump(theta, f)

    # Return output force
    inputs = [theta, theta_v]
    force = net.activate(inputs)[0]
    return force


# def controller_call(theta, theta_v):
#     '''
#     Uses the winner ANN.
#     '''
#     with open('winnerANN.pkl', 'rb') as f:
#         winner = pickle.load(f)

#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                             "config.txt")

#     net = neat.nn.FeedForwardNetwork.create(winner, config)

#     inputs = [theta, theta_v]
#     force = net.activate(inputs)[0]
#     return force