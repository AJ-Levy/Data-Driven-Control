import pickle
import numpy as np
import neat

class storage:
    def __init__(self, dt, end_time):
        self.time = 0
        self.angle = []
        self.angle_v = []
        self.net = None
        self.dt = dt
        self.end_time = end_time

    def update(self, theta, theta_v):
        self.time = round(self.time+self.dt, 3)
        self.angle.append(theta)
        self.angle_v.append(theta_v)

    def get_net(self):
        with open('network.pkl', 'rb') as f:
            net_in = pickle.load(f)
        self.net = net_in

    def reset(self):
        self.time = 0
        self.angle = []
        self.angle_v = []
        self.net = None

    def write_data(self):
        with open('angleNEAT.pkl', 'wb') as f:
            pickle.dump(self.angle, f)
        with open('angle_vNEAT.pkl', 'wb') as f:
            pickle.dump(self.angle_v, f)


controller = storage(dt = 0.005, end_time=8)
    
def controller_call(theta, theta_v):
    '''
    Method that MATLAB calls for NEAT.
    '''
    #controller.write_data()
    # Reset sim and write data if out of bounds/time's up
    if (theta > 1.4) or (theta < -1.4) or (controller.time >= controller.end_time):
        controller.write_data()
        controller.reset()
        return 0
    
    # Get ANN
    if controller.net == None:
        controller.get_net()

    # Return output force and update
    force = 30*controller.net.activate([theta])[0]
    controller.update(theta, theta_v)
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

#     inputs = [theta]
#     force = 30*net.activate(inputs)[0]
#     return force