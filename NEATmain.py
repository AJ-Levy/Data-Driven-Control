import matlab.engine
import neat
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def fitness(angle_list):
    '''
    Fitness is determined by the average squared
    diplacement from the verticle.
    '''
    average = np.mean(np.power(angle_list, 2))
    fitness = (np.pi/2)**2 - average
    print("fitness score:", fitness)
    return fitness


def eval_genomes(genomes, config):
    '''
    NEAT eval_genomes method to run the sim 
    and evaluate the fitness of each genome.
    '''
    if os.path.exists("angleNEAT.pkl"):
        os.remove("angleNEAT.pkl")
    if os.path.exists("angle_vNEAT.pkl"):
        os.remove("angle_vNEAT.pkl")

    for genome_id, genome in genomes:
        
        # Create and send ANN to controller
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.pkl', 'wb') as f:
            pickle.dump(net, f)
        
        # Call simulation for the genome (ANN)
        print("Running simulation...")
        eng.sim('pendSimNEAT.slx')

        # Recieve angle data
        with open('angleNEAT.pkl', 'rb') as f:
            angle = pickle.load(f)
        os.remove('angleNEAT.pkl')
        with open('angle_vNEAT.pkl', 'rb') as f:
            angle_v = pickle.load(f)
        os.remove('angle_vNEAT.pkl')
        # print(angle)
        # plt.plot(angle)
        # plt.show()
        #inputa = input("readY)")
        # Return fitness
        genome.fitness = fitness(angle)


def run(config_file):
    '''
    NEAT run method to handle configuration, 
    population, and statistics.
    '''
    # Load configuration
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    
    # Create the population and add stats reporter.
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Parallel Evaluator
    #pe = neat.ParallelEvaluator(3, eval_genomes)

    # Set up MATLAB engine
    print("Setting up engine...")
    global eng
    eng = matlab.engine.start_matlab()

    # Run NEAT and return best Genome
    pop.run(eval_genomes, 25)
    return stats.best_genome()


def main():
    '''
    Main method.
    '''
    winner = run("config.txt")
    with open('winnerANN.pkl', 'wb') as f:
            pickle.dump(winner, f)
    print(winner)

if __name__ == '__main__':
    main()