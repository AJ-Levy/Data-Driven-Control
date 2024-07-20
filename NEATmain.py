import matlab.engine
import neat
import pickle
import numpy as np
import os

def fitness(data):
    '''
    Fitness is determined by the average 
    diplacement from the desired rest point.
    '''
    average = np.mean(np.abs(data))
    fitness = np.pi/2 - average
    print("fitness score:", fitness)
    return fitness


def eval_genomes(genomes, config):
    '''
    NEAT eval_genomes method to run the sim 
    and evaluate the fitness of each genome.
    '''
    if os.path.exists("dataNEAT.pkl"):
        os.remove("dataNEAT.pkl")

    for genome_id, genome in genomes:
        
        # Create and send ANN to controller
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.pkl', 'wb') as f:
            pickle.dump(net, f)
        
        # Call simulation for the genome (ANN)
        print("Running simulation...")
        eng.sim('pendSimNEAT.slx')

        # Recieve angle data
        data = []
        with open('dataNEAT.pkl', 'rb') as f:
            try:
                while True:
                    data.append(pickle.load(f))
            except EOFError:
                pass
        os.remove("dataNEAT.pkl")
    
        # Return fitness
        genome.fitness = fitness(data)


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
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Set up MATLAB engine
    print("Setting up engine...")
    global eng
    eng = matlab.engine.start_matlab()

    # Run NEAT and return best Genome
    p.run(eval_genomes, 40)
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