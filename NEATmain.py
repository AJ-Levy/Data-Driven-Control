import matlab.engine
import neat
import dill
import neat.config
import numpy as np
import matplotlib.pyplot as plt
import NEATactivations as activations

def fitness(angle_lst, angle_v_lst, time_lst):
    '''
    Fitness is determined by:
    sum[(pi/2 - angle)^3] / num. iterations
    '''
    dt = 0.001
    t_max = 4
    n_max = t_max/dt
    fitness_sum = 0

    for k in range(len(angle_lst)):
        if -np.pi/2 < angle_lst[k] < np.pi/2:
            fitness_sum += (np.pi/2 - np.abs(angle_lst[k]))**3

    fitness = fitness_sum/n_max
    
    print("Fitness score:", round(fitness,4))
    return fitness


def eval_genomes(genomes, config):
    '''
    NEAT eval_genomes method to run the sim 
    and evaluate the fitness of each genome.
    '''
    for genome_id, genome in genomes:
        
        # Create and send ANN to controller
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)
        
        # Call simulation for the genome (ANN)
        eng.eval("out = sim('pendSimNEAT.slx');", nargout=0)

        # Get angles
        angle_2d = eng.eval("out.angle")
        angle_lst = []
        for angle in angle_2d:
            angle_lst.append(angle[0])

        # Get angle velocities
        angle_v_2d = eng.eval("out.angle_v")
        angle_v_lst = []
        for angle_v in angle_v_2d:
            angle_v_lst.append(angle_v[0])

        # Get time
        time_2d = eng.eval("out.time")
        time_lst = []
        for time in time_2d:
            time_lst.append(time[0]) 

        genome.fitness = fitness(angle_lst, angle_v_lst, time_lst)

def run(config_file):
    '''
    NEAT run method to handle configuration, 
    population, and statistics.
    '''
    # Load configuration
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    
    # Add my own activation functions
    activation_functions = activations.get_functions()
    for name, function in activation_functions:
        config.genome_config.add_activation(name, function)
    
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
    pop.run(eval_genomes, 30)
    eng.quit()
    return stats.best_genome()


def main():
    '''
    Main method.
    '''
    winner = run("config.txt")
    with open('winnerANN.dill', 'wb') as f:
            dill.dump(winner, f)
    print(winner)

if __name__ == '__main__':
    main()