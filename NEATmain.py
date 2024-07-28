import matlab.engine
import neat
import pickle
import neat.config
import numpy as np
import os
import matplotlib.pyplot as plt

def fitness(angle_lst, angle_v_lst, time_lst):
    '''
    Fitness is determined by:
    (1/num. iterations) *
    sum [time * (w_a*(a_worst - a_error)^2 + w_v*(v_worst - v_error)^2)]
    '''
    dt = 0.001
    t_max = 5
    n_max = t_max/dt
    fitness_sum = 0
    w_a = 1 #angle weighting
    w_v = 0 #velocity weighting

    if (len(angle_lst) != len(angle_v_lst)) or (len(angle_lst) != len(time_lst)):
        print("ERROR: List storage is of varying lengths")
        return None
    
    else:
        for k in range(len(angle_lst)):
            if -np.pi/2 < angle_lst[k] < np.pi/2:
                fitness_sum += ( (w_a * angle_lst[k]**2) + (w_v * angle_v_lst[k]**2) )

        fitness_sum += ( ((t_max-time_lst[-1])/dt) * (w_a * (np.pi/2)**2) )
        fitness = (np.pi/2)**2 - fitness_sum/n_max
        
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
        with open('network.pkl', 'wb') as f:
            pickle.dump(net, f)
        
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

def tan(x):
    return np.tan(x)

def run(config_file):
    '''
    NEAT run method to handle configuration, 
    population, and statistics.
    '''
    # Load configuration
    #neat.config.genome_config.add_activation('my_tan', tan)
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
    pop.run(eval_genomes, 20)
    eng.quit()
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