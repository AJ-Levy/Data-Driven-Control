import matlab.engine
import neat
import dill
import neat.config
import numpy as np
import matplotlib.pyplot as plt
import bbcNEATactivations as activations

def fitness(voltage_lst, time_lst):
    '''
    Fitness is determined by:
    1 / (1 + 0.1*avg[goal - measured]^2)
    '''
    dt = 5e-6
    t_max = 0.6
    n_max = t_max/dt
    fitness_sum = 0
    goal = -70

    for k in range(len(voltage_lst)):
        fitness_sum += (1/10) * (voltage_lst[k] - goal)**2

    fitness_avg = fitness_sum/n_max

    fitness = 1/(1+fitness_avg)
    
    print("Fitness score:", round(fitness,4))
    return fitness


def get_data():
    '''
    Collect and return workspace data
    '''
    # Get voltages
    voltage_2d = eng.eval("out.voltage")
    voltage_lst = []
    for voltage in voltage_2d:
        voltage_lst.append(voltage[0])

    # Get time
    time_2d = eng.eval("out.tout")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0]) 

    pulse_2d = eng.eval("out.pulse")
    pulse_lst = []
    for pulse in pulse_2d:
        pulse_lst.append(pulse[0])
    
    pwm_2d = eng.eval("out.pwm")
    pwm_lst = []
    for pwm in pwm_2d:
        pwm_lst.append(pwm[0])

    return voltage_lst, time_lst, pulse_lst, pwm_lst


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
        eng.eval("out = sim('bbcSimNEAT.slx');", nargout=0)

        # Evaluate and assign fitness
        voltage_lst, time_lst, pulse_lst, pwm_lst = get_data()
        genome.fitness = fitness(voltage_lst, time_lst)


def run(config_file):
    '''
    NEAT run method to handle configuration, 
    population, and statistics.
    '''
    # Load configuration
    global config
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
    pop.run(eval_genomes, 10)
    return stats.best_genome()


def show_winner(winner):
    '''
    Plot results of winning ANN
    '''
    print(winner)
    print("Running winner ANN...")

    # Create winner ANN and send to controller
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('network.dill', 'wb') as f:
            dill.dump(net, f)
    eng.eval("out = sim('bbcSimNEAT.slx');", nargout=0)

    voltage_lst, time_lst, pulse_lst, pwm_lst = get_data()
    plt.plot (time_lst, voltage_lst)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (v)")
    plt.show()
    plt.plot (time_lst, pulse_lst)
    plt.xlabel("Time (s)")
    plt.ylabel("Pulse from controller")
    plt.show()
    plt.plot (time_lst, pwm_lst)
    plt.xlabel("Time (s)")
    plt.ylabel("pwm output")
    plt.show()
    print("Complete.")


# def main():
#     '''
#     Main method.
#     '''
#     winner = run("bbcConfig.txt")
#     with open('bbcWinnerANN.dill', 'wb') as f:
#             dill.dump(winner, f)
#     show_winner(winner)
#     eng.quit()

# if __name__ == '__main__':
#     main()

def show_winner_test():
    with open('bbcWinnerANN.dill', 'rb') as f:
        winner = dill.load(f)

    global config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                "bbcConfig.txt")

    global eng
    eng = matlab.engine.start_matlab()
    show_winner(winner)
show_winner_test()