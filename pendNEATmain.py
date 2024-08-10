import matlab.engine
import neat
import dill
import neat.config
import numpy as np
import matplotlib.pyplot as plt
import pendNEATactivations as activations

def fitness(angle_lst, angle_v_lst, time_lst):
    '''
    Fitness is determined by:
    sum[cos(theta)] / num. iterations
    '''
    dt = 0.001
    t_max = 4
    n_max = t_max/dt
    n_reached = time_lst[-1]/dt
    fitness_sum = 0

    for k in range(len(angle_lst)):
        if -np.pi/2 <= angle_lst[k] <= np.pi/2:
            fitness_sum += (1-np.abs((2*angle_lst[k])/np.pi))**2

    fitness = (fitness_sum)/n_max
    
    print("Fitness score:", round(fitness,4))
    return fitness


def get_data():
    '''
    Collect and return workspace data
    '''
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
    time_2d = eng.eval("out.tout")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0]) 
        
    return angle_lst, angle_v_lst, time_lst


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

        # Set random intial angle and call sim for the genome
        delta = 0.5
        model = 'pendSimNEAT'
        intial_angle = np.random.uniform(-delta, delta)
        while intial_angle <= 0.05 and intial_angle >= -0.05:
            intial_angle = np.random.uniform(-delta, delta)
        eng.set_param(f'{model}/Pendulum and Cart', 'init', str(intial_angle), nargout=0)

        # Set random noise seeds
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_seed_v = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise_v', 'seed', f'[{noise_seed_v}]', nargout=0)

        # Evaluate and assign fitness
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        angle_lst, angle_v_lst, time_lst = get_data()
        genome.fitness = fitness(angle_lst, angle_v_lst, time_lst)


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
    eng.load_system('pendSimNEAT', nargout=0)


    # Run NEAT and return best Genome
    pop.run(eval_genomes, 20)
    return stats.best_genome()


def show_winner(winner):
    '''
    Plot results of winning ANN
    '''
    print(winner)
    print("Running winner ANN...")
    
    i = -0.5
    while i <= 0.5:
        # Create winner ANN and send to controller
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)
        
        # Set random intial angles and call sim
        model = 'pendSimNEAT'
        intial_angle = i
        i += 0.2
        eng.set_param(f'{model}/Pendulum and Cart', 'init', str(intial_angle), nargout=0)

        # Set random noise seeds
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_seed_v = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise_v', 'seed', f'[{noise_seed_v}]', nargout=0)
        
        # Run sim
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        angle_lst, angle_v_lst, time_lst = get_data()
        plt.plot(time_lst, angle_lst)

    # Plot
    error_lst_pos = []
    error_lst_neg = []
    for i in range(len(time_lst)):
        error_lst_pos.append(0.05)
        error_lst_neg.append(-0.05)

    plt.plot(time_lst, error_lst_pos, linestyle='dashed', color = 'grey')
    plt.plot(time_lst, error_lst_neg, linestyle='dashed', color = 'grey')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.xlim(0,4)
    plt.show()
    print("Complete.")


def main():
    '''
    Main method.
    '''
    winner = run("pendConfig.txt")
    with open('bbcWinnerANN.dill', 'wb') as f:
            dill.dump(winner, f)
    show_winner(winner)
    eng.quit()

if __name__ == '__main__':
    main()

# def show_winner_test():
#     with open('winnerANN.dill', 'rb') as f:
#         winner = dill.load(f)

#     global config
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                                 "pendConfig.txt")
    
#     activation_functions = activations.get_functions()
#     for name, function in activation_functions:
#         config.genome_config.add_activation(name, function)

#     global eng
#     eng = matlab.engine.start_matlab()
#     eng.load_system('pendSimNEAT', nargout=0)

#     show_winner(winner)
# show_winner_test()