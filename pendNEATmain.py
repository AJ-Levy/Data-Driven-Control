import matlab.engine
import neat
import dill
import neat.config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pendNEATactivations as activations
import os

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
            fitness_sum += (1-np.abs(2*angle_lst[k]/np.pi))**2

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
    fitness_lst = []
    for genome_id, genome in genomes:
        
        # Create and send ANN to controller
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)

        # Set random intial angle and call sim for the genome
        delta = 1
        model = 'pendSimNEAT'
        intial_angle = np.random.uniform(-delta, delta)
        while intial_angle <= 0.05 and intial_angle >= -0.05:
            intial_angle = np.random.uniform(-delta, delta)
        eng.set_param(f'{model}/Pendulum and Cart', 'init', str(intial_angle), nargout=0)

        # Set random noise seeds
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_power = 0
        eng.set_param(f'{model}/Noise', 'Cov', f'[{noise_power}]', nargout=0)
        noise_seed_v = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise_v', 'seed', f'[{noise_seed_v}]', nargout=0)
        noise_power_v = 0
        eng.set_param(f'{model}/Noise_v', 'Cov', f'[{noise_power_v}]', nargout=0)

        # Evaluate and assign fitness
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        angle_lst, angle_v_lst, time_lst = get_data()
        fitness_score = fitness(angle_lst, angle_v_lst, time_lst)
        genome.fitness = fitness_score
        fitness_lst.append(fitness_score)

    # Write fitness scores to file
    if os.path.exists('pendFitness.dill'):
        with open('pendFitness.dill', 'rb') as f:
            existing_lst = dill.load(f)
    else:
        existing_lst = []
    existing_lst.append(fitness_lst)
    with open('pendFitness.dill', 'wb') as f:
        dill.dump(existing_lst, f)


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

    if os.path.exists('pendFitness.dill'):
        os.remove('pendFitness.dill')

    # Run NEAT and return best Genome
    pop.run(eval_genomes, 25)
    return stats.best_genome()


def show_winner(winner_lst):
    '''
    Plot results of winning ANN
    '''
    # print(winner)
    print("Running winner ANN...")
    # Set fonts for plotting
    font_path = 'LinLibertine_Rah.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Linux Libertine'
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(8, 6))
    stab_time_lst = []
    combined_lst = []
    
    # i = -1.000001
    # while i <= 1:
    for winner in winner_lst:
        print(winner)
        i=0.6
        # Create winner ANN and send to controller
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)
        
        # Set random intial angles and call sim
        model = 'pendSimNEAT'
        initial_angle = round(i,1)
        eng.set_param(f'{model}/Pendulum and Cart', 'init', str(initial_angle), nargout=0)

        # Set random noise
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        noise_power = 0
        eng.set_param(f'{model}/Noise', 'Cov', f'[{noise_power}]', nargout=0)
        noise_seed_v = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise_v', 'seed', f'[{noise_seed_v}]', nargout=0)
        noise_power_v = 0
        eng.set_param(f'{model}/Noise_v', 'Cov', f'[{noise_power_v}]', nargout=0)
        
        # Run sim
        eng.eval(f"out = sim('{model}.slx');", nargout=0)
        angle_lst, angle_v_lst, time_lst = get_data()

        # Get stability time
        stab_time = None
        for j in range(len(angle_lst)):
            if -0.05 <= angle_lst[j] <= 0.05:
                if stab_time == None:
                    stab_time = time_lst[j]
            else:
                stab_time = None
        stab_time_lst.append(stab_time)

        # Combined data
        if combined_lst == []:
            combined_lst = angle_lst
        else:
            for k in range(len(angle_lst)):
                combined_lst[k] = combined_lst[k] + angle_lst[k]

        plt.plot(time_lst, angle_lst)#, color='darkgrey')
        # i += 0.4

    # Print stabilization 
    if None in stab_time_lst:
        print("Failed to stabalize")
    else:
        print("Average stab time:", round(np.mean(stab_time_lst),3), "+/-", round(np.std(stab_time_lst),3))
        print("Best stab time:", round(min(stab_time_lst),3))
        print("Worst stab time:", round(max(stab_time_lst),3))

    # # Average data
    # for l in range(len(combined_lst)):
    #     combined_lst[l] = combined_lst[l]/5
    # plt.plot(time_lst, combined_lst, color = "red", label = "Averaged ANN performance")

    # Plot
    plt.axhline(y=0.05, linestyle='--', color = 'k', label='$\pm$ 0.05 steady state error')
    plt.axhline(y=-0.05, linestyle='--', color = 'k')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.xlim(0,0.5)
    plt.ylim(-1,1)
    plt.legend()

        
    plt.savefig('plot_test.pdf', format='pdf')
    plt.show()


    # ### Fitness ###
    # plt.figure(figsize=(8, 6))
    # # Read data from the file
    # with open('pendFitness5.dill', 'rb') as f:
    #     fitness_lst = dill.load(f)

    # # Plot fitness per generation and average
    # average_fitness = []
    # max_fitness = []
    # for x in range(len(fitness_lst)):
    #     generation = []
    #     for i in range(len(fitness_lst[x])):
    #         generation.append(x)
    #     plt.scatter(generation, fitness_lst[x], color='darkgrey')
    #     average_fitness.append(np.mean(fitness_lst[x]))
    #     max_fitness.append(np.max(fitness_lst[x]))
    # plt.plot(max_fitness, color='blue', label='Max Fitness')
    # plt.plot(average_fitness, color='red', label='Average Fitness')
    # plt.savefig('fitness_test.pdf', format='pdf')
    # plt.xlabel('Generation Number')
    # plt.ylabel('Fitness Score')
    # plt.legend()
    # plt.xlim(0,25)
    # plt.ylim(0,1.25)
    # plt.savefig('fitness_test.pdf', format='pdf')
    # plt.show()
    
    print("Complete.")


# def main():
#     '''
#     Main method.
#     '''
#     winner = run("pendConfig.txt")
#     with open('pendWinner.dill', 'wb') as f:
#             dill.dump(winner, f)
#     show_winner(winner)
#     eng.quit()

# if __name__ == '__main__':
#     main()

def show_winner_test():
    '''
    Uncomment this, and comment out main method to run the winner again
    '''
    with open('pendWinner1.dill', 'rb') as f:
        winner1 = dill.load(f)
    # with open('pendWinner2.dill', 'rb') as f:
    #     winner2 = dill.load(f)
    # with open('pendWinner3.dill', 'rb') as f:
    #     winner3 = dill.load(f)
    # with open('pendWinner4.dill', 'rb') as f:
    #     winner4 = dill.load(f)
    # with open('pendWinner5.dill', 'rb') as f:
    #     winner5 = dill.load(f)

    global config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                "pendConfig.txt")
    
    activation_functions = activations.get_functions()
    for name, function in activation_functions:
        config.genome_config.add_activation(name, function)

    global eng
    eng = matlab.engine.start_matlab()
    eng.load_system('pendSimNEAT', nargout=0)

    show_winner([winner1])
    eng.quit()
show_winner_test()