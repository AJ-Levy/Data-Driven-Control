import matlab.engine
import neat
import dill
import neat.config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

def fitness(voltage_lst, time_lst):
    '''
    Fitness is determined by:
    1 / (1 + 0.1*avg[goal - measured]^2)
    '''
    dt = 5e-6
    t_max = 0.15
    n_max = t_max/dt
    fitness_sum = 0
    goal = -30
    error = 5

    for k in range(len(voltage_lst)):
        if (goal-error) <= voltage_lst[k] <= (goal+error):
            fitness_sum += (error - abs(voltage_lst[k]-goal)) / error
        else:
            fitness_sum += 0

    fitness = fitness_sum/n_max
    
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
    fitness_lst = []
    for genome_id, genome in genomes:
        
        # Create and send ANN to controller
        print(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open('network.dill', 'wb') as f:
            dill.dump(net, f)

        # # Set random noise seed
        # noise_seed = str(np.random.randint(1, high=40000))
        # eng.set_param(f'bbcSimNEAT/Noise', 'seed', f'[{noise_seed}]', nargout=0)

        # # Set random initial Voltage
        # voltage = str(np.random.randint(20, high=70))
        # eng.set_param(f'bbcSimNEAT/BBC/V_source_value', 'Amplitude', voltage, nargout=0)
        # print("Initial Voltage:", voltage)
        
        # Evaluate and assign fitness
        eng.eval("out = sim('bbcSimNEAT.slx');", nargout=0)
        voltage_lst, time_lst, pulse_lst, pwm_lst = get_data()
        fitness_score = fitness(voltage_lst, time_lst)
        genome.fitness = fitness_score
        fitness_lst.append(fitness_score)

    # Write fitness scores to file
    if os.path.exists('bbcFitness.dill'):
        with open('bbcfitness.dill', 'rb') as f:
            existing_lst = dill.load(f)
    else:
        existing_lst = []
    existing_lst.append(fitness_lst)
    with open('bbcFitness.dill', 'wb') as f:
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

    # Create the population and add stats reporter.
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Set up MATLAB engine
    print("Setting up engine...")
    global eng
    eng = matlab.engine.start_matlab()
    eng.load_system('bbcSimNEAT', nargout=0)

    if os.path.exists('bbcFitness.dill'):
        os.remove('bbcFitness.dill')

    # Run NEAT and return best Genome
    pop.run(eval_genomes, 15)
    return stats.best_genome()


def show_winner(winner):
    '''
    Plot results of winning ANN
    '''
    goal = -30
    print(winner)
    print("Running winner ANN...")
    # Set fonts for plotting
    font_path = 'LinLibertine_Rah.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Linux Libertine'
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(8, 6))
    stab_time_lst = []
    
    # Create winner ANN and send to controller
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('network.dill', 'wb') as f:
            dill.dump(net, f)
    
    # Run winner and get data
    model = 'bbcSimNEAT'
    eng.eval(f"out = sim('{model}.slx');", nargout=0)
    voltage_lst, time_lst, pulse_lst, pwm_lst = get_data()

    # Get stability time
    stab_time = None
    for j in range(len(voltage_lst)):
        if (goal-3) <= voltage_lst[j] <= (goal+3):
            if stab_time == None:
                stab_time = time_lst[j]
        else:
            stab_time = None
    stab_time_lst.append(stab_time)
    
    # Show average duty cycle
    on = 0
    for pwm in pwm_lst:
        if pwm == 1:
            on+=1
    print("Average Duty Cycle:", on/len(pwm_lst))

    # Print stabilization 
    if None in stab_time_lst:
        print("Failed to stabalize")
    else:
        print("Average stab time:", round(np.mean(stab_time_lst),3), "+/-", round(np.std(stab_time_lst),3))
        print("Best stab time:", round(min(stab_time_lst),3))
        print("Worst stab time:", round(max(stab_time_lst),3))

    # Plot
    plt.plot(time_lst, voltage_lst, label='Output Voltage')
    plt.axhline(y=(goal-3), linestyle='--', color = 'k', label='$\pm$ 3 steady state error')
    plt.axhline(y=(goal+3), linestyle='--', color = 'k')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.show()
    plt.plot(time_lst, pulse_lst)
    plt.xlabel("Time (s)")
    plt.ylabel("Controller Duty Cycle (V)")
    plt.show()


    ### Fitness ###
    # Read data from the file
    with open('bbcFitness.dill', 'rb') as f:
        fitness_lst = dill.load(f)

    # Plot fitness per generation and average
    average_fitness = []
    for x in range(len(fitness_lst)):
        generation = []
        for i in range(len(fitness_lst[x])):
            generation.append(x)
        plt.scatter(generation, fitness_lst[x], color='darkgrey')
        average_fitness.append(np.mean(fitness_lst[x]))
    plt.plot(average_fitness, color='red')
    plt.show()


    print("Complete.")


def main():
    '''
    Main method.
    '''
    winner = run("bbcConfig.txt")
    with open('bbcWinner.dill', 'wb') as f:
            dill.dump(winner, f)
    show_winner(winner)
    eng.quit()

if __name__ == '__main__':
    main()


# def show_winner_test():
#     '''
#     Uncomment this, and comment out main method to run the winner again
#     '''
#     with open('bbcWinner.dill', 'rb') as f:
#         winner = dill.load(f)

#     global config
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                                 "bbcConfig.txt")

#     global eng
#     eng = matlab.engine.start_matlab()
#     eng.load_system('bbcSimNEAT', nargout=0)

#     show_winner(winner)
#     eng.quit()
# show_winner_test()