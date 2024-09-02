import numpy as np

class QLearningAgent:

    def __init__(self, num_actions=5):
        # files
        self.qfile = 'qtable_BC.npy'
        self.convergence_file = 'qconverge_BC.txt'
        # number of episodes
        self.total_episodes =1250
        # learning rate
        self.alpha = 0.01
        # discount factor
        self.gamma = 0.995
        # epsilon-greedy
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay_val = 0.995
        self.episode_threshold = self.total_episodes//10
        # defining q-table
        self.qtable = np.load(self.qfile)
        # last state and action
        self.last_state = None
        self.last_action = None
        self.actions = [0.1, 0.3, 0.5, 0.7, 0.9]
        # state parameters
        self.fail_state = -1
        self.num_actions = num_actions
        self.num_states =  24 # 0 - 71
        # track convergence by cumulative reward
        self.cum_reward = 0
        self.cum_rewards = []
        self.current_episode = 1
        # count number of iterations
        self.time_steps = 0
        # total time steps
        self.dt = 5e-6
        self.total_time = 0.3
        self.total_steps = self.total_time/self.dt

    def get_state(self, voltage):
        '''
        Convert continous parameters into discrete states
        '''
        if (voltage < -100 or voltage > 100):
            return self.fail_state
        
        # voltages
        if voltage < -60: box = 0
        elif voltage < -40: box = 1
        elif voltage < -30: box = 2
        elif voltage < -23: box = 3
        elif voltage < -16: box = 4
        elif voltage < -13: box = 5
        elif voltage < -10: box = 6
        elif voltage < -8: box = 7
        elif voltage < -5: box = 8
        elif voltage < -3: box = 9
        elif voltage < -1: box = 10
        elif voltage < 0: box = 11
        elif voltage < 1: box = 12
        elif voltage < 3: box = 13
        elif voltage < 5: box = 14
        elif voltage < 8: box = 15
        elif voltage < 10: box = 16
        elif voltage < 13: box = 17
        elif voltage < 16: box = 18
        elif voltage < 23: box = 19
        elif voltage < 30: box = 20
        elif voltage < 40: box = 21
        elif voltage < 60: box = 22
        else: box = 23
    
        return box
            

    def select_action(self, state, num_episodes):
        '''
        Selects next action using epsilon greedy strategy,
        returning the index of the action i.e. 0 or 1.
        '''
        if self.epsilon > self.min_epsilon and num_episodes > self.episode_threshold:
            self.decay_epsilon(num_episodes)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.qtable[state, :]) 
        
    def decay_epsilon(self, num_episodes):
        '''
        Epsilon decay is employed after some number of episodes
        to allow both exploration and optimisation.
        '''
        self.epsilon = np.power(self.epsilon_decay_val, num_episodes - self.episode_threshold)

    
    def update(self, state, action, reward, next_state, num_episodes):
        '''
        Implementation of QLearning governing equation
        '''
        q_old = self.qtable[state, action]
        q_new = reward + self.gamma * np.max(self.qtable[next_state])
        self.qtable[state, action] = q_old + self.alpha * (q_new - q_old)
        
        # collect convergence data
        rew = 0
        if state > 8 and state < 14:
            rew = 1.0

        if self.current_episode == num_episodes:
            self.cum_reward += rew
        else:
            self.cum_rewards.append(self.cum_reward)
            self.cum_reward = rew
            self.current_episode += 1
            self.time_steps = 0

        # save updated qtable and convergence data
        if num_episodes == self.total_episodes:
            np.save(self.qfile, self.qtable)
            
            with open(self.convergence_file, "w") as f:
                for i, reward in enumerate(self.cum_rewards):
                    f.write(f'{i}#{reward}\n')


    def get_output(self, voltage, ref_voltage, num_episodes):
        '''
        Applies QLearning algorithm to select an action
        and then return an appropriate output signal
        '''
        self.time_steps += 1

        state = self.get_state(voltage)
        if self.last_state is not None:
            reward = self.reward_function(voltage, ref_voltage)
            self.update(self.last_state, self.last_action, reward, state, num_episodes)
        action = self.select_action(state, num_episodes)
        self.last_state = state
        self.last_action = action
        duty_cycle = self.actions[action]

        return duty_cycle
        
    def reward_function(self, voltage, ref_voltage):
        '''
        simple for now
        '''
        
        
        # First Reward Function
        return 1/(1 + (voltage)**2)
        
        '''
        # Second (worse-performing) Reward Function
        v_out = abs(voltage - ref_voltage)
        if (v_out >= 3/4 * ref_voltage) and (v_out <= 5/4 * ref_voltage):
            return (abs(ref_voltage/4) - abs(voltage))**2
        return 0
        '''
        
        
# QLearning agent
agent = QLearningAgent()     

def controller_call(voltage, ref_voltage, num_episodes):
    '''
    Method that MATLAB calls for QLearning
    '''
    global agent
    output = agent.get_output(voltage, ref_voltage, num_episodes)
    return output