import numpy as np

class QLearningAgent:

    def __init__(self, num_actions=2):
        # files
        self.qfile = 'qtable_BBC.npy'
        self.convergence_file = 'qconverge_BBC.txt'
        # number of episodes
        self.total_episodes = 1000
        # learning rate
        self.alpha = 0.4
        # discount factor
        self.gamma = 0.99
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
        # forces to be applied to cart
        self.duty_cycles = [0, 1]
        # state parameters
        self.fail_state = -1
        self.num_actions = num_actions
        self.num_states = 144 # 0 - 17
        # track convergence by cumulative reward
        self.cum_reward = 0
        self.cum_rewards = []
        self.current_episode = 1
        # count number of iterations
        self.time_steps = 0

    def get_state(self, voltage):
        '''
        Convert continous parameters into discrete states
        (source: https://pages.cs.wisc.edu/~finton/qcontroller.html)
        '''
        pass
            

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
        rew = 0.0
        if state != self.fail_state:
            rew = 1.0

        if self.current_episode == num_episodes:
            self.cum_reward += rew
        else:
            self.cum_rewards.append(self.cum_reward/self.time_steps)
            self.cum_reward = rew
            self.current_episode += 1
            self.time_steps = 0

        # save updated qtable and convergence data
        if num_episodes == self.total_episodes:
            np.save(self.qfile, self.qtable)
            
            with open(self.convergence_file, "w") as f:
                for i, reward in enumerate(self.cum_rewards):
                    f.write(f'{i}#{reward}\n')


    def get_output(self, voltage, num_episodes):
        '''
        Applies QLearning algorithm to select an action
        and then return an appropriate output signal
        '''
        self.time_steps += 1

        state = self.get_state(voltage)
        if self.last_state is not None:
            reward = self.reward_function(state)
            self.update(self.last_state, self.last_action, reward, state, num_episodes)
        action = self.select_action(state, num_episodes)
        self.last_state = state
        self.last_action = action
        duty_cycle = self.duty_cycles[action] 

        return duty_cycle
        
    def reward_function(self, state):
        '''
        simple for now
        '''
        if state == self.fail_state:
            return 0.0
        return 1.0
        
        
# QLearning agent
agent = QLearningAgent()     

def controller_call(voltage, num_episodes):
    '''
    Method that MATLAB calls for QLearning
    '''
    global agent
    duty_cycle = agent.get_output(voltage, num_episodes)
    return duty_cycle