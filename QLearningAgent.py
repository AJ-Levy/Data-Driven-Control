import numpy as np

class QLearningAgent:

    def __init__(self, num_actions=4):
        # files
        self.qfile = 'qtable.npy'
        self.convergence_file = 'qconverge.txt'
        # number of episodes
        self.total_episodes = 1500
        # learning rate
        self.alpha = 0.1
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
        self.forces = [10.0, 30.0, -10.0, -30.0]
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

    def get_state(self, theta, theta_dot):
        '''
        Convert continous parameters into discrete states
        (source: https://pages.cs.wisc.edu/~finton/qcontroller.html)
        '''
        theta = np.rad2deg(theta)
        theta_dot = np.rad2deg(theta_dot)
        box = 0

        # Failure state
        if theta < -60 or theta > 60:
            return self.fail_state
        
        # angles
        if (theta < -51): box = 0
        elif(theta < -46): box = 1
        elif (theta < -41): box = 2
        elif (theta < -36): box = 3
        elif (theta < -31): box = 4
        elif(theta < -26): box = 5
        elif (theta < -21): box = 6
        elif(theta < -16): box = 7
        elif (theta < -11): box = 8
        elif (theta < -6): box = 9
        elif (theta < -1): box = 10
        elif (theta < 0): box = 11
        elif (theta < 1): box = 12
        elif (theta < 6): box = 13
        elif (theta < 11): box = 14
        elif(theta < 16): box = 15
        elif (theta < 21): box = 16
        elif (theta < 26): box = 17
        elif(theta < 31): box = 18
        elif (theta < 36): box = 19
        elif (theta < 41): box = 20
        elif (theta < 46): box = 21
        elif(theta < 51): box = 22
        else: box = 23

        # angular velocities
        if (theta_dot < -50): pass
        elif (theta_dot < -25): box += 24
        elif (theta_dot < 0): box += 48
        elif (theta_dot < 25): box += 72
        elif (theta_dot < 50):  box += 96
        else: box += 120

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


    def get_force(self, theta, theta_dot, num_episodes):
        '''
        Applies QLearning algorithm to select an action
        and then apply a force to the cart
        '''
        self.time_steps += 1

        state = self.get_state(theta, theta_dot)
        if self.last_state is not None:
            reward = self.reward_function(theta, theta_dot, self.last_action)
            self.update(self.last_state, self.last_action, reward, state, num_episodes)
        action = self.select_action(state, num_episodes)
        self.last_state = state
        self.last_action = action
        force = self.forces[action] 

        return force
        
    def reward_function(self, theta, theta_dot, last_action):
        '''
        if state == self.fail_state:
            return 0.0
        return 1.0

        (source: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)
        '''
        # bonus for remaining close to equilibirum
        stability_bonus = 10 if abs(theta) < 0.05 and abs(theta_dot) < 0.1 else 0

        angle_penalty = theta**2 
        ang_velocity_penalty = 0.1 * theta_dot**2 
        action_penalty = 0.001 * self.forces[last_action]**2

        return -(angle_penalty + ang_velocity_penalty + action_penalty) + stability_bonus
        
        
# QLearning agent
agent = QLearningAgent()     

def controller_call(rad_big, theta_dot, num_episodes):
    '''
    Method that MATLAB calls for QLearning
    '''
    global agent
    # Normalize the angle (between -pi and pi)
    theta = (rad_big%(np.sign(rad_big)*2*np.pi))
    if theta >= np.pi:
        theta -= 2 * np.pi
    force = agent.get_force(theta, theta_dot, num_episodes)
    return force