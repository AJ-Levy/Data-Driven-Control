## Q-Learning for Inverted Pendulum Control

### Files
There are five key files - including two Simulink models and three Python scripts.
1. `pendSimQTraining.slx` simulates the inverted pendulum starting at a supplied intial angle for a number of episodes.
2. `QLearningAgent.py` provides a training agent that implements the Q-Learning algorithm using an epsilon-greedy strategy and a system-specific reward function in order to build up an optimal Q-Table.
3. `pendSimQController.slx` simulates the inverted pendulum starting at a supplied intial angle a single time and saves all measurements to MATLAB's Workspace.
4. `QLearningController.py` provides a means of controlling the system through the use of a exclusively greedy strategy and the post-training Q-Table.
5. `QLearningMain.py` allows one to train the system for a specified number of episodes, supply the simulations with intial angles, run the trained controller, and plot the results. 

### Execution