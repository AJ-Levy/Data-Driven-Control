# Data-Driven Control

This project compares the performance of data-driven control methods, Q-Learning and NEAT, with one another as well as traditional PID control. The methods are applied to non-linear binary control systems, specifically the inverted pendulum and DC-DC buck converters.

## Authors
**Matthew Fleischman**<br>
*University of Cape Town* <br>
*FLSMAT002@myuct.ac.za* 
<br>

**Ariel J. Levy**<br>
*University of Cape Town* <br>
*LVYARI002@myuct.ac.za*
___

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Installation

1. Ensure that MATLAB is installed on your system. It can be downloaded from [MathWorks](https://www.mathworks.com/products/matlab.html).

2. Clone this repository:
    ```bash
   git clone https://github.com/AJ-Levy/Data-Driven-Control.git
    ```

3. Place this directory within the MATLAB directory:
   ```bash
   mv Data-Driven-Control MATLAB/
   ```
   
4. Navigate to the project directory:
    ```bash
    cd MATLAB/Data-Driven-Control
    ```
    
5. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. Launch MATLAB and verify that the correct Python interpreter is being used with the following commands:
    ```MATLAB
    python_interpretor = '<path_to_python_version>';
    pyenv('Version', python_interpretor);
    ```
    
## Usage

To run the simulations for each control method on the respective systems, follow these steps:

### 1. Ensure all dependencies are installed. 
Make sure the necessary Python packages are installed as per the installation instructions.

### 2. Running Simulations
Each control method for both the inverted pendulum and DC-DC buck converter systems has a main Python file. To execute the simulation, simply run the corresponding file for the desired system and control method:

 - Inverted Pendulum

    - PID: Run `PIDMainPend.py`
    - Q-Learning: Run `QLearningMainPend.py`

- DC-DC Buck Converter

    - PID: Run `PIDMainBC.py`
    - Q-Learning: Run `QLearningMainBC.py`

### 3. Execution
Each main file is self-contained. Running the script will automatically start the simulation, displaying results like control method details and plots of system behaviour. 

To execute a file, use the command line or your preferred Python environment:
```bash
python <control_method>Main<system>.py
```

### 4. Adjustments

To modify the parameters of a specific control method, follow these instructions:

- PID

    Go to the appropriate `PIDController` file and change the values of `Kp`, `Ki`, and `Kd` in this line of code:
    ```python
    pid_controller = PIDController(Kp=x, Ki=y, Kd=z)
    ```

- Q-Learning

    Go to the appropriate `QLearningAgent` file and change the values of `alpha` and `gamma` in this line of code.
    ```python
    agent = QLearningAgent(alpha=s, gamma=t) 
    ```

Adjust these parameters according to your needs to fine-tune the performance of the control methods.

## Acknowledgements

We would like to express our gratitude to our supervisor, Prof. K. Prag, for her invaluable guidance and feedback throughout this research project.

  
