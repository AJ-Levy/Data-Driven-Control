import matlab.engine
import pickle
import matplotlib.pyplot as plt
import os

def main():

    # Remove old data
    if os.path.exists("anglePID.pkl"):
        os.remove("anglePID.pkl")
    if os.path.exists("timePID.pkl"):
        os.remove("timePID.pkl")
    
    # Run sim
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    print("Running simulation...")
    eng.sim('pendSimPID.slx')
    print("Simulation complete")

    # Load angles
    angles = []
    with open('anglePID.pkl', 'rb') as f:
        try:
            while True:
                angles.append(pickle.load(f))
        except EOFError:
            pass
        
    # Load time
    time = []
    with open('timePID.pkl', 'rb') as f:
        try:
            while True:
                time.append(pickle.load(f))
        except EOFError:
            pass

    # Plot data
    plt.plot(time, angles)
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.title("Angle of pendulum over time")
    plt.show()

if __name__ == '__main__':
    main()

# # Should work but doesnt
# answer = eng.workspace['out.angle']
# print(answer)