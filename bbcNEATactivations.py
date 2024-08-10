def square(x):
    import numpy as np
    from scipy import signal
    return 0.5*signal.square(2 * np.pi * 5 * x) + 0.5


def get_functions():
    return [("square_", square)]
