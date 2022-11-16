tau = 0.75

def square_wave(t):
    return signal.square(t)

def rectangular_wave(t):
    return signal.square(t, duty = tau)

coefficent_calculator(rectangular_wave, 2*pi, 99, 1, 6*pi)
