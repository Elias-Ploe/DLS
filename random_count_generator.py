import numpy as np

config = {
    'number_bins': 10000,
    'time_step': 1,
    'tau': 100,
    'initial_prob': 0.5,
    'path': '/home/elias/proj/_photon_correlation/photon_counts'
}

def generate_correlated_binary(n, tau, initial_p, time_step):

    decay = 1 + np.exp(-(time_step / tau))


    probabilities = np.zeros(n)
    photon_sequence = np.zeros(n)

    probabilities[0] = initial_p

    for i in range(1, n):
        # term one makes prob strongly correlated for t << tau, term 2 does the opposite 
        probabilities[i] = decay * probabilities[i-1] + 1/40 * np.random.randn()

    for i in range(n):
        # this just takes random number for stochastic effect of single count measurement and compares to probs.
        photon_sequence[i] = 1 if np.random.rand() < probabilities[i] else 0

    return photon_sequence

photon_sequence = generate_correlated_binary(config['number_bins'], config['tau'], config['initial_prob'], config['time_step'])



np.save(f"{config['path']}.npy", photon_sequence)


