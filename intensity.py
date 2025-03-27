import numpy as np
import matplotlib.pyplot as plt
import os



def get_intensity(sequence):

    intensity = np.sum(sequence)
    return intensity


def binary_to_arr(file_path):

    with open(file_path, "rb") as file:
        binary_data = file.read()

    photon_counts = np.frombuffer(binary_data[9:], dtype=np.uint8) # remove timestamp or whatever is at the beginning (first 9 values)
    
    return photon_counts


def get_all_intensities(folder_path):
    file_names = sorted(
        [f for f in os.listdir(folder_path) 
        if f.startswith('OUT') 
        and f.endswith('.DAT')
        and os.path.isfile(os.path.join(folder_path, f))],
        key=lambda x: int(''.join(filter(str.isdigit, x)))  # Extract numbers and sort numerically
    )


    all_intensities = []
    for f in file_names:
        file_path = os.path.join(folder_path, f) 

        print(f'Making sequence: {f}')

        sequence = binary_to_arr(file_path)
        intensity = get_intensity(sequence)
        all_intensities.append(intensity)

    return all_intensities


def plot_setup():
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlabel(r"$t'$")
    ax.set_ylabel(r"$I(t)$")
    ax.tick_params(axis='both', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    return fig, ax



def analyse_intensities(folder_path):
    """ Calcualte the intensity for all Measurements. The Data needs to be in binary format. """
    intensities = get_all_intensities(folder_path)
    t_values = np.arange(len(intensities))
    fig, ax = plot_setup()
    ax.plot(t_values, intensities,
            marker='o', linestyle='', markersize=3, markerfacecolor='none', markeredgecolor='tomato',
            label = 'Intensity_data')
    ax.legend()
    plt.show()
    
path = '/home/elias/proj/_photon_correlation/data_24_03_thymol/'
analyse_intensities(path)
    

