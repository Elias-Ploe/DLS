import numpy as np
import matplotlib.pyplot as plt
import os


config = {
    'folder_path': '/home/elias/proj/_photon_correlation/new_meas/0.077_T_15',
    'real_time': 30e-6,
    'channels': 5000000,
}







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
    ax.set_xlabel('t [h]')
    ax.set_ylabel('I [counts]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    return fig, ax




def get_real_time_axis(time_step, channels, measurements, unit):

    real_time = channels * time_step

    time_axis = []
    for i in range(len(measurements)):
        time_axis.append(real_time*(i+1))

    if unit == 'h':
        return np.array(time_axis) / 3600 #convert to hours
    if unit == 's':
        return np.array(time_axis)
    else:
        raise ValueError('choose unit: h or s')


def analyse_intensities(folder_path):
    """ Calcualte the intensity for all Measurements. The Data needs to be in binary format. """
    intensities = get_all_intensities(folder_path)
    t_values = get_real_time_axis(config['real_time'], config['channels'], intensities, 'h')
    fig, ax = plot_setup()
    ax.plot(t_values, intensities,
            marker='o', linestyle='', markersize=3, markerfacecolor='none', markeredgecolor='tomato',
            label = 'Intensity_data')
    ax.legend()

    save_path = os.path.join(folder_path, 'intensities.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def walking_avg_int():

    
    paths = ['/home/elias/proj/_photon_correlation/new_meas/0.076_may_long',
             '/home/elias/proj/_photon_correlation/for_plot_walk_avg/high_concentration/meas_4',
            '/home/elias/proj/_photon_correlation/for_plot_walk_avg/high_concentration/meas_5'
    ]
    colors = ['#44AA99','#117733', '#DDCC77']
    temperature = [0.19, 0.33, 0.45]

    fig, ax = plt.subplots(figsize=(6, 4))

    for conc, path, col in zip(temperature, paths, colors):

        intensities = get_all_intensities(path)

        N = len(intensities)
        avg_bin = 5
        all_avg_r = []

        j = 0
        while j < N // avg_bin:

            bin_data = intensities[j * avg_bin : (j + 1) * avg_bin]
            avg_r = np.mean(bin_data)
            all_avg_r.append(avg_r)
   
            j += 1

        all_avg_r = np.array(all_avg_r)
        t = get_real_time_axis(config['real_time'], config['channels'], all_avg_r, 'h') * avg_bin

        plt.plot(t, all_avg_r, color=col, label=f'{conc} wt. % thymol')
    

    ax.set_xlabel('t (h)', fontsize=12)
    ax.set_ylabel(r'I ($\mathrm{counts}$)', fontsize=12)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # <-- ADD THIS LINE
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/home/elias/proj/_photon_correlation/walk_avg_int.png', dpi=300)
    plt.show()
    

analyse_intensities(config['folder_path'])
#walking_avg_int()
