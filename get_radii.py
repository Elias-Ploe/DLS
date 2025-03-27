import numpy as np
import matplotlib.pyplot as plt
import os
from autocorrelator import PlotManager, CurveFitting, autocorrelation_fft, particle_radius, radius_std


def binary_to_arr(file_path):

    with open(file_path, "rb") as file:
        binary_data = file.read()

    photon_counts = np.frombuffer(binary_data[9:], dtype=np.uint8) # remove timestamp or whatever is at the beginning (first 9 values)
    
    return photon_counts




def acf_from_binaryfiles(folder_path):
  
    
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.startswith('OUT') and f.endswith('.DAT')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))  # Extract numbers and sort numerically
    )
    if not file_names:
        raise ValueError('no files dummy')

    acf_folder = os.path.join(folder_path, 'acf')
    os.makedirs(acf_folder, exist_ok=True)

    for i, f in enumerate(file_names):
        file_path = os.path.join(folder_path, f)  # Path to input file
        save_path = os.path.join(acf_folder, f"acf_bin_{i}.dat")   # Path to output file

        print(f'cooking {f}:')

        sequence = binary_to_arr(file_path)
        acf_binned, bins = autocorrelation_fft(sequence, binning=1.03)


        stacked = np.column_stack((bins, acf_binned))
        np.savetxt(save_path, stacked, fmt='%e')



def analyse_all_acf(folder_path, model, error = True):

    file_names = sorted([f for f in os.listdir(folder_path) if f.startswith('acf_bin_') and f.endswith('.dat')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
        ) # this is extermely important
    data_acf = [np.loadtxt(os.path.join(folder_path, f), usecols = [1]) for f in file_names]
    data_bins = [np.loadtxt(os.path.join(folder_path, f), usecols = [0]) for f in file_names]



    all_tau = []
    all_std_dev = []
    for i, (acf, bins) in enumerate(zip(data_acf, data_bins)):
        try:
            print(f'fitting: {i}')
            fit = CurveFitting(model, acf, bins).make_fit()
            parameters = fit.get_params()

            if error is True:
                std_dev = fit.get_std_error()
                all_std_dev.append(std_dev)


            if model == 'frisken':
                all_tau.append(1/parameters[2])
            elif model == 'kww':
                all_tau.append(parameters[2])
            elif model == 'exp':
                all_tau.append(parameters[1])

        except Exception as e:
            print(f"Fit failed for {i}, reason: {e}")
            continue

    return np.array(all_tau), np.array(all_std_dev)




def plot_r(r, std_dev, error = True):
    fig, ax = plt.subplots(figsize=(6, 4))  

    if error:
        ax.errorbar(
            x=np.arange(len(r)),  
            y=r,                  
            yerr=std_dev,           # array of errors
            fmt='o',              
            markersize=3,
            markerfacecolor='none',
            color='cornflowerblue',
            ecolor='lightskyblue',  
            elinewidth=0.5,
            capsize=1.5,               # length of the error bar caps
            label='r'
        )

    elif not error:
        ax.plot(r, marker='o', linestyle='', markerfacecolor='none', 
                markersize=3, color='cornflowerblue', label='r')

    #ax.axhline(mean_r, color='orchid', linestyle='--', linewidth=2, alpha=0.8, 
    #        label=rf'$\bar{{r}} = {np.round(mean_r, 8)}$')

    ax.set_xlabel('Measurement', fontsize=12)
    ax.set_ylabel('Radius', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, frameon=True)

    plt.savefig('/home/elias/proj/_photon_correlation/all_r.png', dpi=300, bbox_inches='tight')
    plt.show()





def main_r(folder_path, time_step = 30e-6, model = 'kww', filter = 1e5):

    """Converts binary to .dat, calcuates acf and saves acf data inside subfolder in Folderpath.
       Then fits the chosen model onto each dataset inside the folder, calculates the particle radius and std deviations.
       Finally plots all radii for all datasets."""

    # get acf, tau and std_dev from dataset
    acf_folder = os.path.join(folder_path, 'acf')

    if not os.path.isdir(acf_folder):
        print('no acf data found. Peforming calculation')
        acf_from_binaryfiles(folder_path)
        print('Acf and tau calculated succesfully')

    taus, std_devs = analyse_all_acf(acf_folder, model)
    tau_path = os.path.join(folder_path, 'all_tau.txt')
    np.savetxt(tau_path, taus)
    

    # look for crazy tau values and filter
    for i, value in enumerate(taus):
        if value > filter:
            print("oh god... tau = ", value, 'Data = ', i+1)

    mask = (taus >= 0) & (taus < filter) 
    tau_masked = taus[mask]
    std_devs = std_devs[mask]

    # calculate r, std_dev, mean_r and plot all the data
    r = particle_radius(tau_masked * time_step)
    std_dev_r = radius_std(std_devs * time_step)
    #mean_r = np.mean(r)
    plot_r(r, std_dev_r)










# fix this :/
def walking_avg_r(acf_folder, time_step = 30e-6, model = 'kww', filter = 1e5):
    taus, std_devs = analyse_all_acf(acf_folder, model)

    for i, value in enumerate(taus):
        if value > filter:
            print("oh god... tau = ", value, 'Data = ', i+1)

    mask = (taus >= 0) & (taus < filter) 
    tau_masked = taus[mask]

    # calculate r, std_dev, mean_r and plot all the data
    r = particle_radius(tau_masked * time_step)

    N = len(r)
    avg_bin = 10
    all_avg_r = []

    j=0
    while j < N // avg_bin:
        avg_r = np.mean(r[j * avg_bin : (j + 1) * avg_bin])
        all_avg_r.append(avg_r)
        j += 1
    
    plt.plot(all_avg_r)
    plt.show()



path = '/home/elias/proj/_photon_correlation/data_24_03_thymol/'
main_r(path, model='kww')