import numpy as np
import matplotlib.pyplot as plt
import os
from autocorrelator import CurveFitting, autocorrelation_fft
import viscocity as vscy



config = {
    'folder_path': '/home/elias/proj/_photon_correlation/10C_thym',
    'real_time': 30e-6,
    'channels': 5000000,
    'model': 'kww',
    'T': 283,
    'date':10.6, # this assures the right mole fraction for viscocitiy are calculated
    'filter': 1e3
}






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
        acf_binned, bins = autocorrelation_fft(sequence, binning=1.01)


        stacked = np.column_stack((bins, acf_binned))
        np.savetxt(save_path, stacked, fmt='%e')




def fit_all_acf(folder_path, model, error = True):

    file_names = sorted([f for f in os.listdir(folder_path) if f.startswith('acf_bin_') and f.endswith('.dat')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
        ) # this is extermely important
    data_acf = [np.loadtxt(os.path.join(folder_path, f), usecols = [1]) for f in file_names]
    data_bins = [np.loadtxt(os.path.join(folder_path, f), usecols = [0]) for f in file_names]

    all_tau = []
    all_std_dev = []
    all_alpha = []
    for i, (acf, bins) in enumerate(zip(data_acf, data_bins)):
        try:
            print(f'fitting: {i}')
            fit = CurveFitting(model, acf, bins).make_fit()
            tau = fit.get_tau()

            if error is True:
                std_dev = fit.get_std_error()
                all_std_dev.append(std_dev)
            
            if model == 'kww':
                alpha = fit.get_params()[1]
                all_alpha.append(alpha)

            all_tau.append(tau)

        except Exception as e:
            print(f"Fit failed for {i}, reason: {e}")
            continue
    
    if model == 'kww':
        return np.array(all_tau), np.array(all_std_dev), np.array(all_alpha)
    
    else:
        np.array(all_tau), np.array(all_std_dev), np.array([])




def particle_radius(tau, viscocity = 0.932e-3, scattering_angle = (np.pi/2), wavelenght = 528e-9, refr_index = 1.333, T = 23 + 273.15):

    k_B = 1.380649e-23 
    q = ((4 * np.pi * refr_index / wavelenght)) * np.sin(scattering_angle/2)
    R = (q**2 * tau * k_B * T) / (6* np.pi * viscocity)
    return R




def radius_std(tau_std, viscocity=0.932e-3, scattering_angle=np.pi/2,
               wavelength=528e-9, refr_index=1.333, T=296.15):
    
    k_B = 1.380649e-23
    q   = (4 * np.pi * refr_index / wavelength) * np.sin(scattering_angle/2)
    # Derivative of R/tau:
    K   = (q**2 * k_B * T) / (6 * np.pi * viscocity)
    return  K * tau_std # sigma




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




def plot_r(r, std_dev):
    r = r*1e6      
    std_dev = std_dev*1e6
    fig, ax = plt.subplots(figsize=(6, 4))  
    x_vals = get_real_time_axis(config['real_time'], config['channels'], r, 'h')

    
    ax.errorbar(
        x=x_vals, y=r, yerr=std_dev,
        fmt='none', ecolor='cornflowerblue', elinewidth=0.7, capsize=1, alpha=0.5, label=rf'$\sigma$'
    )

    ax.plot(
        x_vals, r, 
        '.', markersize=3, color='cornflowerblue', label='r'  # Label for markers
    )   


    #ax.axhline(mean_r, color='orchid', linestyle='--', linewidth=2, alpha=0.8, 
    #        label=rf'$\bar{{r}} = {np.round(mean_r, 8)}$')

    ax.set_xlabel('t [h]', fontsize=12)
    ax.set_ylabel(r'r [$\mathrm{\mu m}$]', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_path = os.path.join(config['folder_path'], 'all_radii.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()






def main_r(folder_path, time_step = 30e-6, model = 'kww', filter = 1e3):

    """Converts binary to .dat, calcuates acf and saves acf data inside subfolder in Folderpath.
       Then fits the chosen model onto each dataset inside the folder, calculates the particle radius and std deviations.
       Finally plots all radii for all datasets."""

    # get acf, tau and std_dev from dataset
    acf_folder = os.path.join(folder_path, 'acf')

    if not os.path.isdir(acf_folder):
        print('no acf data found. Peforming calculation')
        acf_from_binaryfiles(folder_path)
        print('Acf and tau calculated succesfully')

    tau_path = os.path.join(folder_path, 'all_tau.txt')

    if not os.path.isfile(tau_path):
        taus, std_devs, alphas = fit_all_acf(acf_folder, model)
        data = np.column_stack((taus, std_devs))
        np.savetxt(tau_path, data)

    elif os.path.isfile(tau_path):
        print('using existing values')
        taus = np.loadtxt(tau_path, usecols=0)
        std_devs = np.loadtxt(tau_path, usecols=1)
    

    # look for crazy tau values and filter
    for i, value in enumerate(taus):
        if value > filter:
            print("oh god... tau = ", value, 'Data = ', i+1)

    mask = (taus >= 0) & (taus < filter) 
    tau_masked = taus[mask]
    std_devs = std_devs[mask]
    alphas = alphas[mask]

    # calculate r, std_dev, mean_r and plot all the data
    visc = vscy.get_viscocity(config['T'], config['date'])

    r = particle_radius(tau_masked * time_step, viscocity=visc, T=config['T'])
    std_dev_r = radius_std(std_devs * time_step, viscocity=visc, T=config['T'])

    plot_r(r, std_dev_r)


    r_path = os.path.join(folder_path, 'all_r.txt')
    stacked = np.column_stack((r, std_dev_r, alphas)) # save filtered r and tau to file for later use
    np.savetxt(r_path, stacked)








def walking_avg_r():

    base_path = '/home/elias/proj/_photon_correlation/plot_low_c_T_copy/meas_'
    temperature = [20, 15, 10]


    #base_path = '/home/elias/proj/_photon_correlation/for_plot_walk_avg/high_concentration/meas_'
    # meas_3: 0.19 meas_4: 0.33, meas_5 0.45 wt. % thymol, 
    #thymol_concentrations = [0.19, 0.33, 0.45]


    paths = [f"{base_path}{i}/all_r.txt" for i in range(1, 4)]
    #colors = ['#88CCEE', '#44AA99', '#117733', '#DDCC77', '#EE9966', '#CC6677', '#CC6677']
    #colors = ['#44AA99','#117733', '#DDCC77']
    colors = ['#CC6677', '#DDCC77', '#44AA99']
    

    fig, ax = plt.subplots(figsize=(6, 4))

    for conc, path, col in zip(temperature, paths, colors):
        r = np.loadtxt(path, usecols=0)*1e6
        r_dev = np.loadtxt(path, usecols=1)*1e6
        r = r[20:]
        r_dev = r_dev[20:]

        N = len(r)
        avg_bin = 5
        all_avg_r = []
        all_std_r = []

        j = 0
        while j < N // avg_bin:
            bin_data = r[j * avg_bin : (j + 1) * avg_bin]
            bin_errs = r_dev[j * avg_bin : (j + 1) * avg_bin]

            avg_r = np.mean(bin_data)

            # Combined error per bin:
            # 1. Measurement errors
            # 2. Statistical std
            # Total bin uncertainty (gauß) = sqrt( sum(meas_errors^2) / n^2 + std^2 / n )
            stat_std = np.std(bin_data, ddof=1)
            meas_var = np.sum(bin_errs**2) / avg_bin**2
            stat_var = stat_std**2 / avg_bin
            total_std = np.sqrt(meas_var + stat_var)

            all_avg_r.append(avg_r)
            all_std_r.append(total_std)
            j += 1

        all_avg_r = np.array(all_avg_r)
        all_std_r = np.array(all_std_r)

        t = get_real_time_axis(config['real_time'], config['channels'], all_avg_r, 'h') * avg_bin

        plt.plot(t, all_avg_r, color=col, label=f'{conc} °C')
        plt.fill_between(t, all_avg_r - all_std_r, all_avg_r + all_std_r,
                         color=col, alpha=0.2)
    

    ax.set_xlabel('t (h)', fontsize=12)
    ax.set_ylabel(r'r ($\mathrm{\mu m}$)', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/home/elias/proj/_photon_correlation/walk_avg.png', dpi=300)
    plt.show()



if __name__ == "__main__":

    #main_r(config['folder_path'], config['real_time'], config['model'], config['filter'])
    walking_avg_r()