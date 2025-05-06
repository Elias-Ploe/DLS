import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from get_radii import get_real_time_axis

config = {
    'file_path': '/home/elias/proj/_photon_correlation/final_temp_30/all_r.txt',
    'real_time': 30e-6,
    'channels': 5000000,
}


def load_and_log(file_path):

    #load r and std dev and convert to mus -> s
    r = np.loadtxt(file_path, usecols=0) * 1e6
    std_dev = np.loadtxt(file_path, usecols=1) * 1e6
    t = get_real_time_axis(config['real_time'], config['channels'], r, 's')

    # convert to double log
    ln_r = np.log(r)
    ln_t = np.log(t)
    ln_dev = std_dev / r 

    return ln_r, ln_t, ln_dev



def plot_setup(ax):
    ax.set_xlabel('ln(t) [s]', fontsize=12)
    ax.set_ylabel(r'ln(r) [$\mathrm{\mu m}$]', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

def plot_ln(file_path):
    """file_path: column 0: r, column 1: std_dev
    plots ln(r(ln(t))) and shows plot """

    ln_r, ln_t, ln_dev = load_and_log(file_path)


    fig, ax = plt.subplots(figsize=(6, 4)) 
    
    
    ax.errorbar(
        x=ln_t, y=ln_r, yerr=ln_dev,
        fmt='none', ecolor='cornflowerblue', elinewidth=0.7, capsize=1, alpha=0.5, label=rf'$\sigma$'
    )

    ax.plot(
        ln_t, ln_r, 
        '.', markersize=3, color='cornflowerblue', label='r'  # Label for markers
    )   


    #ax.axhline(mean_r, color='orchid', linestyle='--', linewidth=2, alpha=0.8, 
    #        label=rf'$\bar{{r}} = {np.round(mean_r, 8)}$')

    
    plot_setup(ax)
    plt.show()




def fit_log_data(file_path):

    ln_r, ln_t, ln_dev = load_and_log(file_path)
    plot_ln(file_path)
    t_1 = float(input('t_1 for lienar fit:'))
    t_2 = float(input('t_2 for linear fit:'))

    

    lin_func = lambda t, k, d: k * t + d
    index_1 = np.argmin(np.abs(ln_t - t_1))
    index_2 = np.argmin(np.abs(ln_t - t_2))


    guess = [1, 1]
    popt, pcov = curve_fit(lin_func, ln_t[index_1:index_2], ln_r[index_1:index_2], sigma = ln_dev[index_1:index_2], absolute_sigma=True ,p0 = guess)
    k_dev = np.sqrt(np.diag(pcov)[0])
    
    t_linear = np.linspace(t_1, t_2, 100) # t for plot of fit
    fitted_func = lin_func(t_linear, *popt)

    return t_linear, fitted_func, k_dev, popt[0]


def main(file_path):

    ln_r, ln_t, ln_dev = load_and_log(file_path)
    t_linear, fit, k_dev, k = fit_log_data(file_path)



    fig, ax = plt.subplots(figsize=(6,4))

    #ax.errorbar(x=ln_t, y=ln_r, yerr=ln_dev, fmt='none', ecolor='gold', elinewidth=0.7, capsize=1, alpha=0.4)
    ax.plot(ln_t, ln_r, '.', markersize=3, color='gold', label='r') 
    ax.plot(t_linear, fit, color = 'cornflowerblue', alpha = 0.6, label = rf'$\mathrm{{lin\_fit}},\ k = {k:.4f} \pm {k_dev:.4f}$')
    ax.plot(t_linear, fit, color='cornflowerblue',alpha=0.6, marker='o',markevery=[0, -1],markersize=3)

    plot_setup(ax)
    plt.show()






if __name__ == "__main__":
    main(config['file_path'])