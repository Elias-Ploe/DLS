import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import norm
import viscocity as vscy



config = {
    'file_path': '/home/elias/proj/_photon_correlation/acf_bin_0.dat', # path to acf data
    'model': 'kww',                 # exp, kww, frisken or None for no fit
    'acf_color': 'cornflowerblue',  # matplot colors
    'fit_color': 'dimgray',

#--------- experimental data -------

    'real_time': 30e-6,
    'channels': 5000000,
    'T': 293,
    'date': 20.3, # this assures the right mole fraction for viscocitiy are calculated
}





class PlotManager:

    # FIX STD DEV NONE IN PLOT INFO
    def __init__(self, acf_values, fit, parameters, std_dev, x_values = None):
        self.bins = x_values
        self.acf_values = acf_values
        self.fit = fit
        self.parameters = parameters
        self.x_values = x_values
        self.std_dev = std_dev
        
        if self.parameters is None:
            self.fittype = None
        elif len(self.parameters) == 5:
            self.fittype = 'frisken'
        elif len(self.parameters) == 3:
            self.fittype = 'kww'
        else:
            self.fittype = 'exp'


    def set_up(self, figsize=(6, 4), log_ax_x = True):
        """Sets up the plot with basic configurations like figure size, axes, labels."""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlabel(r"$t'$")
        if log_ax_x:
            self.ax.set_xscale('log')
            self.ax.set_xlim(0.9, 10000)
        self.ax.set_ylim(0.9, 1.8)
        self.ax.set_ylabel(r"$g^{(2)}(t')$")
        self.ax.tick_params(axis='both', labelsize=7)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        
    def plot_acf(self, color = 'cornflowerblue', label_input = 'Acf Data'):

        if self.x_values is None:
            self.ax.plot(self.acf_values, marker='o', linestyle='', markersize=3, 
                     markerfacecolor='none', markeredgecolor=color, label=label_input)
        else: 
            self.ax.plot(self.x_values, self.acf_values, marker='o', linestyle='', markersize=3, 
                     markerfacecolor='none', markeredgecolor=color, label=label_input)
        
        self.ax.plot(0, linestyle = ':', linewidth = 0.7, alpha = 0.7, label = 'Residuals', color = 'blue')


    def plot_fit(self, color = 'orange'):

        if self.fittype is None:
            raise ValueError('specify fit type dumbo')
        
        if self.fittype == 'exp':
            self.ax.plot(self.fit, label=r"$1 + \beta e^{-2 (\frac{t'}{\tau})}$", color= color)
        elif self.fittype == 'kww':
            self.ax.plot(self.fit, label=r"$1 + \beta e^{-2 (\frac{t'}{\tau})^\alpha}$", color= color)
        elif self.fittype == 'frisken':
            self.ax.plot(self.fit, label=r"$B + \beta \exp\left(-2 \bar \Gamma t \right) \left(1 + K_2 t^2 - \frac{K_3}{3} t^3  \right)^2$", color= color) 


    def plot_info(self, info_string=None):
            
        if self.fittype == 'exp':
            self.ax.plot([], marker='', linestyle='',markerfacecolor='none' , label=rf"$\beta = {np.round(self.parameters[0], 2)}$, $\tau = {np.round(self.parameters[1], 2)} \pm {np.round(self.std_dev, 2)}$", markeredgecolor = 'gray')
        elif self.fittype == 'kww':
            self.ax.plot([], marker='', linestyle='',markerfacecolor='none' ,label=rf"$\beta = {np.round(self.parameters[0], 2)}$, $\alpha = {np.round(self.parameters[1], 2)}$, $\tau = {np.round(self.parameters[2], 2)} \pm {np.round(self.std_dev, 2)}$", markeredgecolor = 'gray')
        elif self.fittype == 'frisken':
            gamma_bar_fit = self.parameters[2]
            K2_fit = self.parameters[3]
            self.ax.plot([], marker='', linestyle='',markerfacecolor='none' , label=fr"$\tau = \frac{{1}}{{\bar \Gamma}} = {np.round(1/gamma_bar_fit, 3)}$, $\gamma = \frac{{K_2}}{{\bar \Gamma^2}} = {np.round(K2_fit / gamma_bar_fit**2, 5)}$", markeredgecolor = 'gray')
    
    def plot_residuals(self, x_res, y_res):
        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel(r'$\varepsilon$', fontsize=12)
        self.ax2.set_ylim(-0.04, 0.04)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.tick_params(axis='both', labelsize=7)

        self.ax2.plot(x_res, y_res, linestyle = ':', linewidth = 0.7, alpha = 0.7)


    def show_and_save(self, save_path = "/home/elias/proj/_photon_correlation/plot.png"):

        self.ax.legend(loc='upper right')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()




def generate_correlated_binary(n, tau, initial_p, time_step):

    decay = np.exp(-(2*time_step / tau))


    probabilities = np.zeros(n)
    photon_sequence = np.zeros(n)

    probabilities[0] = initial_p

    for i in range(1, n):
        # term one makes prob strongly correlated for t << tau and adds noise 
        probabilities[i] = decay * probabilities[i-1] + 1/500 * np.random.randn()

    for i in range(n):
        # this just takes random number for stochastic effect of single count measurement and compares to probs.
        photon_sequence[i] = 1 if (0.7) *np.random.rand() < probabilities[i] else 0

    return photon_sequence



def autocorrelation(sequence):

    N = len(sequence)
    normalization = np.mean(sequence)**2  # ⟨I(t)⟩^2
    g2_values = []

    for t in range(N-1):  
        # ⟨I(t), <I(t + t')>
        correlation = np.mean(sequence[:N-t] * sequence[t:])

        # g^(2) = <I(t), I(t + t')> / <I(t)>^2
        g2 = correlation / normalization
        g2_values.append(g2)

    return np.array(g2_values) 



def autocorrelation_fft(sequence, binning = None):

    N = len(sequence)
    normalization = np.mean(sequence)**2  # ⟨I(t)⟩^2
    pad = 2*N #add 0 to avoid convolution artifacts as explained by mr. Sepiol ;)
    sequence_fft = np.fft.fft(sequence, pad) #note first N will be sequence, 2nd N will be zeros!

    psd = sequence_fft * np.conjugate(sequence_fft) 

    correlation_padded = 1/N * np.real(np.fft.ifft(psd)) # inverse fourer and take real 

    correlation = correlation_padded[:N]

    g2_values = correlation / normalization

    if binning is None:
        return g2_values, np.arange(len(g2_values))

    #binning

    start = 1
    ratio = binning
    bins = [start]

    while bins[-1] < N:
        bins.append(bins[-1] * ratio)

    bins = np.unique(np.array(bins).astype(int))

    binned_g2 = np.zeros(len(bins) - 1)
    bin_centers = np.zeros(len(bins) - 1)

    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i + 1]
        binned_g2[i] = np.mean(g2_values[start:end])  
        bin_centers[i] =  (end + start) / 2.0


    return binned_g2, bin_centers - 3/2 # start at 0




class CurveFitting:

    def __init__(self, model, acf, bins = None):

        self.model_name = model
        
        self.models = {
            "exp": {"func": self.exp_fit, "guess": [0.5, 50]}, # beta, tau,
            "frisken": {"func": self.frisken_fit, "guess": [1, 1, 0.1, 0.001, 0.001]}, # B, beta, gamma_bar, mu2, mu3
            "kww": {"func": self.kww_fit, "guess": [0.5, 1, 50]}, # beta, alpha, tau
        }

        if model not in self.models:
            raise ValueError(f"{model} not found. Choose from {list(self.models.keys())}")

        self.y_values = acf
        self.func = self.models[model]['func']
        self.guess = self.models[model]['guess'] # this will be updated in a loop
        
        if bins is not None:
            self.x_values = bins

        elif bins is None: 
            self.x_values = np.arange(len(self.y_values))

        self.popt = None
        self.pcov = None
        self.fitted_func = None


    @staticmethod
    def exp_fit(t, beta, tau):
        return 1 + beta * np.exp(-2 * (t / tau))

    
    @staticmethod
    def frisken_fit(t, B, beta, gamma_bar, Mu2, Mu3):
        return B + beta * np.exp(-2*gamma_bar*t) * (1 + Mu2/2 * t**2 - Mu3/6 * t**3)**2
    
    @staticmethod

    def kww_fit(t, beta, alpha, tau):
        return 1 + beta * np.exp(-2 * ((t / tau) ** alpha))


    
    def make_fit(self):

        original_amplitude = self.y_values[0]
        cutoff = (original_amplitude / 100) +1
        index = np.where(self.y_values <= cutoff)[0][0] + 1

        #manual cutoff:
        #cutoff = 10000
        #index = np.where(self.x_values >= cutoff)[0][0]

        if self.model_name == "frisken":
        # add bounds to frisken
            lower_bounds = [0.99, 0, 0, 0, 0]
            upper_bounds = [1.01, np.inf, np.inf, np.inf, np.inf]
            bounds = (lower_bounds, upper_bounds)
            self.popt, self.pcov = curve_fit(self.func, self.x_values[0:index], self.y_values[0:index],
                                            p0=self.guess)
        else:
            self.popt, self.pcov = curve_fit(self.func, self.x_values[0:index], self.y_values[0:index],
                                            p0=self.guess)
            
        x_linear = np.arange(int(max(self.x_values)))
        self.fitted_func = self.func(x_linear, *self.popt)

        self.models[self.model_name]['guess'] = list(self.popt)
        self.guess = list(self.popt)
        print(f"Cutoff at index {index}, t = {self.x_values[index]:.2f}, g2 = {self.y_values[index]:.3f}")

        return self

    
    def get_curve(self):
        return self.fitted_func

    def get_params(self):
        return self.popt

    def get_std_error(self):

        if self.model_name == 'exp':
            return np.sqrt(np.diag(self.pcov)[1])
        
        if self.model_name == 'kww':
            return np.sqrt(np.diag(self.pcov)[2])
        
        if self.model_name == 'frisken':
            return np.sqrt(self.popt[3]) / self.popt[2]
    
    def get_tau(self):
        
        if self.model_name == 'exp':
            return self.popt[1]
        
        if self.model_name == 'kww':
            return self.popt[2]
        
        if self.model_name == 'frisken':
            return 1/self.popt[2]
        
    def get_residuals(self):
        if self.popt is None:
            raise ValueError("Fit the curve first using make_fit().")
        
        cutoff = (self.y_values[0] / 100) + 1
        index = np.where(self.y_values <= cutoff)[0][0] + 1
        
        x_residuals = self.x_values[:index]  # <-- REAL time axis
        y_pred = self.func(x_residuals, *self.popt)
        residuals = self.y_values[:index] - y_pred
    
        return x_residuals, residuals
    



def particle_radius(tau, viscocity = 1.0016e-3, scattering_angle = (np.pi/2), wavelenght = 528e-9, refr_index = 1.333, T = 293):

    k_B = 1.380649e-23 
    q = ((4 * np.pi * refr_index / wavelenght)) * np.sin(scattering_angle/2)
    R = (q**2 * tau * k_B * T) / (6* np.pi * viscocity)
    return R




def radius_std(tau_std, viscocity=1.0016e-3, scattering_angle=np.pi/2,
               wavelength=528e-9, refr_index=1.333, T=293):
    k_B = 1.380649e-23
    q   = (4 * np.pi * refr_index / wavelength) * np.sin(scattering_angle/2)
    # Derivative of R/tau:
    K   = (q**2 * k_B * T) / (6 * np.pi * viscocity)
    return K * tau_std




def binary_to_arr(file_path):

    with open(file_path, "rb") as file:
        binary_data = file.read()

    photon_counts = np.frombuffer(binary_data[9:], dtype=np.uint8) # remove timestamp or whatever is at the beginning (first 9 values)
    
    return photon_counts




def single_acf_from_binary(file_path):

    sequence = binary_to_arr(file_path)
    #sequence = np.loadtxt(file_path, usecols = [1])
    acf_binned, bins = autocorrelation_fft(sequence, binning=1.10) # 1.01!!
    
    return acf_binned, bins



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
        acf_binned, bins = autocorrelation_fft(sequence, binning=1.03) #1.03!!!!


        stacked = np.column_stack((bins, acf_binned))
        np.savetxt(save_path, stacked, fmt='%e')



def main_acf(file_path, model):

    print('Calculating ACF')
    acf_values, bins = single_acf_from_binary(file_path)
    
    if model is not None:
        print('Fitting Model:', model)
        fitter_bin = CurveFitting(model, acf_values, bins).make_fit()
        fit = fitter_bin.get_curve()
        parameters = fitter_bin.get_params()
        std_dev = fitter_bin.get_std_error()
    else:
        print('Skipping Fit — Plotting ACF Only')
        fit = None
        parameters = None
        std_dev = None

    plotter = PlotManager(acf_values, fit, parameters, std_dev, bins)
    plotter.set_up()
    plotter.plot_acf(color=config['acf_color'])

    if model is not None:
        plotter.plot_fit(config['fit_color'])
        plotter.plot_info()

    plotter.show_and_save()





# ------------- other stuff ----------

def plot_many_acf():
    """path_list = ['/home/elias/proj/_photon_correlation/concentration_formation/0.052g/OUT15.DAT',
                 '/home/elias/proj/_photon_correlation/concentration_formation/0.072g/OUT20.DAT',
                 '/home/elias/proj/_photon_correlation/concentration_formation/0.108g/OUT20.DAT',
                 '/home/elias/proj/_photon_correlation/concentration_formation/0.151g/OUT15.DAT',
                 '/home/elias/proj/_photon_correlation/concentration_formation/0.202g/OUT22.DAT',
                 '/home/elias/proj/_photon_correlation/concentration_formation/0.312g/OUT25.DAT'
    ]"""

    path_list = ['/home/elias/proj/_photon_correlation/new_meas/0.076g_may/OUT10.DAT',
                 '/home/elias/proj/_photon_correlation/new_meas/0.100g_may/OUT15.DAT',
                 '/home/elias/proj/_photon_correlation/concentration_sonication_formation/0.151/OUT15.DAT',
                 '/home/elias/proj/_photon_correlation/concentration_sonication_formation/0.297/OUT15.DAT',
                 ]
    
    """path_list = ["/home/elias/proj/_photon_correlation/dif_sonication_low_c/0.074g_8min/OUT15.DAT",
                 "/home/elias/proj/_photon_correlation/dif_sonication_low_c/0.072g_16min/OUT15.DAT",
                 "/home/elias/proj/_photon_correlation/dif_sonication_low_c/0.073_32min/OUT15.DAT"]"""

    colors = ['#88CCEE', '#44AA99', '#117733', '#DDCC77', '#EE9966', '#CC6677']

    first_acf, first_bins = single_acf_from_binary(path_list[0])
    plotter = PlotManager(first_acf, None, None, None, first_bins)
    plotter.set_up()
    #plotter.plot_acf(color=colors[0], label_input='concentration_1')

    #thymol_percentages = [0.14, 0.18, 0.27, 0.39, 0.51, 0.81]
    thymol_percentages = [0.19, 0.26, 0.40, 0.76]

    for conc, path, color in zip(thymol_percentages, path_list[:], colors[:]):
        print(f'Processing {path}')
        acf_values, bins = single_acf_from_binary(path)
        plotter.acf_values = acf_values
        plotter.x_values = bins
        plotter.plot_acf(color=color, label_input=f'{round(conc, 2)} wt. %')

    plotter.show_and_save()




def print_radius(file_path, model, time_step, temp, date):

    acf_values, bins = single_acf_from_binary(file_path)
    fitter_bin = CurveFitting(model, acf_values, bins).make_fit()
    fit = fitter_bin.get_curve()
    parameters = fitter_bin.get_params()
    std_dev = fitter_bin.get_std_error()
    tau = fitter_bin.get_tau()

    visc = vscy.get_viscocity(temp, date)

    r = particle_radius(tau * time_step, T=temp, viscocity=visc)
    std_dev_r = radius_std(std_dev * time_step, T=temp, viscocity=visc)


    print(rf'{r} $\pm$ {std_dev_r}')
    




def test_binning(path):
    sequence = np.loadtxt(path, usecols = [1])
    acf_data, bins = autocorrelation_fft(sequence, binning = 1.01)
    acf_data_unbinned = autocorrelation_fft(sequence)


    fitter_bin = CurveFitting('exp', acf_data, bins).make_fit()
    params = fitter_bin.get_params()
    std_error = fitter_bin.get_std_error()

    fitter = CurveFitting('exp', acf_data_unbinned).make_fit()
    param_no_bin = fitter.get_params()
    std_error_no_bin = fitter.get_std_error()


    plt.plot(acf_data_unbinned[1:], marker = '.', markerfacecolor = 'none', markersize = 2 ,linestyle = '', label = 'acf', color = 'gold')
    plt.plot(bins, acf_data, marker = 'o', markerfacecolor = 'none', markersize = 3, linestyle = '', label = 'acf_binned', color = 'cornflowerblue')
    plt.xscale('log')
    plt.gca().set_xlim(0.9, 10000)
    plt.gca().set_ylim(0.9, 1.8)
    plt.gca().set_xlabel(r"$t'$")
    plt.gca().set_ylabel(r"$g^{(2)}(t')$")
    plt.gca().tick_params(axis='both', labelsize=7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().legend()

    plt.savefig('/home/elias/proj/_photon_correlation/binnobin.png', dpi=300)
    plt.show()
    
    print('w binning:', params[1], rf'$\pm$', std_error, 'no binning:', param_no_bin[1], rf'$\pm$', std_error_no_bin)


def plot_different_fits(file_path, model_list):
    print('Calculating ACF')
    #acf_values, bins = single_acf_from_binary(file_path)
    acf_values, bins = np.loadtxt(file_path, usecols= 1), np.loadtxt(file_path, usecols= 0)
    # Initialize plotter only once
    plotter = None

    for i, model in enumerate(model_list):
        print('Fitting Model:', model)
        fitter_bin = CurveFitting(model, acf_values, bins).make_fit()
        fit = fitter_bin.get_curve()
        parameters = fitter_bin.get_params()
        std_dev = fitter_bin.get_std_error()

        # Set up plotter on first iteration
        if i == 0:
            plotter = PlotManager(acf_values, fit, parameters, std_dev, bins)
            plotter.set_up()
            plotter.plot_acf(label_input="ACF Data", color='darkgray')  # Plot raw ACF once

        # Update PlotManager with new fit info (temporarily)
        plotter.fit = fit
        plotter.parameters = parameters
        plotter.std_dev = std_dev
        plotter.fittype = model

        plot_colors = ['#555555', 'gold', '#729ECE']

        plotter.plot_fit(color=plot_colors[i])

    plotter.show_and_save()
    

def plot_residuals(file_path, model):

    acf_values, bins = np.loadtxt(file_path, usecols= 1), np.loadtxt(file_path, usecols= 0)
    
    print('Fitting Model:', model)
    fitter_bin = CurveFitting(model, acf_values, bins).make_fit()
    fit = fitter_bin.get_curve()
    parameters = fitter_bin.get_params()
    std_dev = fitter_bin.get_std_error()
    resx, resy = fitter_bin.get_residuals()

    plotter = PlotManager(acf_values, fit, parameters, std_dev, x_values = bins)
    plotter.set_up()
    plotter.plot_acf(color=config['acf_color'])
    plotter.plot_fit(config['fit_color'])
    plotter.plot_info()
    plotter.plot_residuals(resx, resy)

    plotter.show_and_save()




if __name__ == "__main__":
    #main_acf(config['file_path'], config['model'])
    #print_radius(config['file_path'], config['model'], config['real_time'], config['T'], config['date'])
    #viscocity = vscy.get_viscocity(config['T'], config['date'])
    #plot_many_acf()
    #plot_different_fits(config['file_path'], ['exp', 'kww', 'frisken'])
    plot_residuals(config['file_path'], 'frisken')