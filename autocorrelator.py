import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import norm






class PlotManager:

    # FIX STD DEV NONE IN PLOT INFO
    def __init__(self, acf_values, fit, parameters, std_dev, x_values = None):
        self.bins = x_values
        self.acf_values = acf_values
        self.fit = fit
        self.parameters = parameters
        self.x_values = x_values
        self.std_dev = std_dev

        if len(self.parameters) == 5:
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
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        
    def plot_acf(self, color = 'cornflowerblue', label_input = 'Acf Data'):

        if self.x_values is None:
            self.ax.plot(self.acf_values, marker='o', linestyle='', markersize=3, 
                     markerfacecolor='none', markeredgecolor=color, label=label_input)
        else: 
            self.ax.plot(self.x_values, self.acf_values, marker='o', linestyle='', markersize=3, 
                     markerfacecolor='none', markeredgecolor=color, label=label_input)


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
            self.ax.plot([], marker='', linestyle='',markerfacecolor='none' , label=fr"$\tau = \frac{{1}}{{\bar \Gamma}} = {np.round(1/gamma_bar_fit, 3)}$, $\gamma = \frac{{K_2^2}}{{\bar \Gamma^2}} = {np.round(K2_fit**2 / gamma_bar_fit**2, 8)}$", markeredgecolor = 'gray')


    def show_and_save(self, save_path = "/home/elias/proj/_photon_correlation/plot.png"):

        self.ax.legend()
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
        # ⟨I(t), I(t + t')⟩
        correlation = np.mean(sequence[:N-t] * sequence[t:])

        # g^(2) = ⟨I(t), I(t + t')⟩ / ⟨I(t), I(t + 0)⟩
        g2 = correlation / normalization
        g2_values.append(g2)

    return np.array(g2_values) 



def autocorrelation_fft(sequence, binning = None):

    N = len(sequence)
    normalization = np.mean(sequence)**2  # ⟨I(t)⟩^2
    pad = 2*N #add 0 to avoid convolution artifacts as explained by mr. Sepiol ;)
    sequence_fft = np.fft.fft(sequence, pad) #note first N will be sequence, 2nd N will be zeros!

    psd = sequence_fft * np.conjugate(sequence_fft) #power spectrum

    correlation_padded = 1/N * np.real(np.fft.ifft(psd)) # inverse fourer and take real 

    correlation = correlation_padded[:N]

    g2_values = correlation / normalization

    if binning is None:
        return g2_values 

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
        self.guess = self.models[model]['guess']
        
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


    
    def make_fit(self, cutoff = 10000):

        index = np.where(self.x_values >= cutoff)[0][0]

        self.popt, self.pcov, = curve_fit(self.func, self.x_values[1:index], self.y_values[1:index], p0 = self.guess)
        
        x_linear = np.arange(int(max(self.x_values)))
        self.fitted_func = self.func(x_linear, *self.popt)

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
            print('std_dev for frisken needs to be added, for now it is 1')
            return 1

    

def test_binning(path):
    sequence = np.loadtxt(path, usecols = [1])
    acf_data, bins = autocorrelation_fft(sequence, binning = 1.03)
    acf_data_unbinned = autocorrelation_fft(sequence)


    fitter_bin = CurveFitting('exp', acf_data, bins).make_fit()
    params = fitter_bin.get_params()
    std_error = fitter_bin.get_std_error()

    fitter = CurveFitting('exp', acf_data_unbinned).make_fit()
    param_no_bin = fitter.get_params()
    std_error_no_bin = fitter.get_std_error()


    plt.plot(acf_data_unbinned[1:], marker = '.', markerfacecolor = 'none', markersize = 2 ,linestyle = '', label = 'acf', color = 'gold')
    plt.plot(bins, acf_data, marker = 'o', markerfacecolor = 'none', markersize = 3, linestyle = '', label = 'acf_binned', color = 'cornflowerblue')
    
    print('w binning:', params[1], rf'$\pm$', std_error, 'no binning:', param_no_bin[1], rf'$\pm$', std_error_no_bin)




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
    return K * tau_std




def plot_distribution(gamma_bar, k2, time_step = 30e-6):

    # gamma_bar mean, k2 var, k3 skew
    radius = particle_radius(1/gamma_bar * time_step) * 1e9 # convert mean radius to nm
    #std_dev_gamma = np.sqrt(k2)
    #std_dev_tau = std_dev_gamma / gamma_bar**2 # standard error prog
    #std_dev = radius_std(std_dev_tau * time_step) * 1e9 # convert standard dev to nm
    std_dev = (np.sqrt(k2) / gamma_bar)  # this is implies a a lot (check Mailer 2015)

    x = np.linspace(radius - 4*std_dev, radius + 4*std_dev, 1000)
    y = norm.pdf(x, loc=radius, scale=std_dev)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color = 'cornflowerblue')
    
    # A vertical line for the mean
    plt.axvline(radius, linestyle='--', label=rf'Mean: {radius:.4g} $\pm$ {std_dev: .4g} nm', color = 'violet')  
    plt.axvline(radius - std_dev, linestyle=':', color='orchid')
    plt.axvline(radius + std_dev, linestyle=':', color='orchid')
    
    # Labels, title, grid, legend
    plt.xlabel('Radius (nm)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend(loc='best')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Layout and save
    plt.tight_layout()
    plt.savefig("/home/elias/proj/_photon_correlation/plot_distr.png", dpi=300)
    plt.show()



def main_acf(file_path, model, distr = False, isacf = False):

    if not isacf:
        sequence = np.loadtxt(file_path, usecols=[1])
        acf_values, bins = autocorrelation_fft(sequence, binning = 1.03)
    elif isacf:
        acf_values = np.loadtxt(file_path, usecols = [1])
        bins = np.loadtxt(file_path, usecols = [0])
    fitter_bin = CurveFitting(model, acf_values, bins).make_fit()
    fit = fitter_bin.get_curve()
    parameters = fitter_bin.get_params()
    std_dev = fitter_bin.get_std_error()

    plt = PlotManager(acf_values, fit, parameters, std_dev, bins)
    plt.set_up()
    plt.plot_acf()
    plt.plot_fit('violet')
    plt.plot_info()
    plt.show_and_save()

    if model == 'frisken' and distr is True:
        plot_distribution(parameters[2], parameters[3])




def analyse_all_acf(folder_path, model, error = True):

    file_names = [f for f in os.listdir(folder_path) if f.startswith('acf_bin_') and f.endswith('.dat')]
    data_acf = [np.loadtxt(os.path.join(folder_path, f), usecols = [1]) for f in file_names]
    data_bins = [np.loadtxt(os.path.join(folder_path, f), usecols = [0]) for f in file_names]



    all_tau = []
    all_std_dev = []
    for acf, bins in zip(data_acf, data_bins):
        try:
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
        except:
            continue

    return np.array(all_tau), np.array(all_std_dev)




def plot_r(r, std_dev , mean_r, error = False):
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

    ax.axhline(mean_r, color='orchid', linestyle='--', linewidth=2, alpha=0.8, 
            label=rf'$\bar{{r}} = {np.round(mean_r, 8)}$')

    ax.set_xlabel('Measurement', fontsize=12)
    ax.set_ylabel('Radius', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, frameon=True)

    plt.savefig('/home/elias/proj/_photon_correlation/all_r.png', dpi=300, bbox_inches='tight')
    plt.show()





def main_r(path, time_step = 30e-6, model = 'kww', filter = 1e5, data_is_acf = True, get_error = False):

    # calculate acf and determine tau for all datasets
    if data_is_acf:
        taus, std_devs = analyse_all_acf(path, model)
    
    if not data_is_acf:

        file_names = [f for f in os.listdir(path) if f.startswith('out_') and f.endswith('.dat')]

        if not file_names:
            print("No .dat files found in, files need to be of format: out_.dat", path)
            return 
        
        for i, f in enumerate(file_names):
            data_acf, data_bins = autocorrelation_fft(np.loadtxt(os.path.join(path, f), usecols=[1]), binning = 1.03)
            output_file = os.path.join(path, f"acf_bin_{i}.dat")
            np.savetxt(output_file, np.column_stack((data_bins, data_acf)), fmt="%e")

        taus, std_devs = analyse_all_acf(path, model, error = get_error)



    # look for crazy tau values and filter
    for i, value in enumerate(taus):
        if value > filter:
            print("oh god... tau = ", value, 'Data = ', i+1)

    mask = (taus >= 0) & (taus < filter) 
    tau_masked = taus[mask]
    
  
    std_devs = std_devs[mask]
    r = particle_radius(tau_masked * time_step)
    std_dev_r = radius_std(std_devs * time_step)
    mean_r = np.mean(r)
    plot_r(r, std_dev_r, mean_r, error = get_error)




def binary_to_arr(file_path):

    with open(file_path, "rb") as file:
        binary_data = file.read()

    photon_counts = np.frombuffer(binary_data[9:], dtype=np.uint8) # remove timestamp or whatever is at the beginning (first 9 values)
    
    return photon_counts




def acf_from_binaryfiles(folder_path):
  
    
    file_names = file_names = sorted(
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
            

#path = '/home/elias/proj/_photon_correlation/data_14_03_thymol'
#acf_from_binaryfiles(path)

# next up calculate the radius and so on from jaja, so adjust analyze all acf and main_r()


path = '/home/elias/proj/_photon_correlation/out_.dat'
test_binning(path)
#main_acf(path, 'exp')
#main_r(path, model = 'exp', filter = 1e10, data_is_acf = False, get_error = True)




# note bei bekannter autocorrtime kann man datensätze fenstern / splitten

# überlege ob vor fit nur plot für besserer guess



