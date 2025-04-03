import numpy as np


def calculate_mole_fractions(m_ethanol, m_water):
    M_ethanol = 46.07  # g/mol
    M_water = 18.015   # g/mol

    n_ethanol = m_ethanol / M_ethanol
    n_water = m_water / M_water

    n_total = n_ethanol + n_water

    x_ethanol = n_ethanol / n_total
    x_water = n_water / n_total

    return x_ethanol, x_water




def jouyban_acree_model(X1, X2, eta_1, eta_2, A0, A1, A2, T): # X: molefraction for each liquid, eta: viscocity, A: parameters from lstsq fit, T: temperature
    term_1 = X1 * np.log(eta_1)
    term_2 = X2 * np.log(eta_2) 
    term_3 = A0 *(X1*X2 / T)
    term_4 = A1 *(X1*X2 *(X1-X2) / T)
    term_5 = A2 *(X1*X2 *(X1-X2)**2 / T)

    ln_eta_m = term_1 + term_2 + term_3 + term_4 + term_5

    return ln_eta_m # log of mixture of viscocity





# data from measurements
mass_data = {
    14.3: {"water": 3.785, "ethanol": 0.187},
    15.3: {"water": 3.472, "ethanol": 0.127},
    20.3: {"water": 3.184, "ethanol": 0.290},
    24.3: {"water": 3.801, "ethanol": 0.197},
    27.3: {"water": 3.801, "ethanol": 0.197},
    3.4: {"water": 3.801, "ethanol": 0.197}
}

# model constant from literature (weird korean one)
A0 = 724.652
A1 = 729.357
A2 = 976.050


# viscocity Data from literature

viscosity_data = {
    293: {"water": 1.0030e-3, "ethanol": 1.1890e-3},
    298: {"water": 0.8914e-3, "ethanol": 1.0995e-3},
    303: {"water": 0.7982e-3, "ethanol": 1.0606e-3},
    308: {"water": 0.7202e-3, "ethanol": 0.9698e-3},
    313: {"water": 0.6531e-3, "ethanol": 0.8661e-3},
    318: {"water": 0.5964e-3, "ethanol": 0.7841e-3},
    323: {"water": 0.5467e-3, "ethanol": 0.7126e-3},
}


def get_viscocity(T, date):

    eta_water = viscosity_data[T]["water"]
    eta_ethanol = viscosity_data[T]["ethanol"]

    mole_frac_ethanol, mole_frac_water = calculate_mole_fractions(mass_data[date]['ethanol'], mass_data[date]['water'])

    ln_eta_m = jouyban_acree_model(mole_frac_water, mole_frac_ethanol, eta_water, eta_ethanol, A0, A1, A2, T)

    return np.exp(ln_eta_m)


    