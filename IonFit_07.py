#!/usr/bin/env python
"""
IonFit - A Comprehensive Electrolyte Properties Analyzer

Version 0.6 20231021

by: Dr. Alessandro Mariani
Partially based on the work of: Dr. Alessandro Innocenti

History:

v 0.7
- The code now also produce the Walden Ionicity as an output
- The temperature-dependent correction factor is now more conform to VTF theory
- Bug fixes

v 0.6
- Introduced the ability for the user to define the chemical composition of the sample directly in the config.ini file
- Bug fixes and performance improvements

v 0.5
- Simplified user input with a single `config.ini` file
- Allowed for different units for the input data
- Modified the restricted VTF fitting to be more scientifically acceptable
- Walden plot procedure updated to handle also non purely ionic systems

v 0.4
- Enabled user specification of the initial guess for the VTF fitting
- Introduced options for three different Temperature-dependent Scale Factors in the VTF equation

v 0.3
- Adjusted for unique output directory and refined VTF fitting
- Users can now specify the maximum and minimum temperatures for predicted values
- General bug fixes

v 0.2
- Code refactored, improved, and extensively documented
- Now displays and saves the relevant plots obtained

v 0.1
- Initial version by joining WonderVTF and easyWalden
"""
import logging # for logging information, warnings, and errors
import os # for interacting with the operating system (e.g., path operations)
import argparse # for parsing command line arguments
import configparser # for parsing configuration files
import numpy as np # for numerical operations
from scipy.optimize import curve_fit, minimize # from scipy.optimize, for fitting data to a model
from sklearn.metrics import mean_squared_error, r2_score # for evaluating the quality of fits
import matplotlib.pyplot as plt # for creating plots
import datetime # for working with dates and times
from typing import List, Dict, Union, Tuple

# Set up logging to output information to a file with a severity level of INFO
logging.basicConfig(filename='IonFit.log', filemode='a', level=logging.INFO)

# Helping functions

def convert_units(value, unit_option, data_type):
    """
    Convert a value to different units based on the specified unit option and data type.

    Parameters:
    - value: The numerical value to be converted.
    - unit_option: An integer indicating which unit conversion to perform.
    - data_type: A string indicating the type of data (e.g., "viscosity", "conductivity", "density").

    Returns:
    - The converted value.
    """
    conversions = {
        "viscosity": {
            1: 1.0,  # mPa s to mPa s
            2: 1000.0,  # Pa s to mPa s
            3: 1.0,  # cP to mPa s
            4: 100.0  # P to mPa s
        },
        "conductivity": {
            1: 1.0,  # mS/cm to mS/cm
            2: 1000.0  # S/cm to mS/cm
        },
        "density": {
            1: 1.0,  # g/cm3 to g/cm3
            2: 0.001  # kg/m3 to g/cm3
        }
    }
    
    try:
        conversion_factor = conversions[data_type][unit_option]
        logging.info("Convertion of %s with factor %f for unit option %f.", data_type, conversion_factor, unit_option)
        return value * conversion_factor
    except KeyError:
        logging.info("Error while converting %s with factor %f for unit option %f.", data_type, conversion_factor, unit_option)
        raise ValueError(f"Invalid unit option {unit_option} or data type {data_type} specified.")

def convert_column(data, unit_option, data_type):
    """
    Convert the units of the specified data column based on the given data type and unit option.

    Parameters:
    - data (array): 2D array containing the data whose second column is to be converted.
    - unit_option (int or str): Option that specifies the desired unit conversion.
    - data_type (str): Type of the data (e.g., 'viscosity', 'conductivity') to determine the type of conversion.

    Returns:
    - array: The input 2D array with the second column converted to the desired units.

    Note:
    - This function assumes that the second column (index 1) of the input data array contains 
      the values to be converted.
    - The conversion is done using the `convert_units` function which must be defined elsewhere.
    """
    data[:, 1] = convert_units(data[:, 1], unit_option, data_type)
    return data

def compute_fit_quality(true_values, predicted_values):
    """
    Compute and return the root mean squared error and R^2 score between 
    true_values and predicted_values.
    
    Parameters:
    - true_values: Actual, observed or true values.
    - predicted_values: Values predicted by the model.

    Returns:
    - rms: Root mean squared error between the true and predicted values.
    - r_squared: R^2 score between the true and predicted values.
    """
    r_squared = r2_score(true_values, predicted_values)
    rms = np.sqrt(mean_squared_error(true_values, predicted_values))
    
    logging.info("Fitting quality ran fine")
    return rms, r_squared

def calculate_effective_molality(species_data: List[Dict[str, Union[float, bool]]]) -> Tuple[float, float]:
    """
    Calculate the effective molality based on the mole abundance and molecular weight of each species.
    The effective molality represents the hypothetical molality if the system was represented by a 
    single ionic and a single non-ionic species.
    
    Parameters:
    - species_data (List[Dict[str, Union[float, bool]]]): A list of dictionaries, 
      each representing a species with its mole abundance, molecular weight, and 
      ionic nature.
    
    Returns:
    - molality: The effective molality value for the ionic species in the system.
    - effective_mw_ionic: The effective molecular weight of the ionic species.
    
    Note:
    This function assumes the configuration file contains sections related to each species detailing 
    their mole abundance, molecular weight, and ionic nature.
    """    
    total_moles = 0
    ionic_mw_weighted_sum = 0
    non_ionic_mw_weighted_sum = 0
    total_ionic_moles = 0
    total_non_ionic_moles = 0
    
    # Loop through each species to calculate the effective molecular weight
    for species in species_data:
        mole_abundance = species['mole_abundance']
        molecular_weight = species['molecular_weight']
        is_ionic = species['is_ionic']
        
        total_moles += mole_abundance
        if is_ionic:
            ionic_mw_weighted_sum += mole_abundance * molecular_weight
            total_ionic_moles += mole_abundance
        else:
            non_ionic_mw_weighted_sum += mole_abundance * molecular_weight
            total_non_ionic_moles += mole_abundance
    
    effective_mw_ionic = ionic_mw_weighted_sum / total_ionic_moles

    if total_non_ionic_moles == 0:
        effective_mw_non_ionic = 0
    else:
        effective_mw_non_ionic = non_ionic_mw_weighted_sum / total_non_ionic_moles
    
    # Calculate mole fractions
    mole_fraction_non_ionic = total_non_ionic_moles / total_moles
    mole_fraction_ionic = total_ionic_moles / total_moles
    
    # Calculate moles of effective non-ionic molecule in 1kg. If only ionic species are in the system, it assumes molality =0
    if effective_mw_non_ionic == 0:
        molality = 0 
    else:
        moles_effective_non_ionic_per_kg = 1000 / effective_mw_non_ionic
        # Determine moles of effective ionic molecule using mole fractions
        moles_effective_ionic = (moles_effective_non_ionic_per_kg * mole_fraction_ionic) / mole_fraction_non_ionic
        # The molality for the effective ionic molecule
        molality = moles_effective_ionic
    
    logging.info("Molality calculation ran fine")
    return molality, effective_mw_ionic

def save_to_output_dir(filename):
    """
    Generate the full path for a file to be saved in the output directory.

    Parameters:
    - filename (str): The name of the file to be saved.

    Returns:
    - str: The full path where the file will be saved.

    Note:
    - This function uses the globally defined 'output_directory'.
    """
    return os.path.join(output_directory, filename)

# Equations definition

def vtf(T, A, B, C, ScaleFactor):
    """
    Vogel-Tamman-Fulcher (VTF) equation. It describes how the transport properties of 
    a fluid depend on temperature. The equation has the form y = A e^(-B/(R(T-C)))
    
    Parameters:
    - T: Temperature in Kelvin.
    - A, B, C: Parameters of the VTF equation.
    - ScaleFactor: Temperature-dependent factor to multipy the pre-exponential factor.
    - A is expressed in mPa s or mS/cm for viscosity or conductivity, respectively
    - B is expressed in J/mol
    - C is expressed in K

    Returns:
    - The specicic transport property of the fluid at temperature T.
    """
    R = 8.3144626 # Gas constant
    if ScaleFactor:
        return (1/np.sqrt(T)) * A * np.exp(-B / (R * (T - C)))
    else:
        return A * np.exp(-B / (R * (T - C)))

def walden_equation(x, alpha, C):
    """
    Compute the Walden equation, which relates log(molar conductivity) to log(fluidity).
    The equation has the form Λη^α=C

    Parameters:
    - x (array-like): Input values (log_fluidity) for the Walden equation.
    - alpha (float): The slope of the Walden plot.
    - C (float): A parameter in the Walden equation. Should be > 0; if <= 0, it is replaced 
      with a small positive number to avoid mathematical errors.

    Returns:
    - array-like: Computed values of log(molar conductivity).

    Notes:
    - If C is non-positive, a warning message is printed, and C is set to 1e-9 to avoid 
      division by zero or log of a non-positive number.
    """
    if C <= 0:
        C = 1e-9
        logging.warning("C is non-positive. Using C={C} instead.")
        print(f"Warning: C is non-positive. Using C={C} instead.")
    return alpha * x + np.log10(C)

# Fitting functions

def fit_data_vtf_unrestricted(data, data_type, guess, ScaleFactor):
    """
    Perform unrestricted VTF fitting on the provided data and save the parameters.

    Parameters:
    - data (array): 2D array containing the data to be fitted.
    - data_type (str): Type of the data (e.g., 'viscosity' or 'conductivity') for logging.
    - guess (tuple): Initial guesses for the VTF parameters (A, B, C).
    - ScaleFactor (boolean): Scale factor for the VTF equation.

    Returns:
    - tuple: Fitted VTF parameters (A, B, C).
    - A is expressed in mPa s or mS/cm for viscosity or conductivity, respectively
    - B is expressed in J/mol
    - C is expressed in K

    Notes:
    - This function also writes the fitted parameters to a file in the output directory.
    """
    T = data[:, 0]
    y = data[:, 1]
    A_init, B_init, C_init = guess
    p0 = [A_init, B_init, C_init]
    bounds = ([0, -np.inf, 10], [np.inf, np.inf, 500])
    
    if data_type == "viscosity":
        ScaleFactor = False

    popt, pcov = curve_fit(lambda T, A, B, C: vtf(T, A, B, C, ScaleFactor), T, y, p0=p0, bounds=bounds, maxfev=10000)
    
    rms, r2 = compute_fit_quality(y, vtf(T, *popt, ScaleFactor))

    with open(save_to_output_dir("fitting_parameters.txt"), "a") as f:
        f.write(f"\n{data_type.capitalize()} fitting (unrestricted):\n")
        param_names = ["A", "B", "C"]
        for i, name in enumerate(param_names):
            f.write(f"{name} = {popt[i]:.5f} ± {pcov[i][i]:.5e}\n")
        f.write(f"\nRMS = {rms:.4g}\nR^2 = {r2:.4g}\n-o-o-o-o-o-o-\n")

    return tuple(popt)

def joint_fit(visc_params_unrestricted, cond_params_unrestricted, C_bounds, viscosity_data_array, conductivity_data_array, ScaleFactor, w):
    """
    Perform a joint fitting procedure on viscosity and conductivity data arrays with the
    VTF equation, constraining the C parameter to be the same for both properties.

    Parameters:
    - visc_params_unrestricted, cond_params_unrestricted: VTF parameters from previous unrestricted fits.
    - C_bounds: Tuple defining the lower and upper bounds for C during fitting.
    - viscosity_data_array, conductivity_data_array: 2D arrays containing viscosity and conductivity data.
    - ScaleFactor: Scale factor for the VTF equation.
    - w: Weighting factor.

    Returns:
    - The fitted VTF parameters for viscosity and conductivity.
    - A is expressed in mPa s or mS/cm for viscosity or conductivity, respectively
    - B is expressed in J/mol
    - C is expressed in K
    """
    T_visc = viscosity_data_array[:, 0]
    y_visc = viscosity_data_array[:, 1]
    
    T_cond = conductivity_data_array[:, 0]
    y_cond = conductivity_data_array[:, 1]
    
    C_avg = (visc_params_unrestricted[2] + cond_params_unrestricted[2]) / 2

    initial_guess = [
        visc_params_unrestricted[0],  # A_visc
        visc_params_unrestricted[1],  # B_visc
        C_avg, 
        cond_params_unrestricted[0],  # A_cond
        cond_params_unrestricted[1]   # B_cond
    ]
    bounds = [(None, None), (None, None), C_bounds, (None, None), (None, None)]

    def objective_function(params, T_visc, y_visc, T_cond, y_cond, ScaleFactor):
        A_visc, B_visc, C, A_cond, B_cond = params
        pred_cond = vtf(T_cond, A_cond, B_cond, C, ScaleFactor)
        ScaleFactor = False
        pred_visc = vtf(T_visc, A_visc, B_visc, C, ScaleFactor)
        residuals_visc = y_visc - pred_visc
        residuals_cond = y_cond - pred_cond
        return w * np.sum(residuals_visc**2) + (1-w) * np.sum(residuals_cond**2)
    
    result = minimize(objective_function, initial_guess, args=(T_visc, y_visc, T_cond, y_cond, ScaleFactor), bounds=bounds)
    
    A_visc_fit, B_visc_fit, C_fit, A_cond_fit, B_cond_fit = result.x
    
    predicted_visc = vtf(T_visc, A_visc_fit, B_visc_fit, C_fit, ScaleFactor)
    predicted_cond = vtf(T_cond, A_cond_fit, B_cond_fit, C_fit, ScaleFactor)
    
    rms_visc, r2_visc = compute_fit_quality(y_visc, predicted_visc)
    rms_cond, r2_cond = compute_fit_quality(y_cond, predicted_cond)
    
    with open(save_to_output_dir("fitting_parameters.txt"), "a") as f:
        f.write("\nViscosity fitting (constrained C: {:.5f}):\n".format(C_fit))
        f.write("A = {:.5f}\nB = {:.5f}\nC = {:.5f}\n".format(A_visc_fit, B_visc_fit, C_fit))
        f.write("\nRMS = {:.4g}\nR^2 = {:.4g}\n".format(rms_visc, r2_visc))
        f.write("-o-o-o-o-o-o-\n")
        
        f.write("\nConductivity fitting (constrained C: {:.5f}):\n".format(C_fit))
        f.write("A = {:.5f}\nB = {:.5f}\nC = {:.5f}\n".format(A_cond_fit, B_cond_fit, C_fit))
        f.write("\nRMS = {:.4g}\nR^2 = {:.4g}\n".format(rms_cond, r2_cond))
        f.write("-o-o-o-o-o-o-\n")

    logging.info("Joint VTF fitting ran fine")
    return (A_visc_fit, B_visc_fit, C_fit), (A_cond_fit, B_cond_fit, C_fit)

def walden_analysis(conductivity_parameters, viscosity_parameters, density_data_array, molality, effective_mw, ScaleFactor, restricted):
    """
    Use density data array to compute molar conductivity, fluidity, and Walden Ionicity. 
    It also fits the Walden equation and saves the parameters, data, and ionicity results.

    Parameters:
    - conductivity_parameters (tuple): Parameters (A, B, C) from the VTF fit for conductivity.
    - viscosity_parameters (tuple): Parameters (A, B, C) from the VTF fit for viscosity.
    - density_data_array (array): 2D array containing density data.
    - molality: Calculated effective molality of Ionic species. It is set to zero if the system is entirely ionic
    - effective_mw: Calculated effective molecular weight of Ionic species
    - ScaleFactor: Scale factor for the VTF equation.
    - restricted (bool): Flag to determine if the VTF fitting is restricted or unrestricted.

    Returns:
    - The fitted parameters for the Walden equation, data used in the Walden plot, Walden ionicity, 
      and Walden fit results.
    - Alpha is a pure number.
    - C is expressed in S cm2 P^Alpha.

    Note:
    This function uses the `calculate_effective_molality` function to determine the effective 
    molality based on the mole abundance and molecular weight of each species. This effective 
    molality represents the hypothetical molality if the system was represented by a single ionic 
    and a single non-ionic species.
    """
    temp_density = density_data_array[:, 0]
    density_data = density_data_array[:, 1]

    interpolated_conductivity = vtf(temp_density, *conductivity_parameters, ScaleFactor)
    interpolated_viscosity_mPas = vtf(temp_density, *viscosity_parameters, ScaleFactor)
    interpolated_viscosity = interpolated_viscosity_mPas * 0.01 # Converts viscosity in Poise from mPa s

    if molality == 0:
        molar_concentration = density_data / effective_mw # in moles/cm^3
    else:
        mass_solute = molality * effective_mw
        mass_solution = 1000 + mass_solute
        volume_solution = mass_solution / density_data
        molar_concentration = molality / volume_solution # in moles/cm^3

    molar_conductivity = (interpolated_conductivity * 1e-3) / molar_concentration

    fluidity = 1 / interpolated_viscosity

    log_molar_conductivity = np.log10(molar_conductivity)
    log_fluidity = np.log10(fluidity)

    deltaW = log_molar_conductivity - log_fluidity
    walden_ionicity = 10 ** deltaW

    param_bounds = ([0.00001, -np.inf], [1.0, np.inf])
    popt_walden, _ = curve_fit(walden_equation, log_fluidity, log_molar_conductivity, p0=[1.0, 1.0], bounds=param_bounds, maxfev=10000)

    log_molar_conductivity_predicted = walden_equation(log_fluidity, *popt_walden)
    rms, r2 = compute_fit_quality(log_molar_conductivity, log_molar_conductivity_predicted)

    with open(save_to_output_dir('fitting_parameters.txt'), 'a') as f:
        if restricted:
            f.write("\nWalden fitting (using constrained VTF fitting parameters):\n")
        else:
            f.write("\nWalden fitting (using unrestricted VTF parameters):\n")
        f.write(f"Alpha = {popt_walden[0]:.4g}\n")
        f.write(f"C = {popt_walden[1]:.4g}\n\n")
        f.write(f"RMS = {rms:.4g}\n")
        f.write(f"R^2 = {r2:.4g}\n-o-o-o-o-o-o-\n")

    with open(save_to_output_dir('WaldenPlot.txt'), 'w') as f:
        f.write("Temperature\tLog10 Fluidity\tLog10 Molar Conductivity\n")
        for t, lf, lm in zip(temp_density, log_fluidity, log_molar_conductivity):
            f.write(f"{t}\t{lf}\t{lm}\n")
    
    with open(save_to_output_dir('WaldenIonicity.txt'), 'w') as f:
        f.write("Temperature\tIonicity\n")
        for t, wi in zip(temp_density, walden_ionicity):
            f.write(f"{t}\t{wi:.4f}\n")

    logging.info("Walden analysis ran fine")
    return popt_walden

# Output

def save_vtf_prediction(file_name, parameters, lower_temperature, upper_temperature, ScaleFactor):
    """
    Generate and save VTF predictions for a temperature range based on the fitting parameters.

    Parameters:
    - file_name (str): Name of the file where the predictions will be saved.
    - parameters (tuple): VTF parameters (A, B, C) to be used for predictions.
    - lower_temperature (float): The lowest temperature for predictions.
    - upper_temperature (float): The highest temperature for predictions.
    - ScaleFactor (boolean): Scale factor for the VTF equation.

    Notes:
    - This function generates predictions at 5K intervals between lower_temperature and
      upper_temperature.
    - The predicted values are saved to a text file in the output directory.
    """
    if not isinstance(file_name, str):
        logging.error("Invalid file_name type. Expected str.")
        return
    
    if not (isinstance(lower_temperature, (int, float)) and isinstance(upper_temperature, (int, float))):
        logging.error("Invalid temperature type. Expected int or float.")
        return
    
    if lower_temperature >= upper_temperature:
        logging.error("Lower temperature should be less than upper temperature.")
        return
    
    if not isinstance(parameters, tuple) or len(parameters) != 3:
        logging.error("Invalid VTF parameters. Expected a tuple of length 3.")
        return
    
    try:
        T_range = np.arange(lower_temperature, upper_temperature + 1, 5)
        predicted_values = vtf(T_range, *parameters, ScaleFactor)
        
        with open(save_to_output_dir(file_name), 'w') as f:
            if 'viscosity' in file_name:
                f.write("Temperature[K]\tViscosity[mPa·s]\n")
            elif 'conductivity' in file_name:
                f.write("Temperature[K]\tConductivity[mS·cm^-1]\n")
            else:
                logging.error("Invalid file_name. Expected 'viscosity' or 'conductivity' in the name.")
                return

            for T, value in zip(T_range, predicted_values):
                f.write(f"{T}\t{value:.4f}\n")
                
        logging.info(f"VTF predictions saved to {file_name}.")
        
    except Exception as e:
        logging.error(f"Error encountered while saving VTF predictions: {e}")

def plot_vtf(data_array, property_name, unit, parameters, ScaleFactor):
    """
    Load data, generate VTF fit, and plot both together in a semilogy plot.

    Parameters:
    - file_path (str): Path to the file containing the data to be plotted.
    - property_name (str): Name of the physical property being plotted (for labeling).
    - unit (str): Units of the property being plotted (for labeling).
    - parameters (tuple): VTF parameters (A, B, C) to be used for generating the fit.
    - ScaleFactor (boolean): Scale factor for the VTF equation.

    Notes:
    - This function generates a plot and saves it as a PNG file in the output directory.
    """
    T = data_array[:, 0] 
    data = data_array[:, 1]
    predicted = vtf(T, *parameters, ScaleFactor)
    plt.figure(figsize=(15/2.54, 12/2.54), dpi=80)
    plt.semilogy(T, data, 'ko', markersize=5, label='Experimental Data')
    plt.semilogy(T, predicted, 'k--', linewidth=2, label='VTF Fit')
    plt.xlabel('Temperature [$K$]', fontsize=16)
    plt.ylabel(f'{property_name} [${unit}$]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to_output_dir(f'{property_name}_plot.png'), dpi=300)
    plt.show(block=False)

def plot_vtf_both(viscosity_data_array, conductivity_data_array, viscosity_parameters, conductivity_parameters, ScaleFactor):
    """
    Generate and display a plot of the VTF fits for both viscosity and conductivity data.

    Parameters:
    - viscosity_data_array (array): 2D array containing viscosity data.
    - conductivity_data_array (array): 2D array containing conductivity data.
    - viscosity_parameters (tuple): VTF parameters (A, B, C) for the viscosity fit.
    - conductivity_parameters (tuple): VTF parameters (A, B, C) for the conductivity fit.
    - ScaleFactor (boolean): Scale factor for the VTF equation.

    Notes:
    - This function generates a dual-axis plot, visualizing viscosity and conductivity 
      on separate y-axes and temperature on the x-axis.
    - Experimental data are plotted with markers, and VTF fits are plotted with dashed lines.
    - The plot is saved as a PNG file in the output directory and is displayed in a window.
    """
    T_visc = viscosity_data_array[:, 0]
    viscosity_data = viscosity_data_array[:, 1]
    
    T_cond = conductivity_data_array[:, 0]
    conductivity_data = conductivity_data_array[:, 1]
    
    predicted_viscosity = vtf(T_visc, *viscosity_parameters, ScaleFactor)
    predicted_conductivity = vtf(T_cond, *conductivity_parameters, ScaleFactor)
    
    fig, ax1 = plt.subplots(figsize=(15/2.54, 12/2.54), dpi=80)
    ax1.set_xlabel('Temperature [$K$]', fontsize=16)
    ax1.set_ylabel('Conductivity [$mS cm^{-1}$]', fontsize=16)
    ax1.semilogy(T_cond, conductivity_data, 'ko', markersize=5, label='Experimental Conductivity')
    ax1.semilogy(T_cond, predicted_conductivity, 'k--', linewidth=2, label='VTF Fit Conductivity')
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Viscosity [$mPa·s$]', fontsize=16)
    ax2.semilogy(T_visc, viscosity_data, 'kd', markersize=5, label='Experimental Viscosity')
    ax2.semilogy(T_visc, predicted_viscosity, 'k-.', linewidth=2, label='VTF Fit Viscosity')
    ax2.tick_params(axis='y', labelsize=12)
    
    fig.tight_layout()
    plt.savefig(save_to_output_dir('Conductivity_Viscosity_plot.png'), dpi=300)
    plt.show(block=False)

def plot_walden(walden_plot_file, *predicted_walden_files_and_labels, restricted=True):
    """
    Plot the Walden plot: a graph of log10(molar conductivity) vs log10(fluidity).

    Parameters:
    - walden_plot_file: file containing the experimental Logarithm (base 10) of fluidity data points.
    - *predicted_walden_files_and_labels: tuple containing pairs of:
        - predicted_walden_file: file containing the calculated Logarithm (base 10) of molar conductivity data points using the Walden fitting.
        - label: string label for the legend.
    """
    # Load data from files
    _, log_fluidity, log_molar_conductivity = np.loadtxt(walden_plot_file, skiprows=1, unpack=True)

    x_lower_limit = np.min(log_fluidity) - 1
    x_upper_limit = np.max(log_fluidity) + 1

    # Plotting
    plt.figure(figsize=(15/2.54, 12/2.54), dpi=80)
    plt.plot(log_fluidity, log_molar_conductivity, 'ko', markersize=5, label='Experimental Data')
    
    # Plotting predicted data
    for predicted_walden_file, label in predicted_walden_files_and_labels:
        predicted_log_fluidity, predicted_log_molar_conductivity = np.loadtxt(predicted_walden_file, unpack=True, skiprows=1)
        plt.plot(predicted_log_fluidity, predicted_log_molar_conductivity, '--', linewidth=2, label=label)
    
    plt.plot([x_lower_limit, x_upper_limit], [x_lower_limit, x_upper_limit], 'k-', linewidth=2, label='Ideal Line')

    plt.xlabel('Log Fluidity / $Poise^{-1}$', fontsize=16)
    plt.ylabel('Log Molar Conductivity / $S cm^2 mol^{-1}$', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.xlim(x_lower_limit, x_upper_limit)
    plt.ylim(x_lower_limit, x_upper_limit)
    plt.tight_layout()
    if restricted:
        plt.savefig(save_to_output_dir('Walden_plot_restricted.png'), dpi=300)
    else:
        plt.savefig(save_to_output_dir('Walden_plot_unrestricted.png'), dpi=300)
    plt.show(block=False)

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Configuration file (e.g., config.ini)")
    args = parser.parse_args()

    if not args.input:
        logging.error("No input configuration file provided. Use the -i or --input option.")
        return

    if not os.path.exists(args.input):
        logging.error(f"Input configuration file {args.input} does not exist.")
        return
    
    config = configparser.ConfigParser()
    try:
        config.read(args.input)
        logging.info(f"Reading configuration file: OK")
    except configparser.Error as e:
        logging.error(f"Error reading configuration file: {e}")
        return
    
    # Parsing configurations
    try:
        files_section = config["Files"]
        species_section = config["Species"]
        parameters_section = config["Parameters"]
        logging.info(f"Sections in configuration file: OK")
    except KeyError as e:
        logging.error(f"Missing section in the configuration file: {e}")
        return
    
    # Fetching files
    viscosity_file = files_section.get("viscosity", "")
    conductivity_file = files_section.get("conductivity", "")
    density_file = files_section.get("density", "")

    # Fetching composition
    total_species = species_section.getint("total_species", "")

    # Fetching parameters
    A_init = parameters_section.getfloat("A", 1.0)
    B_init = parameters_section.getfloat("B", 8000.0)
    C_init = parameters_section.getfloat("C", 200.0)
    lower_temperature = parameters_section.getint("lower_temperature", 273)
    upper_temperature = parameters_section.getint("upper_temperature", 400)
    ScaleFactor = parameters_section.getboolean("scale_factor", False)
    w = parameters_section.getfloat("w", 0.1)
    compute_walden = parameters_section.getboolean("compute_walden", False)

    # Convert units if necessary
    if viscosity_file:
        viscosity_unit_option = parameters_section.getint("viscosity_unit_option")
        viscosity_data = np.loadtxt(viscosity_file)
        viscosity_data = convert_column(viscosity_data, viscosity_unit_option, "viscosity")

    if conductivity_file:
        conductivity_unit_option = parameters_section.getint("conductivity_unit_option")
        conductivity_data = np.loadtxt(conductivity_file)
        conductivity_data = convert_column(conductivity_data, conductivity_unit_option, "conductivity")

    if density_file:
        density_unit_option = parameters_section.getint("density_unit_option")
        density_data = np.loadtxt(density_file)
        density_data = convert_column(density_data, density_unit_option, "density")

    # Check if the temperature limits for predictions are reasonable
    if lower_temperature >= upper_temperature:
        logging.error("The lower temperature limit should be strictly smaller than the upper temperature limit.")
        return

    global output_directory # Directory that will be created and will contain the output files
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_directory = f"output_{timestamp}"
    os.makedirs(output_directory, exist_ok=True)

    open(os.path.join(output_directory, "fitting_parameters.txt"), "w").close() # Create the main output file

    C_values = []
    initial_guess_vtf = (A_init, B_init, C_init)

    if viscosity_file: # Perform fit, save results, and plot viscosity data. Unrestricted
        viscosity_parameters_unrestricted = fit_data_vtf_unrestricted(viscosity_data, "viscosity", initial_guess_vtf, ScaleFactor)
        C_values.append(viscosity_parameters_unrestricted[2])
        save_vtf_prediction('predicted_viscosity_unrestricted.txt', viscosity_parameters_unrestricted, lower_temperature, upper_temperature, ScaleFactor)
        plot_vtf(viscosity_data, 'Viscosity', 'mPa s', viscosity_parameters_unrestricted, ScaleFactor)

    if conductivity_file: # Perform fit, save results, and plot conductivity data. Unrestricted
        conductivity_parameters_unrestricted = fit_data_vtf_unrestricted(conductivity_data, "conductivity", initial_guess_vtf, ScaleFactor)
        C_values.append(conductivity_parameters_unrestricted[2])
        save_vtf_prediction('predicted_conductivity_unrestricted.txt', conductivity_parameters_unrestricted, lower_temperature, upper_temperature, ScaleFactor)
        plot_vtf(conductivity_data, 'Conductivity', 'mS cm^{-1}', conductivity_parameters_unrestricted, ScaleFactor)
    
    if viscosity_file and conductivity_file: # Perform fit, save results, and plot viscosity and conductivity data simultaneously. Restricted
        C_min = min(C_values)
        C_max = max(C_values)
        joint_visc_params, joint_cond_params = joint_fit(viscosity_parameters_unrestricted, conductivity_parameters_unrestricted, (C_min, C_max), viscosity_data, conductivity_data, ScaleFactor, w)
        save_vtf_prediction('predicted_viscosity_restricted.txt', joint_visc_params, lower_temperature, upper_temperature, ScaleFactor)
        save_vtf_prediction('predicted_conductivity_restricted.txt', joint_cond_params, lower_temperature, upper_temperature, ScaleFactor)
        plot_vtf_both(viscosity_data, conductivity_data, joint_visc_params, joint_cond_params, ScaleFactor)

    if viscosity_file and conductivity_file and density_file and compute_walden:
        species_data = []
        fluidity_range = np.arange(-5, 5.1, 0.2)

        try:
            for i in range(1, total_species + 1):
                mole_abundance = species_section.getfloat(f"species{i}_mole_abundance", "")
                molecular_weight = species_section.getfloat(f"species{i}_molecular_weight", "")
                is_ionic = species_section.getboolean(f"species{i}_ionic", "")
                species_data.append({
                    'mole_abundance': mole_abundance,
                    'molecular_weight': molecular_weight,
                    'is_ionic': is_ionic
                })
                logging.info(f"Parsing species data: OK")
        except (configparser.NoOptionError, ValueError) as e:
            logging.error(f"Error processing species data: {e}")
            return

        molality, effective_mw = calculate_effective_molality(species_data)
        
        # Restricted Fit
        restricted_walden = True
        walden_parameters_restricted = walden_analysis(joint_cond_params, joint_visc_params, density_data, molality, effective_mw, ScaleFactor, restricted_walden)
        predicted_walden_restricted = walden_equation(fluidity_range, *walden_parameters_restricted)
        
        # Unrestricted Fit
        restricted_walden = False
        walden_parameters_unrestricted = walden_analysis(conductivity_parameters_unrestricted, viscosity_parameters_unrestricted, density_data, molality, effective_mw, ScaleFactor, restricted_walden)
        predicted_walden_unrestricted = walden_equation(fluidity_range, *walden_parameters_unrestricted)
        
        # Save & Plot Restricted
        with open(os.path.join(output_directory, 'predicted_walden_restricted.txt'), 'w') as f:
            f.write("Log Fluidity[Poise^-1]\tLog Molar Conductivity[S·cm^2·mol^-1]\n")
            for lf, lm in zip(fluidity_range, predicted_walden_restricted):
                f.write(f"{lf:.4f}\t{lm:.4f}\n")
        exp_walden_file_path = os.path.join(output_directory, 'WaldenPlot.txt')
        pred_walden_file_path_restricted = os.path.join(output_directory, 'predicted_walden_restricted.txt')
        plot_walden(exp_walden_file_path, (pred_walden_file_path_restricted, 'Restricted Fit'), restricted=True)
        
        # Save & Plot Unrestricted
        with open(os.path.join(output_directory, 'predicted_walden_unrestricted.txt'), 'w') as f:
            f.write("Log Fluidity[Poise^-1]\tLog Molar Conductivity[S·cm^2·mol^-1]\n")
            for lf, lm in zip(fluidity_range, predicted_walden_unrestricted):
                f.write(f"{lf:.4f}\t{lm:.4f}\n")
        pred_walden_file_path_unrestricted = os.path.join(output_directory, 'predicted_walden_unrestricted.txt')
        plot_walden(exp_walden_file_path, (pred_walden_file_path_unrestricted, 'Unrestricted Fit'), restricted=False)

    logging.info("Program execution completed successfully!")
    plt.show()

if __name__ == "__main__":
    main()
