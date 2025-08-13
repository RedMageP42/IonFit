README File
===========

IonFit - A Comprehensive Electrolyte Properties Analyzer

Version 0.7 20231022

by: Dr. Alessandro Mariani
Partially based on the work of: Dr. Alessandro Innocenti

History:
v 0.7 
- The code now produces the Walden Ionicity as an output in the file `WaldenIonicity.txt`. This output provides insights into the ionic character of the sample based on the Walden plot.
- The temperature-dependent correction factor has been updated to be more in line with the VTF theory.
- General bug fixes.

v 0.6 
- Introduced the ability for the user to define the chemical composition of the sample directly in the config.ini file.
- Bug fixes and performance improvements.

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

===========

OVERVIEW
--------

IonFit.py is designed to analyze electrolyte properties through VTF (Vogel-Tammann-Fulcher) fittings on viscosity and conductivity, and to generate the corresponding Walden Plot. Version 0.7 further enhances the capabilities of IonFit, introducing the Walden Ionicity output, refining the temperature-dependent correction factor in line with VTF theory, and implementing various bug fixes.

DETAILED DESCRIPTION
--------------------

1. REQUIRED LIBRARIES:
   - numpy
   - scipy.optimize
   - sklearn.metrics
   - argparse
   - logging
   - os
   - matplotlib.pyplot
   - datetime
   - configparser 


2. NEW FEATURES IN v0.7:

   a. Walden Ionicity Output: The software now generates an additional output file named `WaldenIonicity.txt`. This file provides insights into the ionic character of the sample based on the Walden plot.

   b. Temperature-dependent Correction Factor: The correction factor has been updated to be more in line with the VTF theory, ensuring more accurate results in the analysis.

   c. General Enhancements: Various bug fixes and improvements have been made to increase the software's reliability and performance.

3. CONFIGURATION FILE (config.ini):

   The config.ini file is structured to guide the user in inputting necessary parameters and file paths in a structured manner, ensuring an organized and error-free input process.

   Sections include:
   - [Files]: Paths to input data files (viscosity, conductivity, density).
   - [Species]: Details about every species in the sample, such as name, mole abundance, molecular weight, and ionic nature.
   - [Parameters]: Various parameters including molecular weight, molality, system type (purely_ionic), initial guesses for VTF parameters, temperature range for predictions, and unit specifications.

   Dummy configuration file:


#######################################################################
#                                                                     #
#          This is the input configuration file for IonFit            #
#                                                                     #
#######################################################################

[Files]

#######################################################################
# Specify the PATH(s) to the file(s) containing the experimental data #
# EXAMPLE: viscosity = path/to/viscosity/file.txt                     #
#######################################################################
viscosity = 
conductivity = 
density = 

[Species]

#######################################################################
# Specify every species in the sample.                                #
# You may duplicate the info block for every species in your system   #
# Please be careful and change the progressive number on the species  #
# parameters, e.g. species1_name, species2_name...                    #
# EXAMPLE:                                                            #
#                                                                     #
# total_species = 2                                                   #
#                                                                     #
# species1_name = Water                                               #
# species1_mole_abundance = 10                                        #
# species1_molecular_weight = 18                                      #
# species1_ionic = False                                              #
#                                                                     #
# species2_name = NaCl                                                #
# species2_mole_abundance = 1                                         #
# species2_molecular_weight = 58                                      #
# species2_ionic = True                                               #
#######################################################################
total_species = 

species1_name = 
species1_mole_abundance = 
species1_molecular_weight = 
species1_ionic = 

[Parameters]

#######################################################################
# Do you want to perform the Walden analysis                          #
# (it requires viscosity, conductivity AND density data)?             #
# True = YES                                                          #
# False = NO                                                          #
# EXAMPLE: compute_walden = False                                     #
# in this example the Walden analysis would NOT be performed          #
#######################################################################
compute_walden = 

#######################################################################
# Initial guess for VTF parameters A, B, C                            #
# A is the pre-exponential factor                                     #
# B is the pseudo activation energy                                   #
# C is the temperature of zero configurational entropy                #
# EQUATION: Y = A * e^(-B/(R(T-C)))                                   #
# EXAMPLE (and suggestes default values):                             #
# A = 1                                                               #
# B = 4000                                                            #
# C = 200                                                             #
#######################################################################
A = 
B = 
C = 

#######################################################################
# Relative weight of viscosity over conductivity in the Joint Fitting #
# (from 0 to 1, with 0.5 being equal weight)                          #
# Only needed if you provide BOTH viscosity and conductivity data     #
# Needed because normally viscosity values are much higher in         #
# magnitude respect to conductivity values                            #
# It could take a few iterations changing this value to get a good    #
# joint fitting                                                       #
# Typical values range from 0.00005 to 0.1                            #
# EXAMPLE (and suggested starting trial): w = 0.01                    #
#######################################################################
w = 

#######################################################################
# Define the temperature (in K) range in which to calculate the       #
# predicted values from the VTF parameters obtained by the fitting    #
# lower_temperature must be higher than the temperature of zero       #
# configurational entropy                                             #
# higher_temperature must be higher than lower_temperature            #
# EXAMPLE (and suggested values):                                     #
# lower_temperature = 260                                             #
# upper_temperature = 400                                             #
#######################################################################
lower_temperature = 
upper_temperature = 

#######################################################################
# Choose to use a temperature-dependent scale fiactor that will       #
# be used to modity the VTF equation (ONLY FOR CONDUCTIVITY).         #
# The scale factor is equal to 1/sqrt(T) and the CONDUCTIVITY         #
# VTF equation becomes: σ = 1/sqrt(T) * A * e^(-B/(R(T-C)))           #
# True = YES                                                          #
# False = NO                                                          #
# EXAMPLE: scale_factor = False                                       #
# in this example the Scale Factor would NOT be used                  #
#######################################################################
scale_factor = 

#######################################################################
# Specify the units in which your experimental data for viscosity     #
# are expressed (1=mPa s; 2=Pa s; 3=cP; 4=P)                          #
# EXAMPLE: viscosity_unit_option = 1                                  #
#######################################################################
viscosity_unit_option = 

#######################################################################
# Specify the units in which your experimental data for conductivity  #
# are expressed (1=mS/cm; 2=S/cm)                                     #
# EXAMPLE: conductivity_unit_option = 1                               #
#######################################################################
conductivity_unit_option = 

#######################################################################
# Specify the units in which your experimental data for density       #
# are expressed (1=g/cm3; 2=kg/m3)                                    #
# EXAMPLE: density_unit_option = 1                                    #
#######################################################################
density_unit_option = 



#######################################################################
#                                                                     #
#                                 END                                 #
#                                                                     #
#######################################################################


4. USAGE:

Usage of IonFit.py has been streamlined with the introduction of the config.ini file. Users can now simply call the script and provide the path to the config file as an argument.

Example:

	$ python IonFit.py -i path_to_config.ini


5. OUTPUT:

Output files include the VTF fitting parameters, predictions of temperature-dependent viscosity and conductivity, Walden plot data, and various PNG images visualizing the VTF fits and Walden plot. All outputs are saved in a timestamped directory for organized recordkeeping.

6. NOTES:

Ensure that the data files provided are formatted correctly with two columns: temperature and the respective property. With the introduction of the ability to define chemical composition, ensure that species details are accurate to obtain meaningful results. Visual inspection of the fitting plots is crucial to verify the quality of the fits.

7. DISCLAIMER:

This script assumes the accuracy and validity of provided data and parameters. Always ensure to visually inspect the fitting plots to verify the fits' quality.

8. LICENSE:

Copyright 2023 Alessandro Mariani

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

9. ACKNOWLEDGEMENTS:

Gratitude to Dr. Alessandro Innocenti, whose work has been pivotal in the development of IonFit.py.
A special thanks goes to Dr. Giovanni Battista Appetecchi, Dr. Matteo Bonomo, Dr. Guinevere Giffin, Dr. Carsten Korte, Dr. Xu Liu, and Prof. Stefano Passerini for the fundamental scientific feedback.

---
Contact: alessandro1 DOT mariani AT polimi DOT it
