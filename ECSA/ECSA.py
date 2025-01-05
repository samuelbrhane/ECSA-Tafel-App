from tkinter import filedialog, simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from numpy.ma.extras import average

def read_cv_data(file_paths):

    scan_rates = []
    anodic_currents = []
    cathodic_currents = []
    colnames = ['Ewe', 'I', 't', 'cycle']

    for file in file_paths:

        data = pd.read_csv(file, delimiter='\t',header = 0, names = colnames, engine = 'python')
        data = data.map(lambda x: float(x.replace(',', '.', )))

        mask = (data['cycle'].values > 2) # Cycle set to 3rd; write maunal cycle input (from 1 to 3 cycles)
        potentials = data['Ewe'].values[np.nonzero(mask)]
        currents = data['I'].values[np.nonzero(mask)]
        time = data['t'].values[np.nonzero(mask)]
        cycle = data['cycle'].values[np.nonzero(mask)]

        #Scan rate calculation; no need for this part, if scan rate is read from header
        scan_array = np.zeros(potentials.size // 4 - 1)
        for i in range(1, potentials.size // 4):
            scan_array[i - 1] = np.abs((potentials[i] - potentials[i - 1]) / (time[i] - time[i - 1]))
        scan_rate = round(average(scan_array), 2)
        scan_rates.append(scan_rate) # [V/s]

        half_index = int(potentials.size / 2)
        anodic_currents.append(currents[0]) # [mA]
        cathodic_currents.append(currents[half_index]) # [mA]

        plt.plot(potentials, currents, label=f'{scan_rate} V/s')

    plt.xlabel('Potential (V)')
    plt.ylabel('Current (mA)')
    plt.title('Cyclic Voltammetry Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

    return scan_rates, anodic_currents, cathodic_currents

def calculate_ecsa(scan_rates, anodic_currents, cathodic_currents):
    C_specific = 0.040 # [mF/cm²]
    geo_area = 0.196 # [cm²]
    delta_currents = np.abs(np.array(anodic_currents) - np.array(cathodic_currents)) / 2 #[mA]
    slope, intercept, r_value, p_value, std_err = linregress(scan_rates, delta_currents)
    C_dl = slope  # [mF]
    ECSA = C_dl / C_specific # [cm²]
    RF = ECSA / geo_area

    # Print results
    print(f"Double-layer capacitance (C_dl): {C_dl:.4f} mF")
    print(f"Specific capacitance (C_specific): {C_specific} µF/cm²")
    print(f"Electrochemically Active Surface Area (ECSA): {ECSA:.8f} cm²")
    print(f"R squared (r_value): {r_value:.4f}")
    print(f"RF: {RF:.8f} ")
    # Step 4: Plot linear regression
    plt.figure(figsize=(8, 6))
    plt.plot(scan_rates, intercept + slope * np.array(scan_rates), 'r--',
             label=f'Linear Fit: Slope = {slope:.4f} mF')
    plt.plot(scan_rates, delta_currents, 'o', label=f'R²: {r_value:4f}')
    plt.xlabel('Scan Rate (mV/s)')
    plt.ylabel('Δ Current (mA)')
    plt.title('Double-Layer Capacitance Determination')
    plt.legend()
    plt.grid(True)

    plt.show()

    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    with open(file_path, 'w') as file:
        file_content = (f'Double-layer capacitance (C_dl): {C_dl:.4f} mF \n' +
            f'Specific capacitance (C_specific): {C_specific} mF/cm² \n' +
            f'Electrochemically Active Surface Area (ECSA): {ECSA:.8f} cm² \n' + f'RF: {RF:.8f}')
        file.write(file_content)

    return C_dl, ECSA, RF

file_paths = filedialog.askopenfilenames(title="Select Files",
                                         filetypes=(('txt files',"*.txt"), ("All files", "*.*")))
scan_rates, anodic_currents, cathodic_currents = read_cv_data(file_paths)
C_dl, ECSA, RF = calculate_ecsa(scan_rates, anodic_currents, cathodic_currents)