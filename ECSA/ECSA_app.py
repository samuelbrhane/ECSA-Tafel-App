import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from numpy.ma.extras import average

class CVDataProcessor:
    def __init__(self, cycle_number):
        self.cycle_number = cycle_number
        self.scan_rates = []
        self.anodic_currents = []
        self.cathodic_currents = []
        self.voltage = []
        self.current = []
        self.de_dt_values = []

    def read_data(self, file_paths):
        for file in file_paths:
            try:
                dE_dt = None
                columns_found = False
                potentials = []
                currents = []
                time = []

                with open(file, 'r') as f:
                    for line in f:
                        line = line.strip().replace(',', '.')

                        if line.startswith("dE/dt") and "unit" not in line:
                            de_dt_split = line.split()
                            dE_dt = float(line.split()[1]) / 1000
                            self.de_dt_values.append(dE_dt)
                            
                        if "Ewe/V" in line and "<I>/mA" in line:
                            columns_found = True
                            continue

                        if columns_found:
                            try:
                                values = line.split()
                                cycle = int(float(values[3]))
                                if cycle == self.cycle_number:
                                    potentials.append(float(values[0]))
                                    currents.append(float(values[1]))
                                    time.append(float(values[2]))
                            except (ValueError, IndexError):
                                continue

                if dE_dt is not None:
                    self.scan_rates.append(dE_dt)

                    half_index = len(currents) // 2
                    self.anodic_currents.append(currents[0])
                    self.cathodic_currents.append(currents[half_index])

                    self.voltage.append(potentials)
                    self.current.append(currents)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process file {file}: {e}")

class CVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECSA Analysis")
        self.root.geometry("600x400")
        self.processor = None
        self._setup_ui()

    def _setup_ui(self):
        title_label = tk.Label(self.root, text="ECSA Analysis", font=("Helvetica", 20, "bold"))
        title_label.pack(pady=20)

        upload_button = tk.Button(self.root, text="Upload Files", command=self.upload_files, width=20)
        upload_button.pack(pady=10)

        analyze_button = tk.Button(self.root, text="Analyze Data", command=self.analyze_data, width=20)
        analyze_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Status: Waiting for files", font=("Helvetica", 12))
        self.status_label.pack(pady=20)

    def upload_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select CV Data Files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_paths:
            messagebox.showerror("Error", "No files selected.")
            return

        try:
            cycle_number = int(simpledialog.askstring("Cycle Number", "Enter the cycle number to analyze:"))
            self.processor = CVDataProcessor(cycle_number)
            self.processor.read_data(file_paths)
            self.status_label.config(text="Status: Files processed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process files: {e}")

    def analyze_data(self):
        if not self.processor or not self.processor.scan_rates:
            messagebox.showerror("Error", "No data available for analysis.")
            return

        # Cyclic Voltammetry Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (voltage, current) in enumerate(zip(self.processor.voltage, self.processor.current)):
            ax.plot(voltage, current, label=f'dE/dt = {self.processor.de_dt_values[i]}')
        ax.set_xlabel('Potential (V)')
        ax.set_ylabel('Current (mA)')
        ax.set_title('Cyclic Voltammetry Curves')
        ax.legend()
        ax.grid(True)
        plt.show()

        # Linear Regression
        delta_currents = abs(np.array(self.processor.anodic_currents) - np.array(self.processor.cathodic_currents)) / 2
        slope, intercept, r_value, _, _ = linregress(self.processor.scan_rates, delta_currents)
        plt.figure(figsize=(8, 6))
        plt.plot(self.processor.scan_rates, intercept + slope * np.array(self.processor.scan_rates), 'r--',
                 label=f'Linear Fit: Slope = {slope:.4f} mF')
        plt.plot(self.processor.scan_rates, delta_currents, 'o', label=f'R²: {r_value:.4f}')
        plt.xlabel('Scan Rate (V/s)')
        plt.ylabel('Δ Current (mA)')
        plt.title('Double-Layer Capacitance Determination')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = CVApp(root)
    root.mainloop()
