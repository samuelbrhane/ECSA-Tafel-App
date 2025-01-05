import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
import math

class FileProcessor:
    def __init__(self, filename, mA_normalization, voltage_adjustment, cycle_number):
        self.filename = filename
        self.mA_normalization = mA_normalization
        self.voltage_adjustment = voltage_adjustment
        self.cycle_number = cycle_number
        self.voltage = []
        self.current = []

    def process(self):
        file_extension = os.path.splitext(self.filename)[1].lower()
        if file_extension in [".txt", ".mpt"]:
            self._process_txt_file()
        elif file_extension == ".csv":
            self._process_csv_file()
        elif file_extension in [".xls", ".xlsx"]:
            self._process_excel_file()
        else:
            raise Exception("Unsupported file type.")

    def _process_txt_file(self):
        columns_found = False
        multi_word_headers = ["Ewe/V vs. SCE", "cycle number", "control changes", "counter inc.", "step time/s", "I Range"]
        
        with open(self.filename, 'r') as file:
            for line in file:
                line = line.strip().replace(',', '.')

                if "Ewe/V" in line and "<I>/mA" in line:
                    headers = line.split()
                    combined_headers = self._combine_headers(headers, multi_word_headers)

                    if "Ewe/V vs. SCE" in combined_headers:
                        combined_headers[combined_headers.index("Ewe/V vs. SCE")] = "Ewe/V"
                    
                    try:
                        ewe_index = combined_headers.index("Ewe/V")
                        current_index = combined_headers.index("<I>/mA")
                        cycle_index = combined_headers.index("cycle number") if "cycle number" in combined_headers else None
                        
                        columns_found = True
                    except ValueError:
                        raise Exception("Required columns not found in the header line.")
                    continue

                if columns_found:
                    try:
                        values = line.split()
                        
                        if cycle_index is not None:
                            cycle_value = int(float(values[cycle_index]))
                            if cycle_value != self.cycle_number:
                                continue 
                        
                        col1 = float(values[ewe_index]) + self.voltage_adjustment
                        col2 = float(values[current_index]) / self.mA_normalization
                        
                        if col2 > 0:
                            self.voltage.append(col1)
                            self.current.append(col2)
                        
                    except (ValueError, IndexError):
                        continue

    def _combine_headers(self, headers, multi_word_headers):
        combined_headers = []
        i = 0
        while i < len(headers):
            if i + 2 < len(headers) and f"{headers[i]} {headers[i + 1]} {headers[i + 2]}" in multi_word_headers:
                combined_headers.append(f"{headers[i]} {headers[i + 1]} {headers[i + 2]}")
                i += 3
            elif i + 1 < len(headers) and f"{headers[i]} {headers[i + 1]}" in multi_word_headers:
                combined_headers.append(f"{headers[i]} {headers[i + 1]}")
                i += 2
            else:
                combined_headers.append(headers[i])
                i += 1
        return combined_headers

    def _process_csv_file(self):
        data = pd.read_csv(self.filename)
        self._extract_data_from_dataframe(data)

    def _process_excel_file(self):
        data = pd.read_excel(self.filename)
        self._extract_data_from_dataframe(data)

    def _extract_data_from_dataframe(self, data):
        if "cycle number" in data.columns:
            data = data[data["cycle number"] == self.cycle_number]

        required_columns = ["Ewe/V", "<I>/mA"]
        if not all(column in data.columns for column in required_columns):
            raise Exception(f"Required columns {required_columns} not found in the dataset.")
      
        data = data[required_columns].dropna()
        data = data[data["<I>/mA"] > 0]
        self.voltage = (data["Ewe/V"] + self.voltage_adjustment).tolist()
        self.current = (data["<I>/mA"] / self.mA_normalization).tolist()
        

class TafelTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tafel Test")
        self.root.geometry("600x400")
        self.root.configure(bg="#f7f7f7")
        self.files = []
        self.mA_normalization = 1
        self.voltage_adjustment = 0
        self.cycle_number = 0
        self.data = []  
        self._setup_ui()

    def _setup_ui(self):
        title_label = tk.Label(
            self.root,
            text="Tafel Test",
            font=("Helvetica", 24, "bold"),
            bg="#f7f7f7",
            fg="#333333"
        )
        title_label.pack(pady=20)

        button_frame = tk.Frame(self.root, bg="#f7f7f7")
        button_frame.pack(pady=30)

        upload_button = ttk.Button(button_frame, text="Upload Dataset", command=self.upload_files, width=20)
        upload_button.grid(row=0, column=0, padx=10, pady=10)

        process_button = ttk.Button(button_frame, text="Process Plots", command=self.process_plots, width=20)
        process_button.grid(row=0, column=1, padx=10, pady=10)

        self.status_label = tk.Label(
            self.root,
            text="Upload datasets to start.",
            font=("Helvetica", 12),
            bg="#f7f7f7",
            fg="#555555"
        )
        self.status_label.pack(pady=20)

    def upload_files(self):
        self.files = filedialog.askopenfilenames(
            title="Select Dataset Files",
            filetypes=[
                ("All Files", "*.*"),
                ("Text Files", "*.txt"),
                ("MPT Files", "*.mpt"),
                ("Excel Files", "*.xls;*.xlsx"),
                ("CSV Files", "*.csv")
            ]
        )
        if self.files:
            self.status_label.config(text=f"{len(self.files)} files selected.")
            self._prompt_user_inputs()

    def _prompt_user_inputs(self):
        try:
            self.cycle_number = float(simpledialog.askstring(
                "Input", "Enter the cycle number:", parent=self.root))
            if self.cycle_number <= 0:
                raise ValueError("Cycle number must be greater than zero.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid cycle number: {e}")
            self.cycle_number = 1
            
        try:
            self.mA_normalization = float(simpledialog.askstring(
                "Input", "Enter the catalyst surface area:", parent=self.root))
            if self.mA_normalization <= 0:
                raise ValueError("Catalyst surface area must be greater than zero.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid catalyst surface area value: {e}")
            self.mA_normalization = 1

        try:
            self.voltage_adjustment = float(simpledialog.askstring(
                "Input", "Enter the reference electrode potential:", parent=self.root))
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid reference electrode potential: {e}")
            self.voltage_adjustment = 0

    def process_plots(self):
        if not self.files:
            messagebox.showerror("Error", "Please upload dataset files first.")
            return

        self.data = []
        for file in self.files:
            processor = FileProcessor(file, self.mA_normalization, self.voltage_adjustment, self.cycle_number)
            
            try:
                processor.process()
                self.data.append((os.path.basename(file), processor.voltage, processor.current))
            except Exception as e:
                messagebox.showerror("Error", f"Error processing {os.path.basename(file)}: {e}")
        
        self.generate_plots()

    def generate_plots(self):
        if not self.data:
            messagebox.showerror("Error", "No data to plot.")
            return
        
        for _, voltage, current in self.data:
            if len(voltage) == 0 or len(current) == 0:
                messagebox.showerror("Error", "No data to plot.")
                return
            
        def customize_ticks(ax):
            ax.tick_params(axis="both", which="major", length=7, width=1.5, direction="in", top=True, right=True)
            ax.tick_params(axis="both", which="minor", length=3, width=1.2, direction="in", top=True, right=True)
            ax.minorticks_on()

        # Full Cycle Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for file_path, voltage, current in self.data:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            ax.plot(voltage, current, label=file_name)
        ax.set_xlabel('E vs. RHE / V', fontsize=12)
        ax.set_ylabel(r'$j \, / \, \mathrm{mA \, cm^{-2}_{geo}}$', fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=False)
        customize_ticks(ax)
        plt.savefig("HER_Voltage_Current.jpg")
        plt.show()

        # Half-cycle Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        half_cycle_results = {}
        for file_path, voltage, current in self.data:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            max_index = np.argmax(current)
            half_voltage = voltage[:max_index + 1]
            half_current = current[:max_index + 1]
            ax.plot(half_voltage, half_current, label=file_name)
        ax.set_xlabel('E vs. RHE / V', fontsize=12)
        ax.set_ylabel(r'$j \, / \, \mathrm{mA \, cm^{-2}_{geo}}$', fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        customize_ticks(ax)
        plt.savefig("HER_Half_Cycle.jpg")
        plt.show()

        for file_path, voltage, current in self.data:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            max_index = np.argmax(current)
            half_voltage = voltage[:max_index + 1]
            half_current = current[:max_index + 1]

            try:
                specific_current = simpledialog.askstring(
                    "Input", f"Enter the specific current value (mA) for {file_name}:", parent=self.root
                )
                if not specific_current:
                    continue
                specific_current = float(specific_current)
            except ValueError:
                messagebox.showinfo("Notice", f"Invalid input for {file_name}. Skipping this dataset.")
                continue

            if min(half_current) <= specific_current <= max(half_current):
                interp_voltage = np.interp(specific_current, half_current, half_voltage)
                adjusted_voltage = interp_voltage - 1.23
                half_cycle_results[file_name] = {
                    "specific_current": specific_current,
                    "interp_voltage": interp_voltage,
                    "adjusted_voltage": adjusted_voltage,
                }
                messagebox.showinfo("Result", f"For {file_name}:\nSpecific Current Value: {specific_current:.2f} mA\nE vs. RHE / V: {interp_voltage:.3f}\nE vs. RHE / V - 1.23: {adjusted_voltage:.3f}")
            else:
                messagebox.showinfo("Notice", f"Specific current {specific_current} mA is out of range for {file_name}.")

        # Tafel Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for file_path, voltage, current in self.data:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            max_index = np.argmax(current)
            half_voltage = voltage[:max_index + 1]
            half_current = current[:max_index + 1]
            log_current = np.log10(half_current)
            ax.plot(log_current, half_voltage, label=file_name)
        ax.set_ylabel('E vs. RHE / V', fontsize=12)
        ax.set_xlabel(r'$\log\,( j ) \, / \, \mathrm{mA \, cm^{-2}_{geo}}$', fontsize=12)
        ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), frameon=False)
        customize_ticks(ax)
        plt.savefig("HER_Tafel_Plot.jpg")
        plt.show()

        # Tafel Plot with Manual Range and Linear Fit
        fig, ax = plt.subplots(figsize=(8, 6))
        x_limits = None
        y_limits = None
        manual_ranges = {}
        for file_path, _, _ in self.data:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                manual_start = float(simpledialog.askstring("Input", f"Enter start potential for {file_name}:", parent=self.root))
                manual_end = float(simpledialog.askstring("Input", f"Enter end potential for {file_name}:", parent=self.root))
                manual_ranges[file_name] = (manual_start, manual_end)
            except ValueError:
                messagebox.showinfo("Notice", f"Invalid manual range for {file_name}. Skipping this dataset.")
                manual_ranges[file_name] = None

        for file_path, voltage, current in self.data:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            max_index = np.argmax(current)
            half_voltage = voltage[:max_index + 1]
            half_current = current[:max_index + 1]
            log_current = np.log10(half_current)

            if manual_ranges[file_name]:
                manual_start, manual_end = manual_ranges[file_name]
                manual_voltage = []
                manual_log_current = []
                for v, log_i in zip(half_voltage, log_current):
                    if manual_start <= v <= manual_end:
                        manual_voltage.append(v)
                        manual_log_current.append(log_i)

                slope, intercept, _, _, _ = linregress(manual_log_current, manual_voltage)
                line_fit = [slope * x + intercept for x in manual_log_current]
                ax.plot(log_current, half_voltage, linestyle='-', linewidth=2, label=f'{file_name}')
                ax.plot(manual_log_current, line_fit, linestyle='-', linewidth=2)
                ax.text(
                    manual_log_current[len(manual_log_current) // 2],
                    line_fit[len(line_fit) // 2] - 0.01,
                    f'≈ {slope * 1000:.2f} mV/dec',
                    fontsize=10,
                    color='black'
                )

                if not x_limits:
                    x_limits = ax.get_xlim()
                if not y_limits:
                    y_limits = ax.get_ylim()

        ax.set_ylabel('E vs. RHE / V', fontsize=12)
        ax.set_xlabel(r'$\log\,( j ) \, / \, \mathrm{mA \, cm^{-2}_{geo}}$', fontsize=12)
        ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), frameon=False)
        customize_ticks(ax)
        plt.savefig("HER_Tafel_LinearFit.jpg")
        plt.show()

        # Linear Fit Only Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        all_y_min = []
        all_y_max = []
        for file_path, voltage, current in self.data:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            max_index = np.argmax(current)
            half_voltage = voltage[:max_index + 1]
            half_current = current[:max_index + 1]
            log_current = np.log10(half_current)

            if manual_ranges[file_name]:
                manual_start, manual_end = manual_ranges[file_name]
                manual_voltage = []
                manual_log_current = []
                for v, log_i in zip(half_voltage, log_current):
                    if manual_start <= v <= manual_end:
                        manual_voltage.append(v)
                        manual_log_current.append(log_i)

                slope, intercept, _, _, _ = linregress(manual_log_current, manual_voltage)
                line_fit = [slope * x + intercept for x in manual_log_current]
                numeric_x = [10 ** x for x in manual_log_current]
                ax.plot(numeric_x, line_fit, linestyle='-', linewidth=2)
                ax.text(
                    numeric_x[len(numeric_x) // 2],
                    line_fit[len(line_fit) // 2] - 0.01,
                    f'≈ {slope * 1000:.2f} mV/dec',
                    fontsize=10,
                    color='black'
                )
                all_y_min.append(min(line_fit))
                all_y_max.append(max(line_fit))

        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
        ax.get_xaxis().set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True)
        ax.tick_params(which='major', length=7, width=1.5, direction='in', top=True, right=True)

        y_min = min(all_y_min)
        y_max = max(all_y_max)
        rounded_y_min = math.floor(y_min * 20) / 20 - 0.05
        rounded_y_max = math.ceil(y_max * 20) / 20 + 0.05

        y_ticks = np.arange(rounded_y_min, rounded_y_max + 0.05, 0.05)

        ax.set_ylim(rounded_y_min, rounded_y_max)
        ax.set_yticks(y_ticks)

        x_min = min(numeric_x)
        x_max = max(numeric_x)
        adjusted_x_min = x_min / 10
        adjusted_x_max = x_max * 10

        ax.set_xlim(adjusted_x_min, adjusted_x_max)
        ax.set_ylabel('E vs. RHE / V', fontsize=12)
        ax.set_xlabel(r'$j \, / \, \mathrm{mA \, cm^{-2}_{geo}}$', fontsize=12)
        customize_ticks(ax)

        plt.savefig("HER_Linear_Fit_Only.jpg")
        plt.show()

        messagebox.showinfo("Success", "All plots generated and saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = TafelTestApp(root)
    root.mainloop()
