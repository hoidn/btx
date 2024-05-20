from analysis_tasks import calculate_p_values, generate_signal_mask, generate_background_mask
from histogram_analysis import calculate_histograms
from histogram_analysis import get_average_roi_histogram, wasserstein_distance
from pathlib import Path
from pump_probe import load_data
from pvalues import calculate_emd_values
from scipy.stats import norm
from xpploader import get_imgs_thresh
import h5py
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import requests

logger = logging.getLogger(__name__)

def save_array_to_png(array, output_path):
    # Create the "images" subdirectory if it doesn't exist
    images_dir = os.path.join(os.path.dirname(output_path), "images")
    os.makedirs(images_dir, exist_ok=True)

    # Construct the PNG file path in the "images" subdirectory
    png_filename = os.path.basename(output_path).replace(".npy", ".png")
    png_path = os.path.join(images_dir, png_filename)

    plt.figure(figsize=(10, 8))
    plt.imshow(array, cmap='viridis')
    plt.colorbar(label='Value')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

class LoadData:
    def __init__(self, config):
        required_keys = ['setup', 'load_data']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.run_number = config['setup']['run']
        self.experiment_number = config['setup']['exp']
        self.roi = config['load_data']['roi']
        self.energy_filter = config['load_data'].get('energy_filter', [8.8, 5])
        self.i0_threshold = config['load_data'].get('i0_threshold', 200)
        self.ipm_pos_filter = config['load_data'].get('ipm_pos_filter', [0.2, 0.5])
        self.time_bin = config['load_data'].get('time_bin', 2)
        self.time_tool = config['load_data'].get('time_tool', [0., 0.005])
        self.output_dir = config['load_data']['output_dir']

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        self.data, self.I0, self.laser_delays, self.laser_on_mask, self.laser_off_mask = get_imgs_thresh(
            self.run_number, self.experiment_number, self.roi,
            self.energy_filter, self.i0_threshold,
            self.ipm_pos_filter, self.time_bin, self.time_tool
        )

    def save_data(self):
        output_file = os.path.join(self.output_dir, f"{self.experiment_number}_run{self.run_number}_data.npz")
        np.savez(
            output_file,
            data=self.data,
            I0=self.I0,
            laser_delays=self.laser_delays,
            laser_on_mask=self.laser_on_mask,
            laser_off_mask=self.laser_off_mask
        )

    def summarize(self):
        summary = f"Loaded data for experiment {self.experiment_number}, run {self.run_number}:\n"
        summary += f"  Data shape: {self.data.shape}\n"
        summary += f"  I0 shape: {self.I0.shape}\n"
        summary += f"  Laser delays shape: {self.laser_delays.shape}\n"
        summary += f"  Laser on mask shape: {self.laser_on_mask.shape}\n"
        summary += f"  Laser off mask shape: {self.laser_off_mask.shape}\n"
        summary += f"  Data min: {np.min(self.data):.2f}, max: {np.max(self.data):.2f}\n"
        summary += f"  Data mean: {np.mean(self.data):.2f}, std: {np.std(self.data):.2f}\n"

        print(summary)

        with open(os.path.join(self.output_dir, f"{self.experiment_number}_run{self.run_number}_summary.txt"), 'w') as f:
            f.write(summary)

class MakeHistogram:
    def __init__(self, config):
        required_keys = ['setup', 'make_histogram']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.input_file = config['make_histogram']['input_file']
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['make_histogram']['output_dir'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            data = np.load(self.input_file)['data']
        except (IOError, KeyError) as e:
            self.logger.error(f"Error loading data from {self.input_file}: {e}")
            raise

        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {data.ndim}D")

        self.data = data

    def generate_histograms(self):
        histograms = calculate_histograms(self.data)
        self.histograms = histograms

    def summarize(self):
        summary_str = f"Histogram summary for {self.exp} run {self.run}:\n"
        summary_str += f" Shape: {self.histograms.shape}\n"
        summary_str += f" Min: {self.histograms.min():.2f}, Max: {self.histograms.max():.2f}\n"
        summary_str += f" Mean: {self.histograms.mean():.2f}\n"

        self.summary_str = summary_str
        summary_path = os.path.join(self.output_dir, "histogram_summary.txt")
        with open(summary_path, 'w') as summary_file:
            summary_file.write(summary_str)
        self.summary_str = summary_str

    def save_histograms(self):
        histograms_path = os.path.join(self.output_dir, "histograms.npy")
        np.save(histograms_path, self.histograms)

    def save_histogram_image(self):
        img_path = os.path.join(self.output_dir, "histograms.png")
        plt.figure(figsize=(8, 6))
        plt.imshow(self.histograms.sum(axis=0), cmap='viridis')
        plt.colorbar(label='Counts')
        plt.title(f"Histograms for {self.exp} run {self.run}")
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    @property
    def histogram_summary(self):
        return {
            "Experiment": self.exp,
            "Run": self.run,
            "Histogram shape": str(self.histograms.shape),
            "Min value": f"{self.histograms.min():.2f}",
            "Max value": f"{self.histograms.max():.2f}",
            "Mean": f"{self.histograms.mean():.2f}",
        }


class MeasureEMD:
    def __init__(self, config):
        required_keys = ['calculate_emd', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.histograms_path = config['calculate_emd']['histograms_path']
        self.roi_coords = config['calculate_emd']['roi_coords']
        self.num_permutations = config['calculate_emd'].get('num_permutations', 1000)
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['calculate_emd']['output_path'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_histograms(self):
        if not os.path.exists(self.histograms_path):
            raise FileNotFoundError(f"Histogram file {self.histograms_path} does not exist")

        self.histograms = np.load(self.histograms_path)

        if self.histograms.ndim != 3:
            raise ValueError(f"Expected 3D histograms array, got {self.histograms.ndim}D")

        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        if not (0 <= roi_x_start < roi_x_end <= self.histograms.shape[1] and
                0 <= roi_y_start < roi_y_end <= self.histograms.shape[2]):
            raise ValueError(f"Invalid ROI coordinates for histogram of shape {self.histograms.shape}")

    def save_emd_values(self):
        output_file_npy = os.path.join(self.output_dir, "emd_values.npy")
        output_file_png = os.path.join(self.output_dir, "emd_values.png")
        np.save(output_file_npy, self.emd_values)
        save_array_to_png(self.emd_values, output_file_png)

    def save_null_distribution(self):
        output_file = os.path.join(self.output_dir, "emd_null_dist.npy")
        np.save(output_file, self.null_distribution)

    def calculate_emd(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        if not (0 <= roi_x_start < roi_x_end <= self.histograms.shape[1] and
                0 <= roi_y_start < roi_y_end <= self.histograms.shape[2]):
            raise ValueError(f"Invalid ROI coordinates for histogram of shape {self.histograms.shape}")

        self.avg_hist = get_average_roi_histogram(self.histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
        self.emd_values = calculate_emd_values(self.histograms, self.avg_hist)
        self.generate_null_distribution()

    def generate_null_distribution(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        roi_histograms = self.histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end]

        num_bins = roi_histograms.shape[0]
        num_x_indices = roi_x_end - roi_x_start
        num_y_indices = roi_y_end - roi_y_start

        null_emd_values = []

        for _ in range(self.num_permutations):
            random_x_indices = np.random.choice(range(num_x_indices), size=num_bins)
            random_y_indices = np.random.choice(range(num_y_indices), size=num_bins)

            bootstrap_sample_histogram = roi_histograms[np.arange(num_bins), random_x_indices, random_y_indices]
            null_emd_value = wasserstein_distance(bootstrap_sample_histogram, self.avg_hist)
            null_emd_values.append(null_emd_value)

        self.null_distribution = np.array(null_emd_values)

    def summarize(self):
        summary = (
            f"EMD calculation for {self.exp} run {self.run}:\n"
            f" Histogram shape: {self.histograms.shape}\n"
            f" ROI coordinates: {self.roi_coords}\n"
            f" EMD values shape: {self.emd_values.shape}\n"
            f" EMD min: {np.min(self.emd_values):.3f}, max: {np.max(self.emd_values):.3f}\n"
            f" Null distribution shape: {self.null_distribution.shape}\n"
            f" Null distribution min: {np.min(self.null_distribution):.3f}, max: {np.max(self.null_distribution):.3f}\n"
        )

        with open(f"{self.output_dir}_summary.txt", 'w') as f:
            f.write(summary)

        self.summary = summary

    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)

class CalculatePValues:
    def __init__(self, config):
        required_keys = ['calculate_p_values', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.emd_path = config['calculate_p_values']['emd_path']
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['calculate_p_values']['output_path'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

        if 'null_dist_path' not in config['calculate_p_values']:
            raise ValueError("Missing required configuration key: 'null_dist_path'")
        self.null_dist_path = config['calculate_p_values']['null_dist_path']

    def load_data(self):
        self.emd_values = np.load(self.emd_path)
        if self.emd_values.ndim != 2:
            raise ValueError(f"Expected 2D EMD values array, got {self.emd_values.ndim}D")

        self.null_distribution = np.load(self.null_dist_path)

    def calculate_p_values(self):
        self.p_values = calculate_p_values(self.emd_values, self.null_distribution)

    def summarize(self):
        summary = (
            f"P-value calculation for {self.exp} run {self.run}:\n"
            f" EMD values shape: {self.emd_values.shape}\n"
            f" Null distribution length: {len(self.null_distribution)}\n"
            f" P-values shape: {self.p_values.shape}\n"
        )

        with open(f"{self.output_dir}_summary.txt", 'w') as f:
            f.write(summary)

        self.summary = summary

    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)

    def save_p_values(self, p_values_path=None):
        if p_values_path is None:
            p_values_path = os.path.join(self.output_dir, "p_values.npy")

        p_values_png_path = os.path.join(self.output_dir, "p_values.png")
        assert os.path.isdir(self.output_dir), f"Output directory {self.output_dir} does not exist"

        np.save(p_values_path, self.p_values)
        save_array_to_png(self.p_values, p_values_png_path)
        assert os.path.isfile(p_values_path), f"P-values file {p_values_path} was not saved correctly"

class BuildPumpProbeMasks:
    def __init__(self, config):
        required_keys = ['generate_masks', 'setup']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        self.histograms_path = config['generate_masks']['histograms_path']
        self.p_values_path = config['generate_masks']['p_values_path']
        self.roi_coords = config['generate_masks']['roi_coords']
        self.threshold = config['generate_masks']['threshold']
        self.bg_mask_mult = config['generate_masks']['bg_mask_mult']
        self.bg_mask_thickness = config['generate_masks']['bg_mask_thickness']
        self.exp = config['setup']['exp']
        self.run = config['setup']['run']
        self.output_dir = os.path.join(config['generate_masks']['output_dir'], f"{self.exp}_{self.run}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_histograms(self):
        if not os.path.exists(self.histograms_path):
            raise FileNotFoundError(f"Histogram file {self.histograms_path} does not exist")

        self.histograms = np.load(self.histograms_path)

        if self.histograms.ndim != 3:
            raise ValueError(f"Expected 3D histograms array, got {self.histograms.ndim}D")

    def load_p_values(self):
        assert os.path.exists(self.p_values_path), f"P-values file {self.p_values_path} does not exist"
        assert os.path.isfile(self.p_values_path), f"P-values path {self.p_values_path} is not a file"

        self.p_values = np.load(self.p_values_path)
        assert self.p_values.ndim == 2, f"Expected 2D p-values array, got {self.p_values.ndim}D"

    def generate_masks(self):
        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_coords

        self.signal_mask = generate_signal_mask(
            self.p_values, self.threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end, self.histograms
        )

        self.bg_mask = generate_background_mask(self.signal_mask, self.bg_mask_mult, self.bg_mask_thickness)

    def summarize(self):
        summary = (f"Mask generation for {self.exp} run {self.run}:\n"
                   f" P-values shape: {self.p_values.shape}\n"
                   f" Threshold: {self.threshold}\n"
                   f" Signal mask shape: {self.signal_mask.shape}\n"
                   f" Signal mask count: {np.sum(self.signal_mask)}\n"
                   f" Background mask shape: {self.bg_mask.shape}\n"
                   f" Background mask count: {np.sum(self.bg_mask)}\n")
        with open(f"{self.output_dir}_summary.txt", 'w') as f:
            f.write(summary)
        self.summary = summary

    def report(self, report_path):
        with open(report_path, 'a') as f:
            f.write(self.summary)

    def save_masks(self):
        signal_mask_path = os.path.join(self.output_dir, "signal_mask.npy")
        bg_mask_path = os.path.join(self.output_dir, "bg_mask.npy")
        np.save(signal_mask_path, self.signal_mask)
        np.save(bg_mask_path, self.bg_mask)
        save_array_to_png(self.signal_mask, os.path.join(self.output_dir, "signal_mask.png"))
        save_array_to_png(self.bg_mask, os.path.join(self.output_dir, "bg_mask.png"))



def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


class PumpProbeAnalysis:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(config['pump_probe_analysis']['output_dir'], f"{config['setup']['exp']}_{config['setup']['run']}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run(self, data, signal_mask, bg_mask):
        # Step 1: Load data (already done externally)
        self.data = data
        self.signal_mask = signal_mask 
        self.bg_mask = bg_mask
        
        # Step 2: Group images into stacks by delay and laser condition
        self.stacks_on, self.stacks_off = self.group_images_into_stacks()
        
        # Steps 3-4: Generate pump-probe curves (signals, std devs, p-values)
        self.delays, self.signals_on, self.std_devs_on, self.signals_off, self.std_devs_off, self.p_values = self.generate_pump_probe_curves()
        
        # Step 5: Plot results
        self.plot_pump_probe_data()
        
        # Step 6: Save pump-probe curves 
        self.save_pump_probe_curves()
        
        # Generate summary and report
        self.summarize()
        self.report()
        
    def group_images_into_stacks(self):
        binned_delays = self.data['binned_delays']
        imgs = self.data['imgs']
        
        delay_output = np.sort(np.unique(binned_delays))
        stacks_on, stacks_off = {}, {}
        
        for delay_val in delay_output:
            idx_on = np.where((binned_delays == delay_val) & self.data['laser_on_mask'])[0]
            idx_off = np.where((binned_delays == delay_val) & self.data['laser_off_mask'])[0]
            
            stacks_on[delay_val] = self.extract_stacks_by_delay(imgs[idx_on])
            stacks_off[delay_val] = self.extract_stacks_by_delay(imgs[idx_off])
            
        return stacks_on, stacks_off
    
    def extract_stacks_by_delay(self, imgs):
        return imgs
    
    def generate_pump_probe_curves(self):
        delays, signals_on, signals_off, std_devs_on, std_devs_off = [], [], [], [], []
        
        for delay in sorted(self.stacks_on.keys()):
            stack_on = self.stacks_on[delay]
            stack_off = self.stacks_off[delay]
            
            signal_on, bg_on, total_var_on = self.calculate_signal_and_background(stack_on)
            signal_off, bg_off, total_var_off = self.calculate_signal_and_background(stack_off)

            norm_signal_on = (signal_on - bg_on) / np.mean(self.data['I0'][self.data['laser_on_mask']])
            norm_signal_off = (signal_off - bg_off) / np.mean(self.data['I0'][self.data['laser_off_mask']])

            std_dev_on = np.sqrt(total_var_on) / np.mean(self.data['I0'][self.data['laser_on_mask']])
            std_dev_off = np.sqrt(total_var_off) / np.mean(self.data['I0'][self.data['laser_off_mask']])

            delays.append(delay)
            signals_on.append(norm_signal_on)
            signals_off.append(norm_signal_off)
            std_devs_on.append(std_dev_on)
            std_devs_off.append(std_dev_off)
            
        p_values = [self.calculate_p_value(signals_on[i], signals_off[i], std_devs_on[i], std_devs_off[i]) for i in range(len(delays))]
            
        return delays, signals_on, std_devs_on, signals_off, std_devs_off, p_values

    def calculate_signal_and_background(self, stack):
        integrated_counts = stack.sum(axis=(1,2))
        signal = np.sum(integrated_counts[self.signal_mask]) 
        bg = np.sum(integrated_counts[self.bg_mask]) * np.sum(self.signal_mask) / np.sum(self.bg_mask)
        var_signal = signal
        var_bg = bg
        total_var = var_signal + var_bg 
        return signal, bg, total_var

    def calculate_p_value(self, signal_on, signal_off, std_dev_on, std_dev_off):
        delta_signal = abs(signal_on - signal_off)
        combined_std_dev = np.sqrt(std_dev_on**2 + std_dev_off**2)
        z_score = delta_signal / combined_std_dev
        return 2 * (1 - norm.cdf(z_score))

    def plot_pump_probe_data(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.errorbar(self.delays, self.signals_on, yerr=self.std_devs_on, fmt='rs-', label='Laser On')  
        ax1.errorbar(self.delays, self.signals_off, yerr=self.std_devs_off, fmt='ks-', mec='k', mfc='white', alpha=0.2, label='Laser Off')
        ax1.set_xlabel('Time Delay (ps)')
        ax1.set_ylabel('Normalized Signal') 
        ax1.legend()
        ax1.grid(True)

        neg_log_p_values = [-np.log10(p) if p > 0 else 0 for p in self.p_values]
        ax2.scatter(self.delays, neg_log_p_values, color='red', label='-log(p-value)')
        for p_val, label in zip([0.05, 0.01, 0.001], ['5%', '1%', '0.1%']):
            neg_log_p_val = -np.log10(p_val)
            ax2.axhline(y=neg_log_p_val, color='black', linestyle='--')
            ax2.text(ax2.get_xlim()[1], neg_log_p_val, label, va='center', ha='left')
        ax2.set_xlabel('Time Delay (ps)')
        ax2.set_ylabel('-log(P-value)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pump_probe_plot.png'))
        
    def save_pump_probe_curves(self):
        np.savez(os.path.join(self.output_dir, 'pump_probe_curves.npz'),
                 delays=self.delays,
                 signals_on=self.signals_on,
                 std_devs_on=self.std_devs_on,
                 signals_off=self.signals_off, 
                 std_devs_off=self.std_devs_off,
                 p_values=self.p_values)
    
    def summarize(self):
        summary = (f"Pump-probe analysis for {self.config['setup']['exp']} run {self.config['setup']['run']}:\n"
                   f" Number of delays: {len(self.delays)}\n" 
                   f" Minimum p-value: {min(self.p_values):.3e}\n"
                   f" Results saved to: {self.output_dir}\n")
        print(summary)
        with open(os.path.join(self.output_dir, 'pump_probe_summary.txt'), 'w') as f:
            f.write(summary)

    def report(self):
        report = (f"Pump-probe analysis for {self.config['setup']['exp']} run {self.config['setup']['run']}\n\n"
                  f"Data loaded from: {self.config['setup']['root_dir']}\n"
                  f"Signal mask loaded from: {self.config['pump_probe_analysis']['signal_mask_path']}\n" 
                  f"Background mask loaded from: {self.config['pump_probe_analysis']['bg_mask_path']}\n\n"
                  f"Analysis parameters:\n"
                  f" I0 threshold: {self.config['pump_probe_analysis']['i0_threshold']}\n"
                  f" IPM pos filter: {self.config['pump_probe_analysis']['ipm_pos_filter']}\n"
                  f" Time bin: {self.config['pump_probe_analysis']['time_bin']} ps\n"
                  f" Time tool: {self.config['pump_probe_analysis']['time_tool']}\n"
                  f" Energy filter: {self.config['pump_probe_analysis']['energy_filter']} keV\n"
                  f" Minimum counts per bin: {self.config['pump_probe_analysis']['min_count']}\n\n"
                  f"Results:\n"
                  f" Number of delays: {len(self.delays)}\n"
                  f" Delays (ps): {self.delays}\n"
                  f" Laser on signals: {self.signals_on}\n"
                  f" Laser on std devs: {self.std_devs_on}\n"
                  f" Laser off signals: {self.signals_off}\n"
                  f" Laser off std devs: {self.std_devs_off}\n"
                  f" p-values: {self.p_values}\n"
                  f" Minimum p-value: {min(self.p_values):.3e}\n\n"
                  f"Plots saved to: {os.path.join(self.output_dir, 'pump_probe_plot.png')}\n"
                  f"Results saved to: {os.path.join(self.output_dir, 'pump_probe_curves.npz')}\n")
        with open(os.path.join(self.output_dir, 'pump_probe_report.txt'), 'w') as f:
            f.write(report)