from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from btx.processing.btx_types import LoadDataInput, LoadDataOutput

class LoadData:
    """Load and preprocess XPP data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Dictionary containing:
                - setup.run: Run number
                - setup.exp: Experiment number
                - load_data.roi: ROI coordinates [x1, x2, y1, y2]
                - load_data.energy_filter: Energy filter parameters [E0, dE]
                - load_data.i0_threshold: I0 threshold value
                - load_data.time_bin: Time bin size in ps
        """
        self.config = config
        
        
    def _apply_energy_threshold(self, data: np.ndarray) -> np.ndarray:
        """Apply energy thresholding to the data.
        
        Args:
            data: Input image data array
            
        Returns:
            Thresholded image data array
        """
        E0, dE = self.config['load_data']['energy_filter']
        
        # Define threshold bands
        thresh_1, thresh_2 = E0 - dE, E0 + dE
        thresh_3, thresh_4 = 2 * E0 - dE, 2 * E0 + dE
        thresh_5, thresh_6 = 3 * E0 - dE, 3 * E0 + dE
        
        # Apply thresholding
        data_cleaned = data.copy()
        data_cleaned[(data_cleaned < thresh_1)
                     | ((data_cleaned > thresh_2) & (data_cleaned < thresh_3))
                     | ((data_cleaned > thresh_4) & (data_cleaned < thresh_5))
                     | (data_cleaned > thresh_6)] = 0
                     
        return data_cleaned

    def _calculate_binned_delays(self, raw_delays: np.ndarray) -> np.ndarray:
        """Calculate binned delays from raw delay values."""
        time_bin = float(self.config['load_data']['time_bin'])
        
        # Handle NaN values
        valid_delays = raw_delays[~np.isnan(raw_delays)]
        if len(valid_delays) == 0:
            raise ValueError("No valid delay values found")
            
        delay_min = np.floor(valid_delays.min())
        delay_max = np.ceil(valid_delays.max())
        
        # Create bins
        half_bin = time_bin / 2
        bins = np.arange(delay_min - half_bin, delay_max + time_bin, time_bin)
        
        # Bin the delays
        binned_indices = np.digitize(raw_delays, bins, right=True)
        binned_delays = bins[binned_indices - 1] + half_bin
        
        # Clip to valid range
        binned_delays = np.clip(binned_delays, delay_min, delay_max)
        
        return binned_delays

    def run(self, input_data: LoadDataInput) -> LoadDataOutput:
        """Run the data loading and preprocessing."""
        if input_data.data is not None:
            # Use provided synthetic data
            data = input_data.data
            I0 = input_data.I0
            laser_delays = input_data.laser_delays
            laser_on_mask = input_data.laser_on_mask
            laser_off_mask = input_data.laser_off_mask
        else:
            # Load from file using get_imgs_thresh
            try:
                from btx.processing.xpploader import get_imgs_thresh
            except ImportError:
                print("Note: get_imgs_thresh not available, only synthetic data mode supported")
                raise
                
            data, I0, laser_delays, laser_on_mask, laser_off_mask = get_imgs_thresh(
                self.config['setup']['run'],
                self.config['setup']['exp'],
                self.config['load_data']['roi'],
                self.config['load_data'].get('energy_filter', [8.8, 5]),
                self.config['load_data'].get('i0_threshold', 200),
                self.config['load_data'].get('ipm_pos_filter', [0.2, 0.5]),
                self.config['load_data'].get('time_bin', 2),
                self.config['load_data'].get('time_tool', [0., 0.005])
            )
            
        # Apply energy thresholding
        data = self._apply_energy_threshold(data)
    
        # Calculate binned delays
        binned_delays = self._calculate_binned_delays(laser_delays)
    
        return LoadDataOutput(
            data=data,
            I0=I0,
            laser_delays=laser_delays,
            laser_on_mask=laser_on_mask,
            laser_off_mask=laser_off_mask,
            binned_delays=binned_delays
        )

    def plot_diagnostics(self, output: LoadDataOutput, save_dir: Path) -> None:
        """Generate diagnostic plots for visual validation."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average frame
        avg_frame = np.mean(output.data, axis=0)
        im1 = ax1.imshow(avg_frame, cmap='viridis')
        ax1.set_title('Average Frame')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Frame-to-frame intensity variation
        ax2.plot(np.mean(output.data, axis=(1,2)), label='Mean Intensity')
        ax2.plot(output.I0, label='I0', alpha=0.5)
        ax2.set_title('Intensity Variation')
        ax2.set_xlabel('Frame')
        ax2.legend()
        
        # 3. Delay distribution
        ax3.hist(output.laser_delays[output.laser_on_mask], bins=50, 
                 alpha=0.5, label='Laser On')
        ax3.hist(output.laser_delays[output.laser_off_mask], bins=50,
                 alpha=0.5, label='Laser Off')
        ax3.set_title('Delay Distribution')
        ax3.set_xlabel('Delay (ps)')
        ax3.legend()
        
        # 4. Binned delays vs raw delays
        ax4.scatter(output.laser_delays, output.binned_delays, alpha=0.1)
        ax4.plot([output.laser_delays.min(), output.laser_delays.max()],
                [output.laser_delays.min(), output.laser_delays.max()],
                'r--', label='y=x')
        ax4.set_title('Binned vs Raw Delays')
        ax4.set_xlabel('Raw Delays (ps)')
        ax4.set_ylabel('Binned Delays (ps)')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'load_data_diagnostics.png')
        plt.close()
