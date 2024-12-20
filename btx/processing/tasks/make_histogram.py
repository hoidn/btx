from pathlib import Path
from typing import Dict, Any
try:
    from line_profiler import profile
    PROFILING = True
except ImportError:
    # Create no-op decorator if line_profiler isn't installed
    def profile(func):
        return func
    PROFILING = False
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import functools
import hashlib
import random
from numba.core.errors import NumbaWarning
import warnings

# Suppress Numba warnings
warnings.filterwarnings('ignore', category=NumbaWarning)

class MakeHistogramInput:
    def __init__(self, load_data_output):
        self.load_data_output = load_data_output

from btx.processing.btx_types import MakeHistogramInput, MakeHistogramOutput

def memoize_subsampled(func):
    """Memoize a function by creating a hashable key using deterministically subsampled data."""
    cache = {}

    @functools.wraps(func)
    def wrapper(self, data, *args, **kwargs):  # Add 'self' as first parameter
        # Generate a hashable key from a deterministic subsample
        shape_str = str(data.shape)  # Now data is the actual array
        seed_value = int(hashlib.sha256(shape_str.encode()).hexdigest(), 16) % 10**8
        random.seed(seed_value)

        subsample_size = min(100, data.shape[0])  # Limit the subsample size to a maximum of 100
        subsample_indices = random.sample(range(data.shape[0]), subsample_size)
        subsample = data[subsample_indices]

        hashable_key = hashlib.sha256(subsample.tobytes()).hexdigest()

        # Check cache
        if hashable_key in cache:
            return cache[hashable_key]

        # Calculate the result and store it in the cache
        result = func(self, data, *args, **kwargs)  # Pass self to the original function
        cache[hashable_key] = result

        return result

    return wrapper

@jit(nopython=True, parallel=True, fastmath=True)
def _calculate_histograms_numba(data, bin_boundaries, hist_start_bin, bins, rows, cols):
    """Basic parallel histogram calculation."""
    hist_shape = (bins, rows, cols)
    histograms = np.zeros(hist_shape, dtype=np.float64)
    
    for row in prange(rows):
        for col in range(cols):
            pixel_data = data[:, row, col]
            for frame in range(len(pixel_data)):
                value = pixel_data[frame]
                bin_idx = np.searchsorted(bin_boundaries, value)
                if bin_idx < bins:
                    histograms[bin_idx, row, col] += 1
                else:
                    histograms[hist_start_bin, row, col] += 1
    
    histograms += 1e-9
    return histograms[hist_start_bin:, :, :]


class MakeHistogram:
    """Generate histograms from XPP data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize histogram generation task.
        
        Args:
            config: Dictionary containing:
                - make_histogram.bin_boundaries: Array of bin boundaries
                - make_histogram.hist_start_bin: Index of first bin to include
        """
        self.config = config
        
        # Set defaults if not provided
        if 'make_histogram' not in self.config:
            self.config['make_histogram'] = {}
        hist_config = self.config['make_histogram']
        
        if 'bin_boundaries' not in hist_config:
            hist_config['bin_boundaries'] = np.arange(5, 30, 0.2)
        if 'hist_start_bin' not in hist_config:
            hist_config['hist_start_bin'] = 1
        

    @memoize_subsampled
    def _calculate_histograms(
        self,
        data: np.ndarray,
        bin_boundaries: np.ndarray,
        hist_start_bin: int
    ) -> np.ndarray:
        """Calculate histograms for each pixel using Numba optimization.
        
        Args:
            data: 3D array (frames, rows, cols)
            bin_boundaries: Array of histogram bin boundaries
            hist_start_bin: Index of first bin to include
            
        Returns:
            3D array of histograms (bins, rows, cols)
        """
        bins = len(bin_boundaries) - 1
        rows, cols = data.shape[1], data.shape[2]
        
        return _calculate_histograms_numba(
            data, 
            bin_boundaries, 
            hist_start_bin,
            bins,
            rows,
            cols
        )

    @profile
    def run(self, input_data: MakeHistogramInput) -> MakeHistogramOutput:
        """Run histogram generation.
        
        Args:
            input_data: Input data container
            
        Returns:
            MakeHistogramOutput containing histograms and bin information
        """
        hist_config = self.config['make_histogram']
        bin_boundaries = np.array(hist_config['bin_boundaries'])
        hist_start_bin = hist_config['hist_start_bin']
        
        data = input_data.load_data_output.data
        
        histograms = self._calculate_histograms(
            data,
            bin_boundaries,
            hist_start_bin
        )
        
        # Calculate bin edges and centers correctly
        bin_edges = bin_boundaries[hist_start_bin:-1]  # Exclude the last edge
        bin_centers = (bin_boundaries[hist_start_bin:-1] + bin_boundaries[hist_start_bin+1:]) / 2
        
        return MakeHistogramOutput(
            histograms=histograms,
            bin_edges=bin_edges,
            bin_centers=bin_centers
        )

    @profile
    def plot_diagnostics(self, output: MakeHistogramOutput, save_dir: Path) -> None:
        """Generate diagnostic plots.
        
        Args:
            output: Histogram calculation output
            save_dir: Directory to save plots
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 5))
        
        # 1. Mean histogram across all pixels (log scale)
        ax1 = fig.add_subplot(121)
        mean_hist = np.mean(output.histograms, axis=(1, 2))
        ax1.semilogy(output.bin_centers, mean_hist, 'b-')
        ax1.set_title('Mean Histogram Across Pixels (Log Scale)')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Counts')
        ax1.grid(True)
        
        # 2. 2D map of histogram total counts
        ax2 = fig.add_subplot(122)
        total_counts = np.sum(output.histograms, axis=0)
        im2 = ax2.imshow(total_counts, cmap='viridis')
        ax2.set_title('Histogram Total Counts Map')
        plt.colorbar(im2, ax=ax2, label='Total Counts')
        
        
        plt.tight_layout()
        plt.savefig(save_dir / 'make_histogram_diagnostics.png')
        plt.close()
