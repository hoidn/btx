from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import functools
import hashlib
import random
from numba.core.errors import NumbaWarning
import warnings

# Suppress Numba TBB warning
warnings.filterwarnings('ignore', category=NumbaWarning)

class MakeHistogramInput:
    def __init__(self, load_data_output):
        self.load_data_output = load_data_output

class MakeHistogramOutput:
    def __init__(self, histograms, bin_edges, bin_centers):
        self.histograms = histograms
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers

def memoize_subsampled(func):
    """Memoize a function by creating a hashable key using deterministically subsampled data."""
    cache = {}

    @functools.wraps(func)
    def wrapper(self, data, *args, **kwargs):
        shape_str = str(data.shape)
        seed_value = int(hashlib.sha256(shape_str.encode()).hexdigest(), 16) % 10**8
        random.seed(seed_value)

        subsample_size = min(100, data.shape[0])
        subsample_indices = random.sample(range(data.shape[0]), subsample_size)
        subsample = data[subsample_indices]

        hashable_key = hashlib.sha256(subsample.tobytes()).hexdigest()

        if hashable_key in cache:
            return cache[hashable_key]

        result = func(self, data, *args, **kwargs)
        cache[hashable_key] = result

        return result

    return wrapper

@jit(nopython=True)
def _calculate_histograms_numba(data, bin_boundaries, hist_start_bin, bins, rows, cols):
    """Numba-optimized histogram calculation optimized for F-order arrays."""
    hist_shape = (bins, rows, cols)
    histograms = np.zeros(hist_shape, dtype=np.float64)
    
    # Process one frame at a time, which is contiguous in memory with F-order
    for frame in range(data.shape[0]):
        frame_data = data[frame]  # This slice is contiguous in F-order
        for row in range(rows):
            for col in range(cols):
                value = frame_data[row, col]
                bin_idx = np.searchsorted(bin_boundaries, value)
                if bin_idx < bins:
                    histograms[bin_idx, row, col] += 1
                else:
                    histograms[hist_start_bin, row, col] += 1
    
    histograms += 1e-9
    return histograms[hist_start_bin:, :, :]

class MakeHistogram:
    """Generate histograms from XPP data with F-order optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize histogram generation task."""
        self.config = config
        
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
            data: 3D array (frames, rows, cols) in Fortran order
            bin_boundaries: Array of histogram bin boundaries
            hist_start_bin: Index of first bin to include
            
        Returns:
            3D array of histograms (bins, rows, cols)
        """
        # Validate inputs
        assert isinstance(data, np.ndarray), f"Expected numpy array, got {type(data)}"
        assert data.ndim == 3, f"Expected 3D array, got shape {data.shape}"
        assert data.flags.f_contiguous, "Input array must be F-contiguous"
        
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

    def run(self, input_data: MakeHistogramInput) -> MakeHistogramOutput:
        """Run histogram generation."""
        hist_config = self.config['make_histogram']
        bin_boundaries = np.array(hist_config['bin_boundaries'])
        hist_start_bin = hist_config['hist_start_bin']
        
        # Ensure data is in F-order
        data = np.asarray(input_data.load_data_output.data, order='F')
        
        histograms = self._calculate_histograms(
            data,
            bin_boundaries,
            hist_start_bin
        )
        
        bin_edges = bin_boundaries[hist_start_bin:-1]
        bin_centers = (bin_edges + bin_boundaries[hist_start_bin+1:]) / 2
        
        return MakeHistogramOutput(
            histograms=histograms,
            bin_edges=bin_edges,
            bin_centers=bin_centers
        )

    def plot_diagnostics(self, output: MakeHistogramOutput, save_dir: Path) -> None:
        """Generate diagnostic plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(15, 5))
        
        # Mean histogram across all pixels (log scale)
        ax1 = fig.add_subplot(121)
        mean_hist = np.mean(output.histograms, axis=(1, 2))
        ax1.semilogy(output.bin_centers, mean_hist, 'b-')
        ax1.set_title('Mean Histogram Across Pixels (Log Scale)')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Counts')
        ax1.grid(True)
        
        # 2D map of total counts
        ax2 = fig.add_subplot(122)
        total_counts = np.sum(output.histograms, axis=0)
        im2 = ax2.imshow(total_counts, cmap='viridis')
        ax2.set_title('Total Counts Map')
        plt.colorbar(im2, ax=ax2, label='Total Counts')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'make_histogram_diagnostics.png')
        plt.close()
