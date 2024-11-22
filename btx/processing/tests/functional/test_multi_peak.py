import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import pytest

@dataclass
class GaussianPeak:
    """Define a 2D Gaussian peak with Poisson statistics."""
    center: Tuple[int, int]  # (x, y) center coordinates
    sigma: float  # Width parameter
    amplitude: float  # Peak multiplier for lambda
    
    def compute_lambda_contribution(self, rows: int, cols: int) -> np.ndarray:
        """Compute this peak's contribution to the lambda (rate) map."""
        y, x = np.ogrid[:rows, :cols]
        x_centered = x - self.center[1]
        y_centered = y - self.center[0]
        r_squared = x_centered*x_centered + y_centered*y_centered
        return self.amplitude * np.exp(-r_squared / (2 * self.sigma**2))

def generate_multi_peak_data(
    rows: int = 100,
    cols: int = 100,
    peaks: List[GaussianPeak] = None,
    base_counts: float = 100.0,
    num_frames: int = 1000,
    num_histogram_bins: int = 50,
    histogram_range: Tuple[float, float] = (0, 300),
    background_roi: Optional[Tuple[int, int, int, int]] = None,
    seed: int = 42,
    save_total_counts: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate synthetic data with multiple 2D Gaussian peaks and Poisson statistics.
    
    Args:
        rows: Number of rows in each frame
        cols: Number of columns in each frame
        peaks: List of GaussianPeak objects defining signals
        base_counts: Base lambda (counts/frame) for background
        num_frames: Number of frames to generate
        num_histogram_bins: Number of bins for histograms
        histogram_range: (min, max) for histogram binning
        background_roi: (x1, x2, y1, y2) for background ROI
        seed: Random seed for reproducibility
        save_total_counts: Optional path to save total counts image
        
    Returns:
        frames: Array of shape (num_frames, rows, cols)
        histograms: Array of shape (num_histogram_bins, rows, cols)
        lambda_map: Array of shape (rows, cols) showing true rate
        true_masks: List of binary masks for each peak
    """
    rng = np.random.default_rng(seed)
    
    if peaks is None:
        peaks = []
        
    if background_roi is None:
        background_roi = (0, rows//4, 0, cols//4)
        
    # Generate base lambda map
    lambda_map = np.full((rows, cols), base_counts)
    
    # Add peaks
    for peak in peaks:
        lambda_map += peak.compute_lambda_contribution(rows, cols) * base_counts
        
    # Generate frames with Poisson noise
    frames = rng.poisson(lam=lambda_map, size=(num_frames, rows, cols))
    
    # Compute histograms
    histograms = np.zeros((num_histogram_bins, rows, cols))
    bin_edges = np.linspace(histogram_range[0], histogram_range[1], num_histogram_bins + 1)
    
    for i in range(rows):
        for j in range(cols):
            hist, _ = np.histogram(frames[:, i, j], bins=bin_edges)
            histograms[:, i, j] = hist
            
    # Save total counts image if requested
    if save_total_counts is not None:
        total_counts = frames.sum(axis=0)
        plt.figure(figsize=(8, 8))
        plt.imshow(total_counts)
        plt.colorbar(label='Total Counts')
        plt.title(f'Total Counts Over {num_frames} Frames')
        plt.savefig(save_total_counts)
        plt.close()
            
    # Generate ground truth masks for each peak
    true_masks = []
    for peak in peaks:
        # Create mask where lambda is significantly elevated above background
        peak_contribution = peak.compute_lambda_contribution(rows, cols)
        # Consider points where rate is elevated by at least 10% of peak's amplitude
        threshold = 0.1 * peak.amplitude
        mask = peak_contribution > threshold
        true_masks.append(mask)
    
    return frames, histograms, lambda_map, true_masks

def plot_synthetic_data_diagnostics(
    frames: np.ndarray,
    histograms: np.ndarray,
    lambda_map: np.ndarray,
    true_masks: List[np.ndarray],
    save_dir: Path,
    filename: str = 'synthetic_data_diagnostics.png'
) -> None:
    """Generate diagnostic plots for synthetic data."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Synthetic Data Diagnostics')
    
    # Plot lambda map
    im0 = axes[0, 0].imshow(lambda_map)
    axes[0, 0].set_title('True λ Map')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Plot example frame
    frame_idx = frames.shape[0] // 2  # Middle frame
    im1 = axes[0, 1].imshow(frames[frame_idx])
    axes[0, 1].set_title(f'Example Frame (#{frame_idx})')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot mean frame
    mean_frame = frames.mean(axis=0)
    im2 = axes[0, 2].imshow(mean_frame)
    axes[0, 2].set_title('Mean Frame')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Plot histogram total counts
    hist_sums = histograms.sum(axis=0)
    im3 = axes[1, 0].imshow(hist_sums)
    axes[1, 0].set_title('Histogram Total Counts')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot true masks
    combined_mask = np.zeros_like(lambda_map)
    for i, mask in enumerate(true_masks, 1):
        combined_mask[mask] = i
    im4 = axes[1, 1].imshow(combined_mask)
    axes[1, 1].set_title('True Peak Masks')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Plot example histogram for peak and background
    if true_masks:
        peak_center = np.where(true_masks[0])[0][0], np.where(true_masks[0])[1][0]
        bg_point = 0, 0  # Corner point for background
        
        axes[1, 2].plot(histograms[:, peak_center[0], peak_center[1]], 
                       label='Peak', alpha=0.7)
        axes[1, 2].plot(histograms[:, bg_point[0], bg_point[1]], 
                       label='Background', alpha=0.7)
        axes[1, 2].set_title('Example Histograms')
        axes[1, 2].legend()
        axes[1, 2].set_yscale('log')
    else:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / filename)
    plt.close()

def test_synthetic_data_generation():
    """Test the synthetic data generation with multiple peaks."""
    # Set parameters consistently
    base_counts = 100.0

    # Define test peaks
    peaks = [
        GaussianPeak(center=(30, 30), sigma=8.0, amplitude=2.0),    # Large peak
        GaussianPeak(center=(70, 70), sigma=6.0, amplitude=3.0),    # Medium peak
        GaussianPeak(center=(20, 70), sigma=4.0, amplitude=4.0),    # Small peak
    ]
    
    # Generate data
    frames, histograms, lambda_map, true_masks = generate_multi_peak_data(
        peaks=peaks,
        base_counts=base_counts,  # Use the defined parameter
        num_frames=1000,
        seed=42,
        save_total_counts=save_dir / 'total_counts.png'
    )
    
    # Basic shape checks
    assert frames.shape[0] == 1000
    assert frames.shape[1:] == (100, 100)
    assert histograms.shape[1:] == (100, 100)
    assert lambda_map.shape == (100, 100)
    assert len(true_masks) == len(peaks)
    
    # Statistical checks
    # 1. Background region should have mean close to base_counts
    bg_mean = frames[:, :10, :10].mean()  # Use corner as background
    assert 95 < bg_mean < 105  # Within 5% of base_counts=100
    
    # 2. Check peak centers (where we know the exact expected value)
    for peak in peaks:
        x, y = peak.center
        center_mean = frames[:, x, y].mean()
        expected_center = base_counts * (1 + peak.amplitude)
        # Allow 10% tolerance due to Poisson statistics
        assert abs(center_mean - expected_center) < 0.1 * expected_center, \
            f"Peak center mean {center_mean:.1f} differs from expected {expected_center:.1f}"
    
    # 3. Check Poisson statistics in background
    bg_var = frames[:, :10, :10].var()
    # For Poisson, mean ≈ variance
    assert abs(bg_mean - bg_var) < 0.1 * bg_mean
    
    # 4. Check mask sizes are in descending order
    mask_sizes = [np.sum(mask) for mask in true_masks]
    assert mask_sizes == sorted(mask_sizes, reverse=True), \
        "Masks not in descending size order"
    
    # Generate diagnostic plots
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'synthetic_data'
    plot_synthetic_data_diagnostics(frames, histograms, lambda_map, true_masks, save_dir)

if __name__ == '__main__':
    # Run test and generate plots
    test_synthetic_data_generation()
    print("\nTest complete. Check the diagnostic plots in the output directory.")
