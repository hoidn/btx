# test_multi_peak_masks.py
import numpy as np
from pathlib import Path
from btx.processing.tests.functional.test_multi_peak import (
    GaussianPeak, 
    generate_multi_peak_data,
    plot_synthetic_data_diagnostics
)
from btx.processing.tasks.build_pump_probe_masks import BuildPumpProbeMasks
from btx.processing.btx_types import (
    BuildPumpProbeMasksInput,
    MakeHistogramOutput,
    CalculatePValuesOutput
)

def test_multi_peak_mask_generation():
    """Test BuildPumpProbeMasks with synthetic multi-peak data."""
    # Set up output directory
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'multi_peak_masks'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test peaks of varying sizes and amplitudes
    peaks = [
        GaussianPeak(center=(30, 30), sigma=8.0, amplitude=2.0),    # Large peak
        GaussianPeak(center=(70, 70), sigma=6.0, amplitude=3.0),    # Medium peak
        GaussianPeak(center=(20, 70), sigma=4.0, amplitude=4.0),    # Small peak
    ]
    
    # Generate synthetic data
    frames, histograms, lambda_map, true_masks = generate_multi_peak_data(
        peaks=peaks,
        base_counts=100.0,
        num_frames=1000,
        num_histogram_bins=50,
        histogram_range=(0, 300),
        background_roi=(0, 20, 0, 20),
        seed=42
    )
    
    # Plot synthetic data diagnostics
    plot_synthetic_data_diagnostics(
        frames, histograms, lambda_map, true_masks,
        save_dir,
        'synthetic_data.png'
    )
    
    # Calculate p-values using chi-squared test
    # (This is a simple example - you might want to use your actual p-value calculation)
    p_values = np.zeros((100, 100))
    background_hist = histograms[:, :20, :20].mean(axis=(1,2))  # Use background ROI
    
    for i in range(100):
        for j in range(100):
            # Simple chi-squared test between each pixel's histogram and background
            hist = histograms[:, i, j]
            chi2_stat = np.sum((hist - background_hist)**2 / (background_hist + 1e-10))
            p_values[i, j] = 1 - np.exp(-chi2_stat/2)  # Approximate p-value
    
    # Create input data structures
    histogram_output = MakeHistogramOutput(
        histograms=histograms,
        bin_edges=np.linspace(0, 300, 51),  # One more than bins
        bin_centers=np.linspace(3, 297, 50)  # Center of each bin
    )
    
    p_values_output = CalculatePValuesOutput(
        p_values=p_values,
        log_p_values=-np.log10(p_values),
        significance_threshold=0.05
    )
    
    # Configure mask builder
    config = {
        'setup': {
            'background_roi_coords': [0, 20, 0, 20]  # Same as used in synthetic data
        },
        'generate_masks': {
            'threshold': 0.05,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5,
            'max_peaks': 10,
            'min_peak_size': 10
        }
    }
    
    # Create and run mask builder
    mask_builder = BuildPumpProbeMasks(config)
    input_data = BuildPumpProbeMasksInput(
        config=config,
        histogram_output=histogram_output,
        p_values_output=p_values_output
    )
    
    output = mask_builder.run(input_data)
    
    # Generate diagnostics
    mask_builder.plot_diagnostics(output, save_dir)
    
    # Add comparison plot of true vs found masks
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('True vs Found Masks Comparison')
    
    # True masks
    combined_true = np.zeros((100, 100))
    for i, mask in enumerate(true_masks, 1):
        combined_true[mask] = i
    ax1.imshow(combined_true)
    ax1.set_title('True Peak Regions')
    
    # Found masks
    combined_found = np.zeros((100, 100))
    for i, pair in enumerate(output.mask_pairs, 1):
        combined_found[pair.signal_mask] = i
    ax2.imshow(combined_found)
    ax2.set_title('Found Signal Masks')
    
    plt.savefig(save_dir / 'mask_comparison.png')
    plt.close()
    
    # Print some statistics
    print("\nMask Generation Results:")
    print(f"Number of true peaks: {len(true_masks)}")
    print(f"Number of found peaks: {len(output.mask_pairs)}")
    
    # Compare mask sizes
    print("\nMask Sizes:")
    print("True masks:", [np.sum(mask) for mask in true_masks])
    print("Found masks:", [pair.size for pair in output.mask_pairs])

if __name__ == '__main__':
    test_multi_peak_mask_generation()
    print("\nTest complete. Check the diagnostic plots in the output directory.")
