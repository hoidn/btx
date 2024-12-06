from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from contextlib import contextmanager

from btx.processing.tests.functional.test_multi_peak import (
    GaussianPeak,
    generate_multi_peak_data
)

from btx.processing.tasks import (
    LoadData, MakeHistogram, MeasureEMD,
    CalculatePValues, BuildPumpProbeMasks, PumpProbeAnalysis
)

from btx.processing.btx_types import (
    LoadDataInput, LoadDataOutput,
    MakeHistogramInput, MeasureEMDInput,
    CalculatePValuesInput, BuildPumpProbeMasksInput,
    PumpProbeAnalysisInput
)

@contextmanager
def safe_plot():
    """Context manager for safe plot generation with cleanup."""
    try:
        yield
    except Exception as e:
        print(f"Warning: Plot generation failed - {str(e)}")
    finally:
        plt.close()

def create_synthetic_input(
    peaks: list[GaussianPeak],
    num_frames: int = 1000,
    rows: int = 100,
    cols: int = 100,
    base_counts: float = 100.0,
    seed: int = 42
) -> Tuple[LoadDataInput, np.ndarray]:
    """Create synthetic input data for XPP pipeline.
    
    Parameters
    ----------
    peaks : list[GaussianPeak]
        List of Gaussian peaks to generate
    num_frames : int
        Number of frames to generate
    rows, cols : int
        Frame dimensions
    base_counts : float
        Base Poisson rate for background
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    LoadDataInput
        Input data for pipeline
    np.ndarray
        True lambda map showing signal distribution
    """
    
    print("\nGenerating synthetic data...")
    print(f"- Number of frames: {num_frames}")
    print(f"- Image size: {rows}x{cols}")
    print(f"- Base counts: {base_counts}")
    print(f"- Number of peaks: {len(peaks)}")
    
    # Generate synthetic frames only
    frames, _, lambda_map, _ = generate_multi_peak_data(
        peaks=peaks,
        base_counts=base_counts,
        num_frames=num_frames,
        rows=rows,
        cols=cols,
        seed=seed
    )
    
    # Verify frame shapes and values
    assert frames.shape == (num_frames, rows, cols), \
        f"Unexpected frame shape: {frames.shape}"
    assert np.all(frames >= 0), "Negative values in frames"
    assert np.all(np.isfinite(frames)), "Non-finite values in frames"
    
    # Create synthetic metadata
    delays = np.zeros(num_frames)  # All at t=0 since we don't care about time
    I0 = np.full(num_frames, 1000.0)  # Constant I0
    
    # Alternate laser on/off
    laser_on_mask = np.zeros(num_frames, dtype=bool)
    laser_on_mask[::2] = True
    laser_off_mask = ~laser_on_mask
    
    # Compute data statistics for histogram binning
    data_min = np.percentile(frames, 1)  # Robust min
    data_max = np.percentile(frames, 99)  # Robust max
    print("\nData statistics:")
    print(f"Frame value range: [{frames.min():.1f}, {frames.max():.1f}]")
    print(f"Robust range (1-99%%): [{data_min:.1f}, {data_max:.1f}]")
    print(f"Mean counts: {frames.mean():.1f}")
    print(f"Laser on/off frames: {np.sum(laser_on_mask)}/{np.sum(laser_off_mask)}")
    
    # Create LoadDataInput - no ROI needed for synthetic data
    input_data = LoadDataInput(
        config=config,
        data=frames,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=laser_on_mask,
        laser_off_mask=laser_off_mask
    )
    
    return input_data, lambda_map, (data_min, data_max)

def validate_step_output(name: str, data: np.ndarray, expected_shape: tuple = None):
    """Validate output from each pipeline step with detailed error messages."""
    print(f"\nValidating {name}:")
    print(f"Shape: {data.shape}")
    print(f"Range: [{data.min():.2e}, {data.max():.2e}]")
    print(f"Mean: {data.mean():.2e}")
    
    if expected_shape:
        assert data.shape == expected_shape, \
            f"{name}: Shape mismatch. Expected {expected_shape}, got {data.shape}"
    
    assert np.all(np.isfinite(data)), \
        f"{name}: Contains non-finite values"
    
    if isinstance(data, np.ndarray) and data.dtype == bool:
        mask_coverage = np.mean(data) * 100
        print(f"Mask coverage: {mask_coverage:.1f}%")

# Configuration - specifically tuned for synthetic data
config = {
    'setup': {
        'run': 999,  # Synthetic run number
        'exp': 'synthetic',
        'background_roi_coords': [0, 20, 0, 20]  # Corner ROI for background
    },
    'load_data': {
        'time_bin': 2.0  # Not used but kept for compatibility
    },
    'make_histogram': {
        # Bin boundaries set after data generation
        'hist_start_bin': 1
    },
    'calculate_emd': {
        'num_permutations': 1000
    },
    'calculate_pvalues': {
        'significance_threshold': 0.05
    },
    'generate_masks': {
        'threshold': 0.05,
        'bg_mask_mult': 2.0,
        'bg_mask_thickness': 5
    },
    'pump_probe_analysis': {
        'min_count': 2,
        'significance_level': 0.05
    }
}

if __name__ == "__main__":
    # Setup output directory
    output_dir = Path("synthetic_pipeline_results")
    output_dir.mkdir(exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    
    # Define synthetic peaks with different characteristics
    peaks = [
        # Large, broad peak with moderate amplitude
        GaussianPeak(center=(30, 30), sigma=8.0, amplitude=2.0),
        # Medium peak with higher amplitude
        GaussianPeak(center=(70, 70), sigma=6.0, amplitude=3.0),
        # Small, intense peak
        GaussianPeak(center=(20, 70), sigma=4.0, amplitude=4.0),
    ]
    
    # Generate raw frame data and get data range for histogram binning
    load_data_input, lambda_map, (data_min, data_max) = create_synthetic_input(peaks)
    
    # Set histogram bins based on actual data range
    bin_width = (data_max - data_min) / 50  # 50 bins across data range
    config['make_histogram']['bin_boundaries'] = np.arange(
        data_min - bin_width,  # Start below min
        data_max + 2*bin_width,  # End above max
        bin_width
    )
    
    print("\nRunning pipeline...")
    
    # 1. Load/process raw frames
    print("\nProcessing raw frames...")
    load_data = LoadData(config)
    load_data_output = load_data.run(load_data_input)
    validate_step_output("load_data", load_data_output.data, 
                        load_data_input.data.shape)
    
    # 2. Create histograms
    print("\nCreating histograms...")
    make_histogram = MakeHistogram(config)
    histogram_input = MakeHistogramInput(
        config=config,
        load_data_output=load_data_output
    )
    histogram_output = make_histogram.run(histogram_input)
    validate_step_output("histograms", histogram_output.histograms)
    with safe_plot():
        make_histogram.plot_diagnostics(histogram_output, diagnostics_dir / "make_histogram")

    # 3. Calculate EMD
    print("\nMeasuring EMD...")
    measure_emd = MeasureEMD(config)
    emd_input = MeasureEMDInput(
        config=config,
        histogram_output=histogram_output
    )
    emd_output = measure_emd.run(emd_input)
    validate_step_output("emd", emd_output.emd_values, 
                        (load_data_output.data.shape[1], 
                         load_data_output.data.shape[2]))
    with safe_plot():
        measure_emd.plot_diagnostics(emd_output, diagnostics_dir / "measure_emd")

    # 4. Calculate p-values
    print("\nCalculating p-values...")
    calculate_pvalues = CalculatePValues(config)
    pvalues_input = CalculatePValuesInput(
        config=config,
        emd_output=emd_output
    )
    pvalues_output = calculate_pvalues.run(pvalues_input)
    validate_step_output("p_values", pvalues_output.p_values, 
                        emd_output.emd_values.shape)
    with safe_plot():
        calculate_pvalues.plot_diagnostics(pvalues_output, diagnostics_dir / "calculate_pvalues")

    # 5. Build masks
    print("\nBuilding masks...")
    build_masks = BuildPumpProbeMasks(config)
    masks_input = BuildPumpProbeMasksInput(
        config=config,
        histogram_output=histogram_output,
        p_values_output=pvalues_output
    )
    masks_output = build_masks.run(masks_input)
    validate_step_output("signal_mask", masks_output.signal_mask.astype(float), 
                        pvalues_output.p_values.shape)
    with safe_plot():
        build_masks.plot_diagnostics(masks_output, diagnostics_dir / "build_masks")

    # 6. Run pump-probe analysis
    print("\nRunning pump-probe analysis...")
    pump_probe = PumpProbeAnalysis(config)
    pump_probe_input = PumpProbeAnalysisInput(
        config=config,
        load_data_output=load_data_output,
        masks_output=masks_output
    )
    pump_probe_output = pump_probe.run(pump_probe_input)
    with safe_plot():
        pump_probe.plot_diagnostics(pump_probe_output, diagnostics_dir / "pump_probe")

    # Save results summary
    print("\nResults Summary:")
    print(f"Number of significant pixels: {np.sum(pvalues_output.p_values < config['calculate_pvalues']['significance_threshold'])}")
    print(f"Signal mask size: {np.sum(masks_output.signal_mask)} pixels")
    print(f"Background mask size: {np.sum(masks_output.background_mask)} pixels")
    print(f"Signal/background ratio: {np.sum(masks_output.signal_mask) / np.sum(masks_output.background_mask):.2f}")
    
    # Generate comparison visualization
    with safe_plot():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True signal distribution
        im1 = ax1.imshow(lambda_map)
        ax1.set_title('True Signal Distribution (λ)')
        plt.colorbar(im1, ax=ax1)
        
        # Found masks
        combined_mask = np.zeros_like(lambda_map)
        combined_mask[masks_output.signal_mask] = 1
        combined_mask[masks_output.background_mask] = 0.5
        im2 = ax2.imshow(combined_mask)
        ax2.set_title('Found Masks\n(Signal=1, Background=0.5)')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(diagnostics_dir / "truth_comparison.png")

    print(f"\nProcessing complete! Results saved in: {output_dir}")
    print("\nCheck diagnostic plots to verify mask generation behavior.")
