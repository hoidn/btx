# test_histogram.py
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

from btx.processing.tasks.make_histogram import MakeHistogram
from btx.processing.btx_types import MakeHistogramInput, LoadDataOutput
from btx.processing.tests.functional.data_generators import generate_synthetic_frames

from typing import Dict, Any, List, Tuple, NamedTuple
from numba.core.errors import NumbaWarning
import warnings

# Suppress Numba warnings
warnings.filterwarnings('ignore', category=NumbaWarning)

class BenchmarkResult(NamedTuple):
    """Store results for a single benchmark run."""
    mean_time: float
    std_time: float
    points_per_second: float
    is_valid: bool

class DataSize(NamedTuple):
    """Define a test data size configuration."""
    frames: int
    rows: int
    cols: int
    
    def total_points(self) -> int:
        return self.frames * self.rows * self.cols
    
    def __str__(self) -> str:
        return f"{self.frames}×{self.rows}×{self.cols}"

def generate_test_data(size: DataSize, order: str = 'F', seed: int = 42) -> np.ndarray:
    """Generate synthetic test data with specified size and memory layout."""
    rng = np.random.default_rng(seed)
    
    # Generate bimodal distribution
    dist1 = rng.normal(10, 1, (size.frames // 2, size.rows, size.cols))
    dist2 = rng.normal(15, 2, (size.frames - size.frames // 2, size.rows, size.cols))
    
    # Set memory layout
    data = np.asarray(np.concatenate([dist1, dist2], axis=0), order=order)
    
    return data

def validate_histogram(histograms: np.ndarray, bin_centers: np.ndarray) -> bool:
    """Validate histogram results against expected peaks."""
    expected_peaks = [10, 15]  # Based on our test data generation
    mean_hist = np.mean(histograms, axis=(1, 2))
    
    # Find peaks
    peaks = []
    for i in range(1, len(mean_hist) - 1):
        if mean_hist[i] > mean_hist[i-1] and mean_hist[i] > mean_hist[i+1]:
            peaks.append(bin_centers[i])
    
    # Check if we found peaks near the expected values
    found_peaks = 0
    for expected in expected_peaks:
        for peak in peaks:
            if abs(peak - expected) < 1.0:
                found_peaks += 1
                break
    
    return found_peaks == len(expected_peaks)


# Create config
config = {
    'make_histogram': {
        'bin_boundaries': np.arange(5, 30, 0.2),
        'hist_start_bin': 1
    }
}

def test_make_histogram_visual():
    """Generate visual diagnostic plots for manual inspection."""
    save_dir = Path(__file__).parent.parent.parent / 'temp' / 'diagnostic_plots' / 'make_histogram'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visual test plots in: {save_dir}")
    
    # Generate synthetic data
    num_frames = 1000
    rows = cols = 100
    data, I0, delays, on_mask, off_mask = generate_synthetic_frames(
        num_frames, rows, cols
    )
    
    # Scale data to match expected range
    data = 5 + (25 * (data - data.min()) / (data.max() - data.min()))
    
    # Create LoadDataOutput
    load_data_output = LoadDataOutput(
        data=data,
        I0=I0,
        laser_delays=delays,
        laser_on_mask=on_mask,
        laser_off_mask=off_mask,
        binned_delays=delays
    )
    
    
    # Create input
    input_data = MakeHistogramInput(
        config=config,
        load_data_output=load_data_output
    )
    
    # Run task
    task = MakeHistogram(config)

def run_benchmark_iteration(processor: MakeHistogram, data: np.ndarray, 
                          config: Dict[str, Any]) -> Tuple[float, np.ndarray]:
    """Run a single benchmark iteration."""
    input_data = MakeHistogramInput(config,
                                LoadDataOutput(data, None, None, None, None, None))
    
    start_time = time.time()
    output = processor.run(input_data)
    elapsed = time.time() - start_time
    
    return elapsed, output.histograms

def benchmark_layout(size: DataSize, order: str, config: Dict[str, Any], 
                    n_iterations: int = 3) -> BenchmarkResult:
    """Run benchmark for a specific data size and memory layout."""
    print(f"\nTesting {order}-order layout for size {size} = {size.total_points():,} points")
    
    # Generate test data
    print("Generating test data...")
    data = generate_test_data(size, order=order)
    print(f"Data is {order}-contiguous: {data.flags[f'{order}_CONTIGUOUS']}")
    
    # Create processor
    processor = MakeHistogram(config)
    
    # Run iterations
    times = []
    is_valid = False
    
    print(f"Running {n_iterations} iterations...")
    for i in range(n_iterations):
        elapsed, histograms = run_benchmark_iteration(processor, data, config)
        times.append(elapsed)
        
        # Validate on first iteration
        if i == 0:
            bin_centers = (config['make_histogram']['bin_boundaries'][:-1] + 
                         config['make_histogram']['bin_boundaries'][1:]) / 2
            is_valid = validate_histogram(histograms, bin_centers)
            if not is_valid:
                print("WARNING: Results validation failed!")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    points_per_second = size.total_points() / mean_time
    
    print(f"  Mean time: {mean_time:.2f}s ± {std_time:.2f}s")
    print(f"  Rate: {points_per_second:,.0f} points/second")
    
    return BenchmarkResult(mean_time, std_time, points_per_second, is_valid)

def plot_comparative_results(results: Dict[DataSize, Dict[str, BenchmarkResult]], 
                           output_file: str = 'histogram_benchmark.png'):
    """Generate comparative performance plots."""
    plt.figure(figsize=(15, 6))
    
    # Processing rates plot
    plt.subplot(1, 2, 1)
    sizes = list(results.keys())
    x = np.arange(len(sizes))
    width = 0.35
    
    # Convert to millions of points per second
    c_rates = [results[size]['C'].points_per_second / 1e6 for size in sizes]
    f_rates = [results[size]['F'].points_per_second / 1e6 for size in sizes]
    
    plt.bar(x - width/2, c_rates, width, label='C-order', color='blue', alpha=0.7)
    plt.bar(x + width/2, f_rates, width, label='F-order', color='red', alpha=0.7)
    
    plt.xlabel('Data Size')
    plt.ylabel('Processing Rate (M points/second)')
    plt.title('Processing Rate Comparison')
    plt.xticks(x, [str(size) for size in sizes], rotation=45)
    plt.legend()
    
    # Speedup plot
    plt.subplot(1, 2, 2)
    speedups = [results[size]['C'].mean_time / results[size]['F'].mean_time 
                for size in sizes]
    plt.bar(x, speedups, color='green', alpha=0.7)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    
    plt.xlabel('Data Size')
    plt.ylabel('Speedup (F-order vs C-order)')
    plt.title('F-order Speedup Factor')
    plt.xticks(x, [str(size) for size in sizes], rotation=45)
    
    # Add speedup labels
    for i, speedup in enumerate(speedups):
        plt.text(i, speedup, f'{speedup:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def run_comparative_benchmark():
    """Run complete comparative benchmark suite."""
    # Test configurations
    sizes = [
        DataSize(1000, 100, 100),   # Small: 10M points
        DataSize(5000, 200, 200),   # Medium: 200M points
        DataSize(10000, 200, 200),  # Large: 400M points
    ]
    
    config = {
        'make_histogram': {
            'bin_boundaries': np.arange(0, 30, 0.2),
            'hist_start_bin': 1
        }
    }
    
    results = {}
    
    for size in sizes:
        print(f"\nBenchmarking size: {size}")
        results[size] = {}
        
        # Test both memory layouts
        for order in ['C', 'F']:
            results[size][order] = benchmark_layout(size, order, config)
        
        # Calculate and display speedup
        speedup = results[size]['C'].mean_time / results[size]['F'].mean_time
        print(f"\nF-order speedup: {speedup:.2f}x")
    
    # Generate plots
    plot_comparative_results(results)
    
    return results

def print_summary(results: Dict[DataSize, Dict[str, BenchmarkResult]]):
    """Print final benchmark summary."""
    print("\nFinal Summary:")
    print("=" * 60)
    for size in results:
        print(f"\nSize: {size}")
        print(f"C-order: {results[size]['C'].mean_time:.2f}s")
        print(f"F-order: {results[size]['F'].mean_time:.2f}s")
        print(f"Speedup: {results[size]['C'].mean_time / results[size]['F'].mean_time:.2f}x")
        print(f"Results valid: C={results[size]['C'].is_valid}, F={results[size]['F'].is_valid}")

if __name__ == '__main__':
    print("Running Histogram Implementation Comparative Benchmark")
    print("-" * 60)
    
    try:
        results = run_comparative_benchmark()
        print("\nBenchmark completed successfully")
        print("Results plot saved as: histogram_benchmark.png")
        print_summary(results)
        
    except Exception as e:
        print(f"Benchmark failed with error: {str(e)}")
        raise
