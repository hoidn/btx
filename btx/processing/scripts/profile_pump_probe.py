import cProfile
import pstats
from pstats import SortKey
from pathlib import Path
import numpy as np

from btx.processing.tasks import PumpProbeAnalysis
from btx.processing.btx_types import PumpProbeAnalysisInput

def create_test_input(input_data: PumpProbeAnalysisInput, n_frames: int = 1000) -> PumpProbeAnalysisInput:
    """Create smaller version of input data for profiling"""
    test_data = input_data.copy()
    test_data.load_data_output.data = test_data.load_data_output.data[:n_frames]
    test_data.load_data_output.I0 = test_data.load_data_output.I0[:n_frames]
    test_data.load_data_output.laser_on_mask = test_data.load_data_output.laser_on_mask[:n_frames]
    test_data.load_data_output.laser_off_mask = test_data.load_data_output.laser_off_mask[:n_frames]
    return test_data

def run_profiling(config: dict, input_data: PumpProbeAnalysisInput, output_dir: Path):
    """Run profiling on PumpProbeAnalysis"""
    
    # Create test dataset
    test_input = create_test_input(input_data)
    
    # Initialize analyzer
    pump_probe = PumpProbeAnalysis(config)
    
    # Profile execution
    profiler = cProfile.Profile()
    profiler.enable()
    
    output = pump_probe.run(test_input)
    pump_probe.plot_diagnostics(output, output_dir / "pump_probe")
    
    profiler.disable()
    
    # Save and print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)
    
    # Save full stats
    stats.dump_stats(output_dir / 'profile_results.prof')
    
    # Print summary to file
    with open(output_dir / 'profile_summary.txt', 'w') as f:
        stats.stream = f
        stats.print_stats(20)  # Top 20 time-consuming functions
        stats.print_callers(20)  # Show what's calling these functions

if __name__ == '__main__':
    from xppl1030522 import config, pump_probe_input, output_dir
    run_profiling(config, pump_probe_input, output_dir)
