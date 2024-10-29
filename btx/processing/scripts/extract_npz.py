#!/usr/bin/env python3
"""
HDF5 Data Extraction Utility for LCLS Data
Aligned with legacy implementation while maintaining modern structure
"""

import h5py
import numpy as np
from pathlib import Path


def apply_energy_filter(frames: np.ndarray, E0: float = 8.8, dE: float = 5.0) -> np.ndarray:
    """
    Filter pixel intensities based on energy bands.
    Zeros out pixels outside the allowed energy ranges.
    
    Args:
        frames: Raw detector frames
        E0: Central energy in keV (default 8.8)
        dE: Energy window in keV (default 5.0)
    """
    thresh_1, thresh_2 = E0-dE, E0+dE
    thresh_3, thresh_4 = 2*E0-dE, 2*E0+dE
    thresh_5, thresh_6 = 3*E0-dE, 3*E0+dE
    
    # Create mask for pixels outside allowed energy ranges
    invalid_mask = ((frames < thresh_1) |
                   ((frames > thresh_2) & (frames < thresh_3)) |
                   ((frames > thresh_4) & (frames < thresh_5)) |
                   (frames > thresh_6))
    
    # Zero out invalid pixels
    filtered_frames = frames.copy()
    filtered_frames[invalid_mask] = 0
    
    return filtered_frames


def extract_data_from_h5(
    filename: str, 
    output_dir: str, 
    use_timetool: bool = False,
    E0: float = 8.8,
    dE: float = 5.0
) -> Path:
    """
    Extract data from HDF5 file and save to NPZ format.
    
    Args:
        filename: Path to HDF5 file
        output_dir: Directory to save NPZ file
        use_timetool: Whether to apply timetool-based filtering
        E0: Central energy for filtering in keV (default 8.8)
        dE: Energy window for filtering in keV (default 5.0)
        
    Returns:
        Path: Path to saved NPZ file
    """
    # Path mappings in h5 file - aligned with legacy implementation
    data_paths = {
        'scanvar': 'enc/lasDelay',       # Corrected delay path
        'i0': 'ipm2/sum',                # I0 intensity
        'roi0': 'jungfrau1M/ROI_0_area', # Raw frames
        'roi0_roi': 'UserDataCfg/jungfrau1M/ROI_0__ROI_0_ROI',  # ROI definition
        'roi0_mask': 'UserDataCfg/jungfrau1M/mask',  # Updated mask path
        'tt_amp': 'tt/AMPL',             # Time tool amplitude
        'xpos': 'ipm2/xpos',             # Beam position
        'ypos': 'ipm2/ypos',             # Beam position
        'evr_code_90': 'evr/code_90',    # Laser on events
        'evr_code_91': 'evr/code_91',    # Laser off events
    }

    # Define filters with physical bounds
    filters = {
        'i0': [200, 20000],           # Intensity filter
        'xpos': [-0.45, 0.45],        # X position filter
        'ypos': [-1.6, 0.],           # Y position filter
    }
    
    # Add timetool filter if enabled
    if use_timetool:
        filters['tt_amp'] = [0.0, np.inf]

    print("Opening h5 file...")
    with h5py.File(filename, 'r') as h5:
        # Get ROI and mask information
        print("Setting up detector ROI and mask...")
        roi_indices = h5[data_paths['roi0_roi']][()]
        idx_tile = roi_indices[0,0]
        roi_slice_y = slice(roi_indices[1,0], roi_indices[1,1])
        roi_slice_x = slice(roi_indices[2,0], roi_indices[2,1])
        
        # Get correct mask for tile
        mask = h5[data_paths['roi0_mask']][idx_tile]
        roi_mask = mask[roi_slice_y, roi_slice_x]
        
        # 1. Get base masks using event codes
        print("Creating base masks from event codes...")
        laser_on = h5[data_paths['evr_code_90']][:] == 1
        laser_off = h5[data_paths['evr_code_91']][:] == 1
        
        # Initialize masks
        laser_on_mask = laser_on.copy()
        laser_off_mask = laser_off.copy()

        # 2. Apply additional filters
        print("Applying filters...")
        for key, (min_val, max_val) in filters.items():
            data = h5[data_paths[key]][:]
            value_filter = np.logical_and(data > min_val, data < max_val)
            
            # Apply filters based on type
            if key == 'tt_amp':
                # Timetool filter only applies to laser-on events
                laser_on_mask = np.logical_and(laser_on_mask, value_filter)
            else:
                # All other filters apply to both masks
                laser_on_mask = np.logical_and(laser_on_mask, value_filter)
                laser_off_mask = np.logical_and(laser_off_mask, value_filter)

        print(f"After filtering: {np.sum(laser_on_mask)} laser-on shots, "
              f"{np.sum(laser_off_mask)} laser-off shots")

        # 3. Get delay values
        print("Getting delay values...")
        delays = h5[data_paths['scanvar']][:]

        # 4. Get I0 values
        print("Getting I0 values...")
        i0 = h5[data_paths['i0']][:]

        # 5. Get detector frames and apply processing
        print("Loading and processing detector frames...")
        # Load raw frames
        frames = h5[data_paths['roi0']][:]
        print('SHAPE', frames.shape)
        
#        # Apply energy filter
#        print(f"Applying energy filter (E0={E0} keV, dE={dE} keV)...")
#        frames = apply_energy_filter(frames, E0, dE)
        
        # Apply ROI mask
        frames = frames * (~np.logical_not(roi_mask))

        # Prepare output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract run number from filename
        run_number = Path(filename).stem.split('Run')[1][:4]
        save_path = output_path / f'run{run_number}_extracted.npz'

        print(f"Saving data to {save_path}")
        np.savez(
            save_path,
            frames=frames,
            delays=delays,
            I0=i0,
            laser_on_mask=laser_on_mask,
            laser_off_mask=laser_off_mask,
            # Save filter values and mask for reference
            filter_values={str(k): v for k, v in filters.items()},
            roi_mask=roi_mask,
            energy_filter={'E0': E0, 'dE': dE}
        )

        print("Extraction complete!")
        return save_path


if __name__ == "__main__":
    # Example usage
    filename = "/sdf/data/lcls/ds/xpp/xppx1003221/hdf5/smalldata/xppx1003221_Run0195.h5"
    output_dir = "processed_xppx1003221"
    extract_data_from_h5(filename, output_dir, use_timetool=True)
