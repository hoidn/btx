import numpy as np
from pathlib import Path
from btx.processing.tests.functional.test_multi_peak import (
    GaussianPeak, 
    generate_multi_peak_data,
    plot_synthetic_data_diagnostics
)
from btx.processing.btx_types import (
    BuildPumpProbeMasksInput,
    BuildPumpProbeMasksOutput,
    SignalMaskStages,
    MakeHistogramOutput,
    CalculatePValuesOutput
)
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

@dataclass
class SignalBackgroundMaskPair:
    """Pair of signal and background masks with size for sorting."""
    signal_mask: np.ndarray
    background_mask: np.ndarray
    size: int

@dataclass
class SignalMaskStages:
    """Track intermediate stages of mask generation."""
    initial: List[np.ndarray]  # All significant clusters (p < threshold)
    filtered: List[np.ndarray]  # After filtering
    final: List[np.ndarray]    # Final masks

@dataclass
class BuildPumpProbeMasksOutput:
    """Output containing mask pairs and intermediate stages."""
    mask_pairs: List[SignalBackgroundMaskPair]
    intermediate_masks: SignalMaskStages

class BuildPumpProbeMasks:
    """Generate signal and background masks from p-values with multi-peak support.
    
    Signal regions are identified as connected components where p_values < threshold,
    indicating significant deviation from the null hypothesis. Regions are ordered
    by size (number of pixels) with the largest region appearing first in the
    output list.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mask generation task.
        
        Args:
            config: Dictionary containing:
                setup:
                    background_roi_coords: [x1, x2, y1, y2] Reference background region
                generate_masks:
                    threshold: Upper bound for significant p-values (default: 0.05)
                    bg_mask_mult: Background mask size multiplier (default: 2.0)
                    bg_mask_thickness: Separation between signal and background (default: 5)
                    max_peaks: Maximum number of signal peaks to return (default: 10)
                    min_peak_size: Minimum pixels for valid signal cluster (default: 10)
        """
        self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if 'setup' not in self.config:
            raise ValueError("Missing 'setup' section in config")
            
        if 'background_roi_coords' not in self.config['setup']:
            raise ValueError("Missing background_roi_coords in setup")
            
        roi = self.config['setup']['background_roi_coords']
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("background_roi_coords must be [x1, x2, y1, y2]")
        
        # Validate generate_masks section
        if 'generate_masks' not in self.config:
            self.config['generate_masks'] = {}
            
        masks_config = self.config['generate_masks']
        
        # Set defaults if not provided
        defaults = {
            'threshold': 0.05,
            'bg_mask_mult': 2.0,
            'bg_mask_thickness': 5,
            'max_peaks': 10,
            'min_peak_size': 10
        }
        
        for key, default_value in defaults.items():
            if key not in masks_config:
                masks_config[key] = default_value
                
        # Validate values
        if not 0 < masks_config['threshold'] < 1:
            raise ValueError("threshold must be between 0 and 1")
            
        if masks_config['bg_mask_mult'] <= 0:
            raise ValueError("bg_mask_mult must be positive")
            
        if masks_config['bg_mask_thickness'] <= 0:
            raise ValueError("bg_mask_thickness must be positive")
            
        if masks_config['max_peaks'] < 1:
            raise ValueError("max_peaks must be positive")
            
        if masks_config['min_peak_size'] < 1:
            raise ValueError("min_peak_size must be positive")

    def _identify_signal_clusters(
        self,
        p_values: np.ndarray,
        threshold: float,
        background_roi: Tuple[int, int, int, int]
    ) -> List[np.ndarray]:
        """Find all clusters where p_values < threshold (significant regions).
        
        Args:
            p_values: Array of p-values
            threshold: Significance threshold
            background_roi: (x1, x2, y1, y2) coordinates of background ROI
            
        Returns:
            List of binary masks for each cluster, sorted by size (descending)
        """
        # Create mask of significant pixels
        significant_pixels = p_values < threshold
        
        # Exclude background ROI region
        x1, x2, y1, y2 = background_roi
        significant_pixels[x1:x2, y1:y2] = False
        
        # Label connected components
        labeled_array, num_features = ndimage.label(significant_pixels)
        
        if num_features == 0:
            return []
            
        # Get sizes and masks for each cluster
        clusters = []
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            size = np.sum(mask)
            if size >= self.config['generate_masks']['min_peak_size']:
                clusters.append((mask, size))
                
        # Sort by size descending and extract masks
        clusters.sort(key=lambda x: x[1], reverse=True)
        
        # Limit number of clusters
        max_peaks = self.config['generate_masks']['max_peaks']
        clusters = clusters[:max_peaks]
        
        return [mask for mask, _ in clusters]

    def _filter_negative_clusters(
        self,
        cluster_array: np.ndarray,
        data: np.ndarray,
        min_size: int = 10
    ) -> np.ndarray:
        """Filter out small negative clusters.
        
        Args:
            cluster_array: Binary array of clusters
            data: Original histogram data
            min_size: Minimum cluster size to keep
            
        Returns:
            Filtered binary mask
        """
        # Rectify orientation first
        cluster_array = self._rectify_filter_mask(cluster_array, data)
        
        # Invert array to work with negative clusters
        inverted_array = np.logical_not(cluster_array)
        
        # Label inverted regions
        labeled_array, _ = ndimage.label(inverted_array)
        
        # Count size of each cluster
        cluster_sizes = np.bincount(labeled_array.ravel())
        
        # Find small clusters
        small_clusters = np.where(cluster_sizes < min_size)[0]
        
        # Create mask of small clusters
        small_cluster_mask = np.isin(labeled_array, small_clusters)
        
        # Return original mask with small negative clusters filled
        return np.logical_or(cluster_array, small_cluster_mask)

    def _rectify_filter_mask(self, mask: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Rectify mask orientation based on data values.
        
        Args:
            mask: Binary mask to rectify
            data: Original histogram data
            
        Returns:
            Rectified binary mask
        """
        if mask.sum() == 0:
            return ~mask
            
        imgs_sum = data.sum(axis=0)
        mean_1 = imgs_sum[mask].mean()
        mean_0 = imgs_sum[~mask].mean()
        
        return ~mask if mean_1 < mean_0 else mask

    def _process_clusters(
        self,
        clusters: List[np.ndarray],
        data: np.ndarray
    ) -> List[np.ndarray]:
        """Process each significant cluster.
        
        Args:
            clusters: List of binary masks for each cluster
            data: Original histogram data
            
        Returns:
            List of processed binary masks
        """
        processed_clusters = []
        for cluster in clusters:
            # Apply filtering
            filtered = self._filter_negative_clusters(
                cluster,
                data,
                min_size=self.config['generate_masks']['min_peak_size']
            )
            
            # Keep only largest component
            labeled_array, num_features = ndimage.label(filtered)
            if num_features > 0:
                # Find largest component
                sizes = np.bincount(labeled_array.ravel())
                largest_label = sizes[1:].argmax() + 1  # Skip background (0)
                processed = labeled_array == largest_label
                processed_clusters.append(processed)
                
        return processed_clusters

    def _create_background_mask(
        self,
        signal_mask: np.ndarray,
        existing_masks: List[np.ndarray],
        background_roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Create background mask for a signal while avoiding existing masks.
        
        Args:
            signal_mask: Binary mask for signal region
            existing_masks: List of existing masks to avoid
            background_roi: (x1, x2, y1, y2) coordinates of background ROI
            
        Returns:
            Binary mask for background region
        """
        bg_config = self.config['generate_masks']
        
        # Calculate target size
        signal_size = np.sum(signal_mask)
        target_size = int(signal_size * bg_config['bg_mask_mult'])
        
        # Create initial buffer zone
        dilated = ndimage.binary_dilation(
            signal_mask,
            iterations=bg_config['bg_mask_thickness']
        )
        
        # Create mask of all excluded regions
        excluded = np.zeros_like(signal_mask)
        for mask in existing_masks:
            excluded |= mask
            
        # Add background ROI to excluded regions
        x1, x2, y1, y2 = background_roi
        excluded[x1:x2, y1:y2] = True
        
        # Grow region until target size or no more growth possible
        bg_mask = np.zeros_like(signal_mask)
        available = ~(dilated | excluded)
        
        if not np.any(available):
            warnings.warn(
                f"No available space for background mask around signal of size {signal_size}",
                RuntimeWarning
            )
            return bg_mask
            
        # Start from boundary of dilated signal
        boundary = ndimage.binary_dilation(dilated) & ~dilated
        current_size = 0
        iterations = 0
        max_iterations = max(signal_mask.shape)  # Limit growth
        
        while current_size < target_size and iterations < max_iterations:
            # Grow from boundary where available
            new_pixels = boundary & available
            if not np.any(new_pixels):
                break
                
            bg_mask |= new_pixels
            current_size = np.sum(bg_mask)
            
            # Update boundary
            boundary = ndimage.binary_dilation(bg_mask) & ~bg_mask
            iterations += 1
            
        if current_size < target_size * 0.5:  # Less than half target size
            warnings.warn(
                f"Background mask only reached {current_size}/{target_size} pixels",
                RuntimeWarning
            )
            
        return bg_mask

    def _generate_background_masks(
        self,
        signal_masks: List[np.ndarray],
        background_roi: Tuple[int, int, int, int]
    ) -> List[np.ndarray]:
        """Create background masks for all signals.
        
        Args:
            signal_masks: List of binary masks for signal regions
            background_roi: (x1, x2, y1, y2) coordinates of background ROI
            
        Returns:
            List of binary masks for background regions
        """
        background_masks = []
        all_masks = []  # Keep track of all masks to avoid
        
        for signal_mask in signal_masks:
            bg_mask = self._create_background_mask(
                signal_mask,
                all_masks + [signal_mask],  # Avoid all existing masks
                background_roi
            )
            background_masks.append(bg_mask)
            all_masks.extend([signal_mask, bg_mask])
            
        return background_masks

    def run(self, input_data: BuildPumpProbeMasksInput) -> BuildPumpProbeMasksOutput:
        """Run mask generation for multiple peaks.
        
        Args:
            input_data: Input data containing p-values and histograms
            
        Returns:
            BuildPumpProbeMasksOutput containing pairs of signal and background masks
            
        Raises:
            ValueError: If inputs are invalid or no significant clusters found
        """
        # Get configuration parameters
        masks_config = self.config['generate_masks']
        background_roi = self.config['setup']['background_roi_coords']
        
        # Find initial clusters
        initial_clusters = self._identify_signal_clusters(
            input_data.p_values_output.p_values,
            masks_config['threshold'],
            background_roi
        )
        
        if not initial_clusters:
            raise ValueError("No significant clusters found")
            
        # Process clusters
        processed_clusters = self._process_clusters(
            initial_clusters,
            input_data.histogram_output.histograms
        )
        
        if not processed_clusters:
            raise ValueError("No clusters remained after processing")
            
        # Generate background masks
        background_masks = self._generate_background_masks(
            processed_clusters,
            background_roi
        )
        
        # Create mask pairs
        mask_pairs = []
        for signal_mask, bg_mask in zip(processed_clusters, background_masks):
            size = np.sum(signal_mask)
            pair = SignalBackgroundMaskPair(
                signal_mask=signal_mask,
                background_mask=bg_mask,
                size=size
            )
            mask_pairs.append(pair)
            
        # Store intermediate stages for diagnostics
        intermediate_masks = SignalMaskStages(
            initial=initial_clusters,
            filtered=processed_clusters,
            final=processed_clusters
        )
        
        return BuildPumpProbeMasksOutput(
            mask_pairs=mask_pairs,
            intermediate_masks=intermediate_masks
        )

    def plot_diagnostics(
        self,
        output: BuildPumpProbeMasksOutput,
        save_dir: Path
    ) -> None:
        """Generate diagnostic plots for multiple peaks.
        
        Args:
            output: Output from mask generation
            save_dir: Directory to save plots
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Multi-Peak Mask Generation Diagnostics')
        
        # 1. Combined signal masks (different colors)
        ax1 = axes[0, 0]
        combined_signal = np.zeros_like(output.mask_pairs[0].signal_mask, dtype=float)
        for i, pair in enumerate(output.mask_pairs, 1):
            combined_signal[pair.signal_mask] = i
        im1 = ax1.imshow(combined_signal)
        ax1.set_title('Signal Masks (colors indicate different peaks)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Combined background masks
        ax2 = axes[0, 1]
        combined_background = np.zeros_like(combined_signal)
        for i, pair in enumerate(output.mask_pairs, 1):
            combined_background[pair.background_mask] = i
        im2 = ax2.imshow(combined_background)
        ax2.set_title('Background Masks')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Mask sizes
        ax3 = axes[1, 0]
        sizes = [pair.size for pair in output.mask_pairs]
        ax3.bar(range(len(sizes)), sizes)
        ax3.set_title('Signal Mask Sizes')
        ax3.set_xlabel('Peak Index')
        ax3.set_ylabel('Size (pixels)')
        
        # 4. Separation validation
        ax4 = axes[1, 1]
        # Create distance transform visualization
        combined_mask = np.zeros_like(combined_signal)
        for pair in output.mask_pairs:
            combined_mask |= pair.signal_mask | pair.background_mask
        distance = ndimage.distance_transform_edt(~combined_mask)
        im4 = ax4.imshow(distance)
        ax4.set_title('Distance Between Masks')
        plt.colorbar(im4, ax=ax4, label='Distance (pixels)')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'multi_peak_diagnostics.png')
        plt.close()
        
        # Additional diagnostic plot: Mask progression
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Mask Generation Stages')
        
        # Initial clusters
        combined = np.zeros_like(combined_signal)
        for i, mask in enumerate(output.intermediate_masks.initial, 1):
            combined[mask] = i
        axes[0].imshow(combined)
        axes[0].set_title('Initial Clusters')
        
        # Filtered clusters
        combined = np.zeros_like(combined_signal)
        for i, mask in enumerate(output.intermediate_masks.filtered, 1):
            combined[mask] = i
        axes[1].imshow(combined)
        axes[1].set_title('Filtered Clusters')
        
        # Final masks with backgrounds
        combined = np.zeros_like(combined_signal)
        for i, pair in enumerate(output.mask_pairs, 1):
            combined[pair.signal_mask] = i
            combined[pair.background_mask] = -i  # Negative values for background
        axes[2].imshow(combined)
        axes[2].set_title('Final Masks (±colors match pairs)')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'mask_progression.png')
        plt.close()
        
        # Print statistics
        print("\nMask Generation Statistics:")
        print(f"Number of peaks found: {len(output.mask_pairs)}")
        for i, pair in enumerate(output.mask_pairs):
            signal_size = pair.size
            bg_size = np.sum(pair.background_mask)
            ratio = bg_size / signal_size if signal_size > 0 else 0
            print(f"\nPeak {i+1}:")
            print(f"  Signal size: {signal_size} pixels")
            print(f"  Background size: {bg_size} pixels")
            print(f"  Background/Signal ratio: {ratio:.2f}")
