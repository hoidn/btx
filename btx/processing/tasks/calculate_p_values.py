from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calculate_pvalues.log'),
        logging.NullHandler()  # Prevents logging to console
    ]
)
logger = logging.getLogger('CalculatePValues')

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

from btx.processing.btx_types import CalculatePValuesInput, CalculatePValuesOutput

class CalculatePValues:
    """Calculate p-values from EMD values and null distribution."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize p-value calculation task.
        
        Args:
            config: Dictionary containing:
                - calculate_pvalues.significance_threshold: P-value threshold (default: 0.05)
                - setup.background_roi_coords: [x1, x2, y1, y2]
        """
        self.config = config
        
        if 'calculate_pvalues' not in self.config:
            self.config['calculate_pvalues'] = {}
        if 'significance_threshold' not in self.config['calculate_pvalues']:
            self.config['calculate_pvalues']['significance_threshold'] = 0.05
            
        # Validate config has necessary ROI coordinates
        if 'setup' not in self.config or 'background_roi_coords' not in self.config['setup']:
            raise ValueError("Configuration must include setup.background_roi_coords")

    def _calculate_p_values(
        self,
        emd_values: np.ndarray,
        null_distribution: np.ndarray
    ) -> np.ndarray:
        """Calculate p-values for each pixel.
        
        Args:
            emd_values: 2D array of EMD values
            null_distribution: 1D array of null distribution values
            
        Returns:
            2D array of p-values
        """
        p_values = np.zeros_like(emd_values)
        min_p_value = 1.0 / (len(null_distribution) + 1)
        
        for i in range(emd_values.shape[0]):
            for j in range(emd_values.shape[1]):
                p_value = np.mean(null_distribution >= emd_values[i, j])
                if p_value == 0:
                    warnings.warn(
                        f"P-value underflow at pixel ({i},{j}). "
                        f"Setting to minimum possible value {min_p_value:.2e}",
                        RuntimeWarning
                    )
                    p_value = min_p_value
                p_values[i, j] = p_value
                
        return p_values

    def _check_background_uniformity(
        self,
        p_values: np.ndarray,
        roi_coords: List[int]
    ) -> None:
        """Check uniformity of p-values within background ROI."""
        # Extract ROI p-values
        x1, x2, y1, y2 = roi_coords
        bg_p_values = p_values[y1:y2, x1:x2].ravel()
        n_pixels = len(bg_p_values)
        
        # Basic statistics
        mean_p = np.mean(bg_p_values)
        median_p = np.median(bg_p_values)
        std_p = np.std(bg_p_values)
        
        # Kolmogorov-Smirnov test against uniform distribution
        ks_stat, ks_pval = stats.kstest(bg_p_values, 'uniform')
        
        # Anderson-Darling test - transform uniform to normal first
        transformed_data = stats.norm.ppf(bg_p_values)
        transformed_data = transformed_data[~np.isnan(transformed_data)]  # Remove any NaNs from 0/1
        ad_stat, ad_crit, ad_sig = stats.anderson(transformed_data, 'norm')
        
        # Log results
        logger.info("\n=== Background ROI P-value Uniformity Check ===")
        logger.info(f"ROI coordinates: {roi_coords}")
        logger.info(f"Number of ROI pixels: {n_pixels}")
        logger.info("\nBasic Statistics:")
        logger.info(f"  - Mean p-value: {mean_p:.3f} (expected 0.5)")
        logger.info(f"  - Median p-value: {median_p:.3f} (expected 0.5)")
        logger.info(f"  - Standard deviation: {std_p:.3f} (expected {np.sqrt(1/12):.3f})")
        logger.info("\nUniformity Tests:")
        logger.info(f"  Kolmogorov-Smirnov test:")
        logger.info(f"    - Statistic: {ks_stat:.3f}")
        logger.info(f"    - P-value: {ks_pval:.3e}")
        logger.info(f"  Anderson-Darling test:")
        logger.info(f"    - Statistic: {ad_stat:.3f} (after normal transform)")
        logger.info(f"    - Critical values: {ad_crit} (for normal distribution)")
        logger.info(f"    - Significance levels: {ad_sig}")
        logger.info(f"    - Test interpretation: values greater than critical value ")
        logger.info(f"      reject null hypothesis at corresponding significance level")
        
        # Issue warnings for significant deviations
        if ks_pval < 0.05:
            msg = (
                f"Background ROI p-values significantly deviate from uniform distribution "
                f"(KS test p={ks_pval:.2e}). This suggests the background may not be IID."
            )
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
        
        # Check for severe deviation in mean/median
        expected_std = np.sqrt(1/12)  # Standard deviation of uniform[0,1]
        mean_zscore = abs(mean_p - 0.5) / (expected_std / np.sqrt(n_pixels))
        if mean_zscore > 3:
            msg = (
                f"Background ROI mean p-value ({mean_p:.3f}) deviates significantly "
                f"from expected 0.5 (z-score = {mean_zscore:.1f})"
            )
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
        
        # Check for unusually small variance
        if std_p < expected_std * 0.5:  # Much smaller than expected
            msg = (
                f"Background ROI p-values show unusually small variance ({std_p:.3f} vs "
                f"expected {expected_std:.3f}). This may indicate over-smoothing."
            )
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)

    @profile
    def run(self, input_data: CalculatePValuesInput) -> CalculatePValuesOutput:
        """Run p-value calculation.
        
        Args:
            input_data: CalculatePValuesInput containing EMD values and null distribution
            
        Returns:
            CalculatePValuesOutput containing p-values and derived data
            
        Raises:
            ValueError: If input data is invalid
        """
        emd_values = input_data.emd_output.emd_values
        null_distribution = input_data.emd_output.null_distribution
        
        # Calculate p-values
        p_values = self._calculate_p_values(emd_values, null_distribution)
        
        # Calculate -log10(p-values) for visualization
        # Handle zeros by using minimum possible p-value
        min_p_value = 1.0 / (len(null_distribution) + 1)
        log_p_values = -np.log10(np.maximum(p_values, min_p_value))
        
        # Get significance threshold
        threshold = self.config['calculate_pvalues']['significance_threshold']
        
        # Print some statistics
        n_significant = np.sum(p_values < threshold)
        print(f"Found {n_significant} significant pixels "
              f"(p < {threshold:.3f}, {n_significant/p_values.size:.1%} of total)")
        
        # Check background uniformity
        roi_coords = self.config['setup']['background_roi_coords']
        self._check_background_uniformity(p_values, roi_coords)
        
        return CalculatePValuesOutput(
            p_values=p_values,
            log_p_values=log_p_values,
            significance_threshold=threshold
        )
        
    def plot_diagnostics(self, output: CalculatePValuesOutput, save_dir: Path) -> None:
        """Generate diagnostic plots."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. P-value spatial distribution (log scale)
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(output.log_p_values, cmap='viridis')
        ax1.set_title('-log10(P-values)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Binary significance mask
        ax2 = fig.add_subplot(222)
        signif_mask = output.p_values < output.significance_threshold
        im2 = ax2.imshow(signif_mask, cmap='RdBu')
        ax2.set_title(f'Significant Pixels (p < {output.significance_threshold:.3f})')
        plt.colorbar(im2, ax=ax2)
        
        # 3. P-value histogram
        ax3 = fig.add_subplot(223)
        # Plot full distribution
        ax3.hist(output.p_values.ravel(), bins=50, density=True, label='All pixels', alpha=1.0)

        # Add background ROI distribution
        roi_coords = self.config['setup']['background_roi_coords']
        x1, x2, y1, y2 = roi_coords
        print(f"Full image shape: {output.p_values.shape}")
        print(f"ROI coords: {roi_coords}")
        print(f"ROI shape: {output.p_values[x1:x2, y1:y2].shape}")
        bg_p_values = output.p_values[y1:y2, x1:x2].ravel()
        print(f"Number of pixels - full: {output.p_values.size}, ROI: {bg_p_values.size}")
        ax3.hist(bg_p_values, bins=50, density=True, label='Background ROI', alpha=0.5, color='orange')

        ax3.axvline(
            output.significance_threshold,
            color='r',
            linestyle='--',
            label=f'p = {output.significance_threshold:.3f}'
        )
        # Add uniform distribution reference line
        ax3.axhline(1.0, color='k', linestyle=':', label='Uniform')
        ax3.set_xlabel('P-value')
        ax3.set_ylabel('Density')
        ax3.set_title('P-value Distribution')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Q-Q plot
        ax4 = fig.add_subplot(224)
        observed_p = np.sort(output.p_values.ravel())
        expected_p = np.linspace(0, 1, len(observed_p))
        ax4.plot(expected_p, observed_p, 'b.', alpha=0.1)
        ax4.plot([0, 1], [0, 1], 'r--', label='y=x')
        ax4.set_xlabel('Expected P-value')
        ax4.set_ylabel('Observed P-value')
        ax4.set_title('P-value Q-Q Plot')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'calculate_pvalues_diagnostics.png')
        plt.close()
