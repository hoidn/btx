from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

from btx.processing.btx_types import CalculatePValuesInput, CalculatePValuesOutput

class CalculatePValues:
    """Calculate p-values for pump-probe analysis"""
    
    def __init__(self):
        pass
        
    @profile
    def run(self, input_data: CalculatePValuesInput) -> CalculatePValuesOutput:
        """Calculate p-values from EMD measurements and null distributions"""
        # Calculate p-values for each delay
        p_values = np.zeros(len(input_data.emd_values))
        for i, (emd, null_dist) in enumerate(zip(input_data.emd_values, 
                                               input_data.null_distributions)):
            p_values[i] = (null_dist > emd).mean()
            
        return CalculatePValuesOutput(
            p_values=p_values
        )
