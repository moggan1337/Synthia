"""
Simulation analysis tools for Synthia.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats, signal


class SimulationAnalyzer:
    """
    Analyze simulation results.
    """
    
    def __init__(self):
        self.time_series: Dict[str, List[float]] = {}
    
    def load_time_series(self, name: str, values: List[float]):
        """Load time series data."""
        self.time_series[name] = values
    
    @staticmethod
    def calculate_oscillation_parameters(values: List[float], 
                                        dt: float = 1.0) -> Dict:
        """
        Calculate oscillation parameters.
        """
        values = np.array(values)
        
        # Find peaks
        peaks, _ = signal.find_peaks(values)
        troughs, _ = signal.find_peaks(-values)
        
        if len(peaks) < 2:
            return {'oscillatory': False}
        
        # Calculate period
        peak_intervals = np.diff(peaks) * dt
        period = np.mean(peak_intervals)
        
        # Calculate amplitude
        amplitudes = values[peaks] - values[troughs[:len(peaks)]]
        amplitude = np.mean(amplitudes) if len(amplitudes) > 0 else 0
        
        # Calculate frequency
        frequency = 1 / period if period > 0 else 0
        
        # Check regularity
        regularity = np.std(peak_intervals) / np.mean(peak_intervals) if len(peak_intervals) > 1 else 0
        
        return {
            'oscillatory': True,
            'period': period,
            'frequency': frequency,
            'amplitude': amplitude,
            'regularity': regularity,
            'num_peaks': len(peaks)
        }
    
    @staticmethod
    def calculate_steady_state(values: List[float], window: int = 100) -> Tuple[float, float]:
        """
        Calculate steady state value and variance.
        """
        if len(values) < window:
            return np.mean(values), np.std(values)
        
        return np.mean(values[-window:]), np.std(values[-window:])
    
    @staticmethod
    def detect_phase_transition(values: List[float], window: int = 50) -> List[int]:
        """
        Detect phase transitions in time series.
        """
        values = np.array(values)
        transitions = []
        
        for i in range(window, len(values) - window):
            before = np.mean(values[i-window:i])
            after = np.mean(values[i:i+window])
            
            # Significant change
            if abs(after - before) > 2 * np.std(values):
                transitions.append(i)
        
        return transitions
    
    @staticmethod
    def calculate_responsetime(values: List[float], 
                            threshold_fraction: float = 0.5) -> float:
        """
        Calculate response time (time to reach threshold).
        """
        values = np.array(values)
        max_value = np.max(values)
        threshold = max_value * threshold_fraction
        
        for i, v in enumerate(values):
            if v >= threshold:
                return i
        
        return len(values)
    
    @staticmethod
    def calculate_adaptation_level(values: List[float]) -> Dict:
        """
        Calculate perfect adaptation metrics.
        """
        values = np.array(values)
        
        # Initial level
        initial = np.mean(values[:10]) if len(values) > 10 else values[0]
        
        # Final level
        final, final_std = SimulationAnalyzer.calculate_steady_state(values.tolist())
        
        # Peak level
        peak = np.max(values)
        
        # Adaptation error
        adaptation_error = abs(final - initial)
        
        # Relative adaptation
        peak_amplitude = peak - initial
        relative_adaptation = adaptation_error / peak_amplitude if peak_amplitude > 0 else 0
        
        return {
            'initial_level': initial,
            'final_level': final,
            'peak_level': peak,
            'adaptation_error': adaptation_error,
            'relative_adaptation': relative_adaptation
        }
    
    @staticmethod
    def compare_distributions(data1: List[float], data2: List[float]) -> Dict:
        """
        Compare two distributions statistically.
        """
        data1 = np.array(data1)
        data2 = np.array(data2)
        
        # t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
        
        # K-S test
        ks_stat, ks_p = stats.ks_2samp(data1, data2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'mean_diff': np.mean(data1) - np.mean(data2)
        }
    
    @staticmethod
    def fit_exponential(values: List[float]) -> Tuple[float, float]:
        """
        Fit exponential decay/growth.
        
        Returns:
            (amplitude, rate_constant)
        """
        values = np.array(values)
        t = np.arange(len(values))
        
        # Log transform for exponential
        log_values = np.log(values + 1e-10)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, log_values)
        
        return (np.exp(intercept), slope)
    
    @staticmethod
    def autocorr(values: List[float], max_lag: int = None) -> np.ndarray:
        """
        Calculate autocorrelation function.
        """
        values = np.array(values)
        n = len(values)
        
        if max_lag is None:
            max_lag = n // 2
        
        mean = np.mean(values)
        var = np.var(values)
        
        autocorr = np.correlate(values - mean, values - mean, mode='full')
        autocorr = autocorr[n - 1:n - 1 + max_lag] / (var * n)
        
        return autocorr
    
    @staticmethod
    def calculate_noise_metrics(values: List[float]) -> Dict:
        """
        Calculate noise characteristics.
        """
        values = np.array(values)
        
        # Mean and variance
        mean = np.mean(values)
        var = np.var(values)
        
        # Fano factor (for count data)
        if mean > 0:
            fano_factor = var / mean
        else:
            fano_factor = 0
        
        # Coefficient of variation
        cv = np.sqrt(var) / mean if mean > 0 else 0
        
        # Signal-to-noise ratio
        snr = mean / np.sqrt(var) if var > 0 else float('inf')
        
        return {
            'mean': mean,
            'variance': var,
            'fano_factor': fano_factor,
            'coefficient_of_variation': cv,
            'signal_to_noise': snr
        }
    
    @staticmethod
    def sensitivity_analysis(data: Dict[str, List[float]], 
                           target: str) -> Dict[str, float]:
        """
        Calculate sensitivity of target to other variables.
        """
        sensitivities = {}
        
        if target not in data:
            return sensitivities
        
        target_values = np.array(data[target])
        target_mean = np.mean(target_values)
        target_std = np.std(target_values)
        
        for var_name, var_values in data.items():
            if var_name == target:
                continue
            
            var_values = np.array(var_values)
            
            # Correlation
            correlation = np.corrcoef(target_values, var_values)[0, 1]
            
            # Sensitivity (normalized)
            if np.std(var_values) > 0:
                sensitivity = correlation * (target_std / np.std(var_values))
            else:
                sensitivity = 0
            
            sensitivities[var_name] = sensitivity
        
        return sensitivities
