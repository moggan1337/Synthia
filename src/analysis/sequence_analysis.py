"""
Sequence analysis tools for Synthia.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class SequenceAnalyzer:
    """
    Comprehensive sequence analysis tools.
    """
    
    def __init__(self):
        self.results: Dict = {}
    
    @staticmethod
    def calculate_complexity(sequence: str, window: int = 50) -> float:
        """
        Calculate sequence complexity (simplified LZW).
        """
        if len(sequence) < 2:
            return 0.0
        
        dictionary = {}
        complexity = 0
        current = ""
        
        for char in sequence:
            combined = current + char
            if combined not in dictionary:
                dictionary[combined] = len(dictionary)
                if len(current) > 0:
                    complexity += 1
                current = char
            else:
                current = combined
        
        return complexity / len(sequence)
    
    @staticmethod
    def codon_usage(sequence: str) -> Dict[str, float]:
        """
        Calculate codon usage frequencies.
        """
        codons = {}
        total = 0
        
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            codons[codon] = codons.get(codon, 0) + 1
            total += 1
        
        if total > 0:
            for codon in codons:
                codons[codon] /= total
        
        return codons
    
    @staticmethod
    def hydrophobic_moment(sequence: str, window: int = 11) -> List[float]:
        """
        Calculate hydrophobic moment profile.
        """
        hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
            'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4,
            'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
            'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
            'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
        }
        
        moments = []
        seq_upper = sequence.upper()
        
        for i in range(len(seq_upper) - window + 1):
            window_seq = seq_upper[i:i+window]
            
            # Calculate mean hydrophobicity
            h_mean = np.mean([hydrophobicity.get(aa, 0) for aa in window_seq])
            
            # Calculate vector magnitude
            angle = 100 * np.pi / 180  # Helix angle
            h_vector = sum(hydrophobicity.get(aa, 0) * np.cos(angle * j) 
                         for j, aa in enumerate(window_seq))
            
            moment = np.sqrt(h_vector**2 + 
                           sum(hydrophobicity.get(aa, 0) * np.sin(angle * j)
                               for j, aa in enumerate(window_seq))**2)
            
            moments.append(moment)
        
        return moments
    
    @staticmethod
    def entropy_profile(sequence: str, window: int = 50) -> List[float]:
        """
        Calculate sequence entropy profile.
        """
        from collections import Counter
        
        entropies = []
        seq_upper = sequence.upper()
        
        for i in range(len(seq_upper) - window + 1):
            window_seq = seq_upper[i:i+window]
            counts = Counter(window_seq)
            
            # Shannon entropy
            entropy = 0.0
            for count in counts.values():
                p = count / window
                if p > 0:
                    entropy -= p * np.log2(p)
            
            entropies.append(entropy)
        
        return entropies
    
    @staticmethod
    def find_repeats(sequence: str, min_length: int = 5) -> Dict[str, List[int]]:
        """
        Find repetitive sequences.
        """
        repeats = {}
        
        for length in range(min_length, min(20, len(sequence) // 2)):
            for i in range(len(sequence) - length + 1):
                repeat = sequence[i:i+length]
                
                # Check if repeated
                count = sequence.count(repeat)
                if count > 1:
                    if repeat not in repeats:
                        repeats[repeat] = []
                    repeats[repeat].append(i)
        
        return repeats
    
    @staticmethod
    def calculate_dinucleotide_bias(sequence: str) -> Dict[str, float]:
        """
        Calculate dinucleotide relative abundance.
        """
        observed = {}
        total = 0
        
        for i in range(len(sequence) - 1):
            di = sequence[i:i+2]
            observed[di] = observed.get(di, 0) + 1
            total += 1
        
        # Calculate observed frequencies
        for di in observed:
            observed[di] /= total
        
        # Expected frequencies (based on mononucleotide frequencies)
        mono = {}
        for base in sequence:
            mono[base] = mono.get(base, 0) + 1
        for base in mono:
            mono[base] /= len(sequence)
        
        # Calculate expected dinucleotide frequencies
        expected = {}
        for b1 in mono:
            for b2 in mono:
                expected[b1 + b2] = mono[b1] * mono[b2]
        
        # Relative abundance
        bias = {}
        for di in observed:
            if di in expected and expected[di] > 0:
                bias[di] = observed[di] / expected[di]
            else:
                bias[di] = 1.0
        
        return bias
    
    @staticmethod
    def local_gc_content(sequence: str, window: int = 100) -> List[float]:
        """
        Calculate local GC content.
        """
        gc_profile = []
        seq_upper = sequence.upper()
        
        for i in range(len(seq_upper) - window + 1):
            window_seq = seq_upper[i:i+window]
            gc = sum(1 for b in window_seq if b in 'GC')
            gc_profile.append(gc / window)
        
        return gc_profile
    
    @staticmethod
    def find_compositional_domains(sequence: str, threshold: float = 0.1) -> List[Dict]:
        """
        Find compositionally distinct domains.
        """
        from collections import Counter
        
        domains = []
        window = 100
        step = 50
        
        i = 0
        while i < len(sequence) - window:
            window_seq = sequence[i:i+window]
            counts = Counter(window_seq)
            
            # Get most common bases
            most_common = counts.most_common(2)
            if len(most_common) == 2:
                base1, count1 = most_common[0]
                base2, count2 = most_common[1]
                
                domain = {
                    'start': i,
                    'end': i + window,
                    'primary_base': base1,
                    'primary_freq': count1 / window,
                    'secondary_base': base2,
                    'secondary_freq': count2 / window
                }
                
                domains.append(domain)
            
            i += step
        
        return domains
    
    @staticmethod
    def calculate_substitution_rate(seq1: str, seq2: str) -> float:
        """
        Calculate simple substitution rate between sequences.
        """
        if len(seq1) != len(seq2):
            return 1.0
        
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)
