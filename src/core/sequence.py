"""
Biological Sequence Classes for Synthia.
Handles DNA, RNA, and Protein sequences with computational biology operations.
"""

import random
import hashlib
from typing import List, Optional, Dict, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import numpy as np


class Nucleotide(Enum):
    """Enumeration of nucleotide bases."""
    ADENINE = 'A'
    THYMINE = 'T'
    GUANINE = 'G'
    CYTOSINE = 'C'
    URACIL = 'U'
    GAP = '-'
    ANY = 'N'
    
    @classmethod
    def DNA Bases(cls) -> List['Nucleotide']:
        """Return DNA nucleotide bases."""
        return [cls.ADENINE, cls.THYMINE, cls.GUANINE, cls.CYTOSINE]
    
    @classmethod
    def RNA Bases(cls) -> List['Nucleotide']:
        """Return RNA nucleotide bases."""
        return [cls.ADENINE, cls.URACIL, cls.GUANINE, cls.CYTOSINE]
    
    def complement(self) -> 'Nucleotide':
        """Return the complementary nucleotide."""
        complements = {
            'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
            'U': 'A'
        }
        return Nucleotide(complements.get(self.value, self.value))


class AminoAcid(Enum):
    """Enumeration of standard amino acids."""
    ALANINE = ('A', 'Ala', 'GCU', 'GCC', 'GCA', 'GCG')
    ARGININE = ('R', 'Arg', 'CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG')
    ASPARAGINE = ('N', 'Asn', 'AAU', 'AAC')
    ASPARTIC_ACID = ('D', 'Asp', 'GAU', 'GAC')
    CYSTEINE = ('C', 'Cys', 'UGU', 'UGC')
    GLUTAMINE = ('Q', 'Gln', 'CAA', 'CAG')
    GLUTAMIC_ACID = ('E', 'Glu', 'GAA', 'GAG')
    GLYCINE = ('G', 'Gly', 'GGU', 'GGC', 'GGA', 'GGG')
    HISTIDINE = ('H', 'His', 'CAU', 'CAC')
    ISOLEUCINE = ('I', 'Ile', 'AUU', 'AUC', 'AUA')
    LEUCINE = ('L', 'Leu', 'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG')
    LYSINE = ('K', 'Lys', 'AAA', 'AAG')
    METHIONINE = ('M', 'Met', 'AUG')
    PHENYLALANINE = ('F', 'Phe', 'UUU', 'UUC')
    PROLINE = ('P', 'Pro', 'CCU', 'CCC', 'CCA', 'CCG')
    SERINE = ('S', 'Ser', 'UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC')
    THREONINE = ('T', 'Thr', 'ACU', 'ACC', 'ACA', 'ACG')
    TRYPTOPHAN = ('W', 'Trp', 'UGG')
    TYROSINE = ('Y', 'Tyr', 'UAU', 'UAC')
    VALINE = ('V', 'Val', 'GUU', 'GUC', 'GUA', 'GUG')
    STOP = ('*', 'Stop', 'UAA', 'UAG', 'UGA')
    
    def __init__(self, one_letter: str, three_letter: str, codons: Tuple[str, ...]):
        self.one_letter = one_letter
        self.three_letter = three_letter
        self.codons = codons
    
    @property
    def molecular_weight(self) -> float:
        """Return molecular weight in Daltons."""
        weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1,
            'C': 121.2, 'E': 147.1, 'Q': 146.1, 'G': 75.1,
            'H': 155.2, 'I': 131.2, 'L': 131.2, 'K': 146.2,
            'M': 149.2, 'F': 165.2, 'P': 115.1, 'S': 105.1,
            'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
            '*': 0.0
        }
        return weights.get(self.one_letter, 0.0)
    
    @property
    def hydrophobicity(self) -> float:
        """Return hydrophobicity index (Kyte-Doolittle scale)."""
        hydro = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
            'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4,
            'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
            'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
            'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
            '*': 0.0
        }
        return hydro.get(self.one_letter, 0.0)


class Sequence:
    """Base class for biological sequences."""
    
    def __init__(self, sequence: Union[str, List]):
        if isinstance(sequence, str):
            self.sequence = list(sequence.upper())
        else:
            self.sequence = sequence
    
    def __len__(self) -> int:
        return len(self.sequence)
    
    def __str__(self) -> str:
        return ''.join(self.sequence)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self)})"
    
    def __getitem__(self, key: Union[int, slice]) -> Union[str, 'Sequence']:
        if isinstance(key, slice):
            return self.__class__(self.sequence[key])
        return self.sequence[key]
    
    def __eq__(self, other: 'Sequence') -> bool:
        return str(self) == str(other)
    
    def to_fasta(self, name: str = "sequence") -> str:
        """Convert to FASTA format."""
        lines = [f">{name}"]
        seq_str = str(self)
        for i in range(0, len(seq_str), 80):
            lines.append(seq_str[i:i+80])
        return '\n'.join(lines)
    
    @property
    def hash(self) -> str:
        """Return SHA-256 hash of sequence."""
        return hashlib.sha256(str(self).encode()).hexdigest()[:16]


class DNA(Sequence):
    """
    DNA Sequence class with computational operations.
    Supports transcription, translation, complement, reverse complement,
    and various sequence analysis operations.
    """
    
    DNA_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    
    def __init__(self, sequence: Union[str, List], circular: bool = False):
        super().__init__(sequence)
        self.circular = circular
        self._validate_dna()
    
    def _validate_dna(self):
        """Validate that sequence contains only DNA nucleotides."""
        valid = set('ATGCN')
        for base in self.sequence:
            if base.upper() not in valid:
                raise ValueError(f"Invalid DNA nucleotide: {base}")
    
    def complement(self) -> 'DNA':
        """Return the complementary DNA strand."""
        comp_seq = [self.DNA_COMPLEMENT.get(b, b) for b in self.sequence]
        return DNA(comp_seq, self.circular)
    
    def reverse_complement(self) -> 'DNA':
        """Return the reverse complement of the DNA sequence."""
        return self.complement()[::-1]
    
    def transcribe(self) -> 'RNA':
        """Transcribe DNA to RNA (replace T with U)."""
        rna_seq = ['U' if b == 'T' else b for b in self.sequence]
        return RNA(rna_seq)
    
    def reverse_transcribe(self) -> 'DNA':
        """Simulate reverse transcription to create cDNA."""
        return self.transcribe().reverse_transcribe()
    
    def translate(self, frame: int = 0) -> 'Protein':
        """Translate DNA to protein sequence."""
        rna = self.transcribe()
        return rna.translate(frame)
    
    def gc_content(self) -> float:
        """Calculate GC content percentage."""
        if len(self) == 0:
            return 0.0
        gc = sum(1 for b in self.sequence if b in 'GC')
        return gc / len(self) * 100
    
    def find_motif(self, motif: str) -> List[int]:
        """Find all occurrences of a motif."""
        motif = motif.upper()
        seq_str = str(self).upper()
        positions = []
        start = 0
        while True:
            pos = seq_str.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions
    
    def find_orfs(self, min_length: int = 100) -> List[Dict]:
        """Find Open Reading Frames (ORFs)."""
        orfs = []
        seq_str = str(self)
        
        for frame in range(3):
            start_codon = 'ATG'
            for i in range(frame, len(seq_str) - 2, 3):
                codon = seq_str[i:i+3]
                if codon == start_codon:
                    # Find stop codon
                    for j in range(i + 3, len(seq_str) - 2, 3):
                        end_codon = seq_str[j:j+3]
                        if end_codon in ['TAA', 'TAG', 'TGA']:
                            orf_len = j + 3 - i
                            if orf_len >= min_length:
                                orfs.append({
                                    'start': i,
                                    'end': j + 3,
                                    'frame': frame,
                                    'length': orf_len,
                                    'sequence': seq_str[i:j+3]
                                })
                            break
        return sorted(orfs, key=lambda x: x['length'], reverse=True)
    
    def restriction_sites(self, enzymes: Dict[str, str] = None) -> Dict[str, List[int]]:
        """Find restriction enzyme cut sites."""
        if enzymes is None:
            enzymes = {
                'EcoRI': 'GAATTC',
                'BamHI': 'GGATCC',
                'HindIII': 'AAGCTT',
                'XhoI': 'CTCGAG',
                'SalI': 'GTCGAC',
            }
        sites = {}
        for enzyme, site in enzymes.items():
            sites[enzyme] = self.find_motif(site)
        return sites
    
    def mutate(self, rate: float = 0.001, seed: int = None) -> 'DNA':
        """Introduce random mutations at given rate."""
        if seed:
            random.seed(seed)
        
        new_seq = []
        for base in self.sequence:
            if random.random() < rate:
                # Mutation: change to different base
                original = base
                while base == original:
                    base = random.choice(['A', 'T', 'G', 'C'])
            new_seq.append(base)
        return DNA(new_seq, self.circular)
    
    def pcr_primers(self, start: int, end: int, tm_range: Tuple[float, float] = (55, 65)) -> Dict:
        """Design PCR primers for a region."""
        region = self[start:end]
        rev_region = region.reverse_complement()
        
        return {
            'forward_primer': str(region),
            'reverse_primer': str(rev_region),
            'product_size': end - start,
            'tm_estimate': self._estimate_tm(str(region)),
        }
    
    def _estimate_tm(self, seq: str) -> float:
        """Estimate melting temperature using simple formula."""
        gc = seq.count('G') + seq.count('C')
        return 4 * gc + 2 * (len(seq) - gc)  # Simple Wallace rule
    
    def restriction_map(self) -> Dict[int, str]:
        """Generate restriction map with cut positions."""
        enzymes = {
            'EcoRI': 'GAATTC',
            'BamHI': 'GGATCC',
            'HindIII': 'AAGCTT',
            'NotI': 'GCGGCCGC',
            'XhoI': 'CTCGAG',
        }
        cut_sites = {}
        for enzyme, site in enzymes.items():
            positions = self.find_motif(site)
            for pos in positions:
                cut_sites[pos] = enzyme
        return dict(sorted(cut_sites.items()))


class RNA(Sequence):
    """
    RNA Sequence class with computational operations.
    Supports translation to protein, reverse transcription,
    and RNA-specific operations.
    """
    
    CODON_TABLE = DNA.CODON_TABLE
    
    def __init__(self, sequence: Union[str, List]):
        super().__init__(sequence)
        self._validate_rna()
    
    def _validate_rna(self):
        """Validate that sequence contains only RNA nucleotides."""
        valid = set('AUGCN')
        for base in self.sequence:
            if base.upper() not in valid:
                raise ValueError(f"Invalid RNA nucleotide: {base}")
    
    def reverse_transcribe(self) -> DNA:
        """Reverse transcribe RNA to DNA."""
        dna_seq = ['T' if b == 'U' else b for b in self.sequence]
        return DNA(dna_seq)
    
    def translate(self, frame: int = 0) -> 'Protein':
        """Translate RNA to protein sequence."""
        seq_str = ''.join(self.sequence[frame:])
        proteins = []
        i = frame
        while i <= len(seq_str) - 3:
            codon = seq_str[i:i+3]
            aa = self.CODON_TABLE.get(codon, 'X')
            proteins.append(aa)
            if aa == '*':
                break
            i += 3
        return Protein(proteins)
    
    def gc_content(self) -> float:
        """Calculate GC content percentage."""
        if len(self) == 0:
            return 0.0
        gc = sum(1 for b in self.sequence if b in 'GC')
        return gc / len(self) * 100
    
    def secondary_structure(self, temperature: float = 37.0) -> Dict:
        """Predict simple RNA secondary structure features."""
        # Simplified secondary structure prediction
        seq_str = str(self)
        gc_pairs = seq_str.count('G') + seq_str.count('C')
        au_pairs = seq_str.count('A') + seq_str.count('U')
        
        # Simple energy estimate
        energy = -(gc_pairs * 3) - (au_pairs * 2)
        
        return {
            'estimated_delta_g': energy,
            'gc_pairs': gc_pairs,
            'au_pairs': au_pairs,
            'mfold_hint': 'Use ViennaRNA for full prediction'
        }
    
    def find_mirna_binding_site(self, seed_type: str = 'typical') -> List[int]:
        """Find potential miRNA binding sites (simplified)."""
        seed_types = {
            'typical': ['GUUCG', 'GUUCC', 'GUUUC'],
            'centered': ['CUUCC', 'UUC'],
            'offset6': ['GCAUCC', 'UCAUCC'],
        }
        seeds = seed_types.get(seed_type, seed_types['typical'])
        sites = []
        seq_str = str(self).upper()
        for seed in seeds:
            sites.extend(self.find_motif(seed))
        return sorted(list(set(sites)))


class Protein(Sequence):
    """
    Protein Sequence class with structural and functional analysis.
    Supports sequence analysis, property calculation, and motif finding.
    """
    
    AMINO_ACID_WEIGHTS = {
        'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1,
        'C': 121.2, 'E': 147.1, 'Q': 146.1, 'G': 75.1,
        'H': 155.2, 'I': 131.2, 'L': 131.2, 'K': 146.2,
        'M': 149.2, 'F': 165.2, 'P': 115.1, 'S': 105.1,
        'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
    }
    
    HYDROPHOBICITY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
        'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4,
        'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
        'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
        'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    }
    
    def __init__(self, sequence: Union[str, List]):
        if isinstance(sequence, str):
            # Parse string into amino acids
            self.sequence = list(sequence.upper())
        else:
            self.sequence = sequence
    
    @property
    def molecular_weight(self) -> float:
        """Calculate molecular weight."""
        weight = 0.0
        for aa in self.sequence:
            weight += self.AMINO_ACID_WEIGHTS.get(aa, 110.0)
        # Subtract water for peptide bonds
        if len(self) > 1:
            weight -= (len(self) - 1) * 18.0
        return weight
    
    @property
    def isoelectric_point(self) -> float:
        """Estimate isoelectric point (pI)."""
        # Simplified pI calculation
        pos = sum(1 for aa in self.sequence if aa in 'RK')
        neg = sum(1 for aa in self.sequence if aa in 'DE')
        if pos > neg:
            return 8.0 + (pos - neg) * 0.5
        else:
            return 5.5 - (neg - pos) * 0.3
    
    def hydrophobicity_profile(self, window_size: int = 9) -> List[float]:
        """Calculate hydrophobicity profile using sliding window."""
        profile = []
        seq = self.sequence
        for i in range(len(seq) - window_size + 1):
            window = seq[i:i + window_size]
            avg = sum(self.HYDROPHOBICITY.get(aa, 0) for aa in window) / window_size
            profile.append(avg)
        return profile
    
    def transmembrane_domains(self, threshold: float = 2.0, window: int = 19) -> List[Dict]:
        """Predict transmembrane helices."""
        profile = self.hydrophobicity_profile(window)
        domains = []
        in_domain = False
        start = 0
        
        for i, value in enumerate(profile):
            if value > threshold and not in_domain:
                in_domain = True
                start = i
            elif value <= threshold and in_domain:
                in_domain = False
                if i - start >= 15:  # Minimum TM helix length
                    domains.append({
                        'start': start,
                        'end': i,
                        'length': i - start,
                        'avg_hydrophobicity': sum(profile[start:i]) / (i - start)
                    })
        
        if in_domain and len(profile) - start >= 15:
            domains.append({
                'start': start,
                'end': len(profile),
                'length': len(profile) - start,
                'avg_hydrophobicity': sum(profile[start:]) / (len(profile) - start)
            })
        
        return domains
    
    def signal_peptide_prediction(self) -> Dict:
        """Predict signal peptide (simplified)."""
        # Check for N-terminal hydrophobic region
        n_term = ''.join(self.sequence[:30])
        hydrophobic_count = sum(1 for aa in n_term if self.HYDROPHOBICITY.get(aa, 0) > 1.5)
        
        return {
            'has_signal_peptide': hydrophobic_count > 10,
            'n_term_hydrophobicity': sum(self.HYDROPHOBICITY.get(aa, 0) for aa in n_term) / len(n_term) if n_term else 0,
            'note': 'Use SignalP for accurate prediction'
        }
    
    def secondary_structure_prediction(self) -> str:
        """Predict secondary structure (simplified Chou-Fasman)."""
        # Simplified secondary structure prediction
        result = []
        helix_formers = set('AELM')
        sheet_formers = set('VIY')
        
        for aa in self.sequence:
            if aa in helix_formers:
                result.append('H')  # Helix
            elif aa in sheet_formers:
                result.append('E')  # Sheet
            else:
                result.append('C')  # Coil
        return ''.join(result)
    
    def coil_domains(self) -> List[Dict]:
        """Identify coiled-coil regions."""
        # Simplified coiled-coil prediction
        seq = ''.join(self.sequence)
        leucine_count = seq[:100].count('L') if len(seq) >= 100 else seq.count('L')
        
        return [{
            'position': 0,
            'length': len(seq),
            'leucine_zipper_score': leucine_count / max(len(seq), 1) * 100
        }]
    
    def find_motif(self, motif: str) -> List[int]:
        """Find motif occurrences."""
        seq_str = str(self)
        motif = motif.upper()
        positions = []
        start = 0
        while True:
            pos = seq_str.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions
    
    def domain_analysis(self) -> Dict:
        """Analyze protein domains."""
        domains = {
            'low_complexity': self._find_low_complexity(),
            'proline_rich': self._find_proline_rich(),
            'charged_regions': self._find_charged_regions(),
        }
        return domains
    
    def _find_low_complexity(self) -> List[Dict]:
        """Find low complexity regions."""
        regions = []
        i = 0
        while i < len(self.sequence):
            aa = self.sequence[i]
            count = 1
            for j in range(i + 1, len(self.sequence)):
                if self.sequence[j] == aa:
                    count += 1
                else:
                    break
            if count >= 10:
                regions.append({'start': i, 'end': i + count, 'aa': aa, 'length': count})
            i += max(count, 1)
        return regions
    
    def _find_proline_rich(self) -> List[Dict]:
        """Find proline-rich regions."""
        regions = []
        i = 0
        while i < len(self.sequence):
            if self.sequence[i] == 'P':
                count = 1
                for j in range(i + 1, len(self.sequence)):
                    if self.sequence[j] == 'P':
                        count += 1
                    else:
                        break
                if count >= 5:
                    regions.append({'start': i, 'end': i + count, 'length': count})
                i += count
            else:
                i += 1
        return regions
    
    def _find_charged_regions(self, threshold: float = 0.3) -> List[Dict]:
        """Find highly charged regions."""
        regions = []
        seq = self.sequence
        window = 20
        
        for i in range(len(seq) - window):
            pos = sum(1 for aa in seq[i:i+window] if aa in 'RK')
            neg = sum(1 for aa in seq[i:i+window] if aa in 'DE')
            charge = (pos + neg) / window
            
            if charge > threshold:
                regions.append({
                    'start': i,
                    'end': i + window,
                    'net_charge': pos - neg,
                    'positive': pos,
                    'negative': neg
                })
        return regions
    
    def structural_classification(self) -> Dict:
        """Classify protein structure (simplified)."""
        seq = ''.join(self.sequence)
        
        # Calculate various properties
        helical = sum(1 for aa in seq if aa in 'AELM')
        sheet = sum(1 for aa in seq if aa in 'VIY')
        
        total = len(seq)
        helical_content = helical / total if total > 0 else 0
        sheet_content = sheet / total if total > 0 else 0
        
        # Simple classification
        if helical_content > 0.45:
            structure = "Predominantly alpha-helical"
        elif sheet_content > 0.35:
            structure = "Predominantly beta-sheet"
        elif helical_content > 0.25 and sheet_content > 0.25:
            structure = "Alpha+beta"
        else:
            structure = "Mixed/Disordered"
        
        return {
            'predicted_structure': structure,
            'helical_fraction': helical_content,
            'sheet_fraction': sheet_content,
            'note': 'Use PSIPRED or other tools for accurate prediction'
        }


def load_fasta(filename: str) -> List[Tuple[str, Sequence]]:
    """Load sequences from FASTA file."""
    sequences = []
    current_name = None
    current_seq = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    seq_str = ''.join(current_seq)
                    if 'T' in seq_str or 'A' in seq_str:
                        if 'U' in seq_str:
                            sequences.append((current_name, RNA(seq_str)))
                        else:
                            sequences.append((current_name, DNA(seq_str)))
                    else:
                        sequences.append((current_name, Protein(seq_str)))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        # Add last sequence
        if current_name:
            seq_str = ''.join(current_seq)
            if 'T' in seq_str or 'A' in seq_str:
                if 'U' in seq_str:
                    sequences.append((current_name, RNA(seq_str)))
                else:
                    sequences.append((current_name, DNA(seq_str)))
            else:
                sequences.append((current_name, Protein(seq_str)))
    
    return sequences


def save_fasta(sequences: List[Tuple[str, Sequence]], filename: str):
    """Save sequences to FASTA file."""
    with open(filename, 'w') as f:
        for name, seq in sequences:
            f.write(seq.to_fasta(name) + '\n')
