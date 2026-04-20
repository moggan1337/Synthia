"""
Cell division simulation for Synthia.
Models cell cycle, division, and differentiation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random


class CellCyclePhase(Enum):
    """Cell cycle phases."""
    G0 = "G0"  # Quiescent
    G1 = "G1"  # First gap
    S = "S"    # DNA synthesis
    G2 = "G2"  # Second gap
    M = "M"    # Mitosis
    CYTOKINESIS = "cytokinesis"


@dataclass
class CellCycleCheckpoints:
    """
    Cell cycle checkpoints for quality control.
    """
    g1_checkpoint: bool = True   # Size and准备好了ness check
    s_checkpoint: bool = True    # DNA replication check
    g2_checkpoint: bool = True  # DNA damage check
    m_checkpoint: bool = True    # Spindle assembly check
    
    def all_passed(self) -> bool:
        """Check if all checkpoints passed."""
        return self.g1_checkpoint and self.s_checkpoint and self.g2_checkpoint and self.m_checkpoint
    
    def reset(self):
        """Reset all checkpoints."""
        self.g1_checkpoint = True
        self.s_checkpoint = True
        self.g2_checkpoint = True
        self.m_checkpoint = True


@dataclass
class ReplicationFork:
    """
    DNA replication fork.
    """
    position: float = 0.0  # Position on chromosome (0-1)
    speed: float = 50.0    # bp/s
    direction: str = '+'   # + or -
    active: bool = False
    
    def advance(self, dt: float):
        """Advance replication fork."""
        if self.active:
            self.position += self.speed * dt / 1e6  # Normalized to genome size


@dataclass
class Chromosome:
    """
    Chromosome in cell division.
    """
    length: int = 5_000_000  # Base pairs
    centromere_position: float = 0.5  # Position (0-1)
    
    # Replication
    replicated_fraction: float = 0.0
    replication_forks: List[ReplicationFork] = field(default_factory=list)
    
    # Segregation
    sister_chromatids_separated: bool = False
    condensed: bool = False
    
    @property
    def is_replicated(self) -> bool:
        return self.replicated_fraction >= 0.99
    
    @property
    def is_condensed(self) -> bool:
        return self.condensed


class CellDivision:
    """
    Cell division mechanics simulation.
    """
    
    def __init__(self, cell_id: int = 0):
        self.cell_id = cell_id
        
        # Cycle state
        self.phase = CellCyclePhase.G1
        self.phase_duration: Dict[CellCyclePhase, float] = {
            CellCyclePhase.G1: 6.0,   # hours
            CellCyclePhase.S: 8.0,
            CellCyclePhase.G2: 2.0,
            CellCyclePhase.M: 1.0,
            CellCyclePhase.CYTOKINESIS: 0.5,
        }
        
        self.phase_timer: float = 0.0
        
        # Checkpoints
        self.checkpoints = CellCycleCheckpoints()
        
        # Chromosomes
        self.chromosomes: List[Chromosome] = []
        self.ploidy: int = 1  # Number of chromosome sets
        
        # Size
        self.size_before_division: float = 1.0
        self.target_size: float = 2.0
        
        # Division asymmetry
        self.asymmetry_ratio: float = 0.5  # 0.5 = symmetric
        
        # Division plane
        self.division_plane: str = "midbody"
    
    def initialize_chromosomes(self, number: int = 1, length: int = 5_000_000):
        """Initialize chromosomes."""
        self.chromosomes = [Chromosome(length=length) for _ in range(number)]
        self.ploidy = number
    
    def start_cycle(self):
        """Start cell cycle."""
        self.phase = CellCyclePhase.G1
        self.phase_timer = 0.0
        self.checkpoints.reset()
        
        # Reset chromosomes
        for chrom in self.chromosomes:
            chrom.replicated_fraction = 0.0
            chrom.sister_chromatids_separated = False
            chrom.condensed = False
    
    def step(self, dt: float, nutrients: float = 1.0) -> bool:
        """
        Advance cell cycle.
        
        Args:
            dt: Time step in hours
            nutrients: Nutrient availability (0-1)
            
        Returns:
            True if cell completed division
        """
        self.phase_timer += dt
        
        # Nutrient-dependent delay
        nutrient_factor = max(0.3, nutrients)
        
        # Phase transitions
        if self.phase == CellCyclePhase.G1:
            self._phase_g1(dt, nutrient_factor)
        elif self.phase == CellCyclePhase.S:
            self._phase_s(dt)
        elif self.phase == CellCyclePhase.G2:
            self._phase_g2(dt)
        elif self.phase == CellCyclePhase.M:
            self._phase_m(dt)
        elif self.phase == CellCyclePhase.CYTOKINESIS:
            return self._phase_cytokinesis(dt)
        
        return False
    
    def _phase_g1(self, dt: float, nutrient_factor: float):
        """G1 phase: growth and preparation."""
        duration = self.phase_duration[CellCyclePhase.G1] / nutrient_factor
        
        if self.phase_timer >= duration:
            self._transition_to(CellCyclePhase.S)
    
    def _phase_s(self, dt: float):
        """S phase: DNA replication."""
        # Check G1 checkpoint
        if not self.checkpoints.g1_checkpoint:
            return
        
        # Replicate chromosomes
        for chrom in self.chromosomes:
            if not chrom.replication_forks:
                # Initialize replication forks
                chrom.replication_forks.append(ReplicationFork(position=0.5, active=True))
            
            # Advance forks
            for fork in chrom.replication_forks:
                fork.advance(dt * 3600)  # Convert to seconds
                chrom.replicated_fraction = min(1.0, fork.position)
        
        # Check if replication complete
        if all(chrom.is_replicated for chrom in self.chromosomes):
            self._transition_to(CellCyclePhase.G2)
    
    def _phase_g2(self, dt: float):
        """G2 phase: preparation for mitosis."""
        duration = self.phase_duration[CellCyclePhase.G2]
        
        if self.phase_timer >= duration and self.checkpoints.all_passed():
            self._transition_to(CellCyclePhase.M)
    
    def _phase_m(self, dt: float):
        """M phase: mitosis."""
        # Chromosome condensation
        for chrom in self.chromosomes:
            chrom.condensed = True
        
        # Metaphase to anaphase transition
        if self.phase_timer >= self.phase_duration[CellCyclePhase.M] * 0.5:
            for chrom in self.chromosomes:
                chrom.sister_chromatids_separated = True
        
        # Complete mitosis
        if self.phase_timer >= self.phase_duration[CellCyclePhase.M]:
            self._transition_to(CellCyclePhase.CYTOKINESIS)
    
    def _phase_cytokinesis(self, dt: float) -> bool:
        """Cytokinesis: cell division."""
        duration = self.phase_duration[CellCyclePhase.CYTOKINESIS]
        
        if self.phase_timer >= duration:
            # Division complete
            return True
        
        return False
    
    def _transition_to(self, new_phase: CellCyclePhase):
        """Transition to new phase."""
        self.phase = new_phase
        self.phase_timer = 0.0
    
    def check_dna_damage(self, damage_sites: int = 0) -> bool:
        """
        Check for DNA damage.
        
        Args:
            damage_sites: Number of damaged sites
            
        Returns:
            True if damage is acceptable
        """
        if damage_sites > 100:
            self.checkpoints.g2_checkpoint = False
            return False
        return True
    
    def trigger_apoptosis(self):
        """Trigger apoptosis."""
        self.phase = CellCyclePhase.G0
        self.checkpoints.g1_checkpoint = False
    
    def get_cycle_progress(self) -> float:
        """Get cycle progress (0-1)."""
        total_duration = sum(self.phase_duration.values())
        elapsed = self.phase_timer
        
        phase_order = [CellCyclePhase.G1, CellCyclePhase.S, 
                       CellCyclePhase.G2, CellCyclePhase.M, 
                       CellCyclePhase.CYTOKINESIS]
        
        for phase in phase_order:
            if self.phase == phase:
                return elapsed / self.phase_duration[phase]
            elapsed += self.phase_duration[phase]
        
        return 1.0 if self.phase == CellCyclePhase.CYTOKINESIS else 0.0
    
    def will_divide_symmetrically(self) -> bool:
        """Determine if division will be symmetric."""
        return abs(self.asymmetry_ratio - 0.5) < 0.1
    
    def calculate_division_sizes(self) -> Tuple[float, float]:
        """Calculate sizes of daughter cells."""
        size = self.size_before_division * self.target_size
        
        size_a = size * self.asymmetry_ratio
        size_b = size * (1 - self.asymmetry_ratio)
        
        return size_a, size_b


class DivisionSimulation:
    """
    Simulation of cell division in populations.
    """
    
    def __init__(self, name: str = "Division"):
        self.name = name
        self.dividing_cells: List[CellDivision] = []
        self.division_history: List[Dict] = []
        
        self.time: float = 0.0
        self.generation: int = 0
    
    def add_cell(self, cell: CellDivision):
        """Add cell to simulation."""
        self.dividing_cells.append(cell)
        cell.start_cycle()
    
    def step(self, dt: float, nutrients: float = 1.0):
        """
        Simulate one step.
        
        Args:
            dt: Time step in hours
            nutrients: Nutrient availability
        """
        new_cells = []
        
        for cell in self.dividing_cells:
            did_divide = cell.step(dt, nutrients)
            
            if did_divide:
                # Create daughter cells
                size_a, size_b = cell.calculate_division_sizes()
                
                daughter_a = CellDivision(cell_id=cell.cell_id * 2)
                daughter_a.size_before_division = size_a
                daughter_a.start_cycle()
                
                daughter_b = CellDivision(cell_id=cell.cell_id * 2 + 1)
                daughter_b.size_before_division = size_b
                daughter_b.start_cycle()
                
                new_cells.extend([daughter_a, daughter_b])
                
                # Record division
                self.division_history.append({
                    'time': self.time,
                    'parent_id': cell.cell_id,
                    'daughter_a_id': daughter_a.cell_id,
                    'daughter_b_id': daughter_b.cell_id,
                    'asymmetry': cell.asymmetry_ratio
                })
            else:
                new_cells.append(cell)
        
        self.dividing_cells = new_cells
        self.time += dt
    
    def run(self, duration: float, dt: float = 0.1, 
           nutrients: float = 1.0, max_population: int = 10000):
        """
        Run division simulation.
        
        Args:
            duration: Duration in hours
            dt: Time step
            nutrients: Nutrient availability
            max_population: Maximum population size
        """
        steps = int(duration / dt)
        
        for _ in range(steps):
            self.step(dt, nutrients)
            
            # Check population limit
            if len(self.dividing_cells) > max_population:
                # Randomly remove excess
                excess = len(self.dividing_cells) - max_population
                for _ in range(excess):
                    self.dividing_cells.pop(random.randint(0, len(self.dividing_cells) - 1))
    
    def get_population_stats(self) -> Dict:
        """Get population statistics."""
        phase_counts = {}
        for cell in self.dividing_cells:
            phase = cell.phase.value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        return {
            'total_cells': len(self.dividing_cells),
            'phase_distribution': phase_counts,
            'total_divisions': len(self.division_history),
            'time': self.time
        }


class DifferentiationModel:
    """
    Cell differentiation during division.
    """
    
    def __init__(self):
        self.stem_cells: List[CellDivision] = []
        self.differentiated_cells: List[CellDivision] = []
        
        self.stem_cell_markers: List[str] = []
        self.differentiation_factors: Dict[str, float] = {}
    
    def set_stem_markers(self, markers: List[str]):
        """Set stem cell marker genes."""
        self.stem_cell_markers = markers
    
    def set_differentiation_factor(self, factor: str, concentration: float):
        """Set differentiation factor concentration."""
        self.differentiation_factors[factor] = concentration
    
    def calculate_differentiation_probability(self, cell: CellDivision) -> float:
        """
        Calculate probability of differentiation.
        """
        # Stem cell maintenance factors
        wnt = self.differentiation_factors.get('Wnt', 0.0)
        notch = self.differentiation_factors.get('Notch', 0.0)
        
        # Base probability
        prob = 0.01
        
        # Wnt promotes stemness
        if wnt > 0.5:
            prob *= 0.5
        
        # Notch promotes differentiation
        if notch > 0.5:
            prob *= 2.0
        
        # Age factor
        if cell.phase_timer > 10:
            prob *= 1.5
        
        return min(0.5, prob)
    
    def differentiate(self, cell: CellDivision) -> bool:
        """
        Attempt to differentiate cell.
        
        Returns:
            True if cell differentiated
        """
        prob = self.calculate_differentiation_probability(cell)
        
        if random.random() < prob:
            # Differentiate
            return True
        
        return False


class AsymmetricDivision:
    """
    Model asymmetric cell division.
    """
    
    def __init__(self):
        self.cortical_determinants: Dict[str, float] = {}
        self.spatial_concentration: Dict[str, List[float]] = {}
    
    def set_determinant(self, name: str, concentration: float, 
                      polarity: str = "apical"):
        """
        Set cortical determinant.
        
        Args:
            name: Determinant name
            concentration: Concentration
            polarity: "apical" or "basal"
        """
        self.cortical_determinants[name] = concentration
        self.spatial_concentration[name] = [concentration, 0.0] if polarity == "apical" else [0.0, concentration]
    
    def calculate_inheritance(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate determinant inheritance by daughters.
        
        Returns:
            (daughter_a_inheritance, daughter_b_inheritance)
        """
        inherit_a = {}
        inherit_b = {}
        
        for name, spatial in self.spatial_concentration.items():
            # Daughter A gets apical determinants
            inherit_a[name] = spatial[0]
            # Daughter B gets basal determinants
            inherit_b[name] = spatial[1]
        
        return inherit_a, inherit_b
    
    def orient_spindle(self, cue_direction: float) -> float:
        """
        Orient mitotic spindle based on external cue.
        
        Args:
            cue_direction: Direction of external cue (radians)
            
        Returns:
            Spindle angle
        """
        # Align with cue
        return cue_direction


class MeiosisSimulation:
    """
    Meiosis simulation for germ cell division.
    """
    
    def __init__(self):
        self.phase = "prophase_I"
        self.crossing_over_sites: int = 0
        self.homolog_pairs_separated: bool = False
    
    def simulate_crossing_over(self, chromosome_a: Chromosome, 
                              chromosome_b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Simulate crossing over between homologous chromosomes.
        
        Returns:
            Recombinant chromosomes
        """
        # Simplified crossing over
        # In real cells, this involves break and repair
        
        return chromosome_a, chromosome_b
    
    def separate_homologs(self):
        """Separate homologous chromosomes in meiosis I."""
        self.homolog_pairs_separated = True
    
    def separate_sister_chromatids(self):
        """Separate sister chromatids in meiosis II."""
        pass


class CellPolarity:
    """
    Cell polarity establishment during division.
    """
    
    def __init__(self):
        self.polarity_axis: str = "apical-basal"
        self.polarity_markers: Dict[str, List[float]] = {}
    
    def establish_polarity(self, external_signal: float):
        """
        Establish polarity based on external signal.
        
        Args:
            external_signal: Signal strength (0-1)
        """
        # Par complex recruitment
        if external_signal > 0.5:
            self.polarity_markers['PAR3'] = [external_signal, 0.0]
            self.polarity_markers['PAR6'] = [external_signal, 0.0]
            self.polarity_markers['aPKC'] = [external_signal, 0.0]
        else:
            self.polarity_markers['LGL'] = [0.0, 1.0 - external_signal]
    
    def get_polarity_profile(self) -> Dict[str, List[float]]:
        """Get polarity marker distribution."""
        return self.polarity_markers
