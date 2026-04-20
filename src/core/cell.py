"""
Cell and Organism classes for Synthia.
Comprehensive cell simulation with division, metabolism, and signaling.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
import time

from .genome import Genome, Gene
from .biochemistry import (
    Molecule, Reaction, Compartment, CompartmentType, BiochemicalNetwork
)
from .sequence import DNA, Protein


class CellType(Enum):
    """Types of cells."""
    PROKARYOTE = "prokaryote"
    EUKARYOTE = "eukaryote"
    BACTERIUM = "bacterium"
    YEAST = "yeast"
    ANIMAL = "animal"
    PLANT = "plant"
    ARCHAEA = "archaea"


class CellState(Enum):
    """Cell lifecycle states."""
    QUIESCENT = "quiescent"
    GROWING = "growing"
    DIVIDING = "dividing"
    DYING = "dying"
    DEAD = "dead"


class MembraneTransport:
    """
    Membrane transport mechanisms.
    """
    
    def __init__(self):
        self.channels: Dict[str, float] = {}  # {channel_name: conductivity}
        self.transporters: Dict[str, Dict] = {}  # {transporter_name: parameters}
        self.membrane_potential: float = -70.0  # mV
    
    def add_channel(self, name: str, conductivity: float):
        """Add an ion channel."""
        self.channels[name] = conductivity
    
    def add_transporter(self, name: str, kinetics: Dict):
        """Add a transporter with kinetics."""
        self.transporters[name] = kinetics
    
    def calculate_flux(self, molecule: str, concentration_in: float,
                      concentration_out: float, temperature: float = 298.15) -> float:
        """
        Calculate membrane flux using Fick's law.
        
        Args:
            molecule: Name of molecule
            concentration_in: Internal concentration (M)
            concentration_out: External concentration (M)
            temperature: Temperature (K)
            
        Returns:
            Flux (mol/(m²·s))
        """
        D = 1e-9  # Diffusion coefficient (m²/s)
        thickness = 5e-9  # Membrane thickness (5 nm)
        
        # Fick's law
        flux = -D * (concentration_in - concentration_out) / thickness
        
        # Adjust for membrane potential if charged
        # (simplified Goldman equation)
        
        return flux
    
    def update_membrane_potential(self, ion_concentrations: Dict[str, Tuple[float, float]]):
        """
        Update membrane potential based on ion gradients.
        Uses Goldman equation (simplified).
        
        Args:
            ion_concentrations: {ion: (internal_mM, external_mM)}
        """
        R = 8.314  # Gas constant
        F = 96485  # Faraday constant
        T = 298.15  # Temperature
        
        # Permeability coefficients (relative)
        P = {'K': 1.0, 'Na': 0.04, 'Cl': 0.45}
        
        # Simplified Goldman equation
        inner_K, outer_K = ion_concentrations.get('K', (140, 5))
        inner_Na, outer_Na = ion_concentrations.get('Na', (14, 140))
        inner_Cl, outer_Cl = ion_concentrations.get('Cl', (4, 110))
        
        # Vm = (RT/F) * ln((PK*[K+]out + PNa*[Na+]out + PCl*[Cl-]in) /
        #                   (PK*[K+]in + PNa*[Na+]in + PCl*[Cl-]out))
        
        inside = P['K'] * outer_K + P['Na'] * outer_Na + P['Cl'] * inner_Cl
        outside = P['K'] * inner_K + P['Na'] * inner_Na + P['Cl'] * outer_Cl
        
        if outside > 0:
            self.membrane_potential = (R * T / F) * np.log(inside / outside) * 1000


@dataclass
class CellVolume:
    """
    Cell volume and growth dynamics.
    """
    initial_volume: float = 1e-12  # Liters (1 picoliter)
    current_volume: float = 1e-12
    target_volume: float = 2e-12  # Volume at division
    growth_rate: float = 0.0  # per hour
    doubling_time: float = 1.0  # hours
    
    def __post_init__(self):
        self.initial_volume = self.current_volume
    
    def grow(self, delta_time: float):
        """
        Grow cell volume.
        
        Args:
            delta_time: Time step in seconds
        """
        if self.growth_rate > 0:
            # Exponential growth
            dt_hours = delta_time / 3600
            self.current_volume = self.initial_volume * np.exp(self.growth_rate * dt_hours)
    
    def will_divide(self) -> bool:
        """Check if cell should divide based on volume."""
        return self.current_volume >= self.target_volume
    
    def reset_after_division(self):
        """Reset volume after cell division."""
        self.current_volume = self.initial_volume
    
    def calculate_growth_rate(self, nutrients: Dict[str, float]) -> float:
        """
        Calculate growth rate based on nutrient availability.
        Uses Monod kinetics.
        """
        # Monod equation: μ = μmax * S / (Ks + S)
        mu_max = 0.5  # Maximum growth rate (per hour)
        Ks = 0.1  # Half-saturation constant
        
        if not nutrients:
            return 0.0
        
        # Use limiting nutrient
        mu = 0.0
        for nutrient, concentration in nutrients.items():
            local_mu = mu_max * concentration / (Ks + concentration)
            mu = max(mu, local_mu)
        
        self.growth_rate = mu
        return mu


@dataclass
class ProteinPool:
    """
    Pool of proteins in the cell.
    """
    proteins: Dict[str, float] = field(default_factory=dict)  # {protein_name: concentration}
    mrna_levels: Dict[str, float] = field(default_factory=dict)  # {gene_name: mRNA count}
    
    def add_protein(self, name: str, concentration: float):
        """Add protein to pool."""
        self.proteins[name] = concentration
    
    def get_protein(self, name: str) -> float:
        """Get protein concentration."""
        return self.proteins.get(name, 0.0)
    
    def remove_protein(self, name: str, amount: float):
        """Remove protein from pool."""
        if name in self.proteins:
            self.proteins[name] = max(0.0, self.proteins[name] - amount)
    
    def get_total_protein(self) -> float:
        """Get total protein concentration."""
        return sum(self.proteins.values())
    
    def update_mrna(self, gene_name: str, transcription_rate: float, 
                   degradation_rate: float = 0.1):
        """Update mRNA level (simple model)."""
        current = self.mrna_levels.get(gene_name, 0.0)
        self.mrna_levels[gene_name] = current + transcription_rate - degradation_rate * current
        self.mrna_levels[gene_name] = max(0.0, self.mrna_levels[gene_name])
    
    def synthesize_protein(self, gene_name: str, translation_rate: float,
                          degradation_rate: float = 0.01):
        """Synthesize protein from mRNA."""
        mrna = self.mrna_levels.get(gene_name, 0.0)
        current = self.proteins.get(gene_name, 0.0)
        
        # Translation
        synthesis = translation_rate * mrna
        # Degradation
        loss = degradation_rate * current
        
        self.proteins[gene_name] = current + synthesis - loss
        self.proteins[gene_name] = max(0.0, self.proteins[gene_name])


@dataclass
class MetabolicState:
    """
    Metabolic state of the cell.
    """
    atp: float = 10.0  # ATP concentration (mM)
    nadh: float = 1.0  # NADH concentration (mM)
    nad: float = 10.0  # NAD+ concentration (mM)
    nadph: float = 1.0  # NADPH concentration (mM)
    nadp: float = 0.1  # NADP+ concentration (mM)
    
    # Energy charge (0-1)
    @property
    def energy_charge(self) -> float:
        """Calculate cellular energy charge."""
        atp_contrib = self.atp
        adp_contrib = self.atp / 2  # Approximate ADP
        amp_contrib = self.atp / 4  # Approximate AMP
        
        total = atp_contrib + adp_contrib + amp_contrib
        if total == 0:
            return 0.5
        
        return atp_contrib / total
    
    @property
    def nadh_nad_ratio(self) -> float:
        """Calculate NADH/NAD+ ratio."""
        if self.nad == 0:
            return float('inf')
        return self.nadh / self.nad
    
    @property
    def nadph_nadp_ratio(self) -> float:
        """Calculate NADPH/NADP+ ratio (reducing power)."""
        if self.nadp == 0:
            return float('inf')
        return self.nadph / self.nadp
    
    def update_atp(self, delta: float):
        """Update ATP level."""
        self.atp = max(0.0, self.atp + delta)
    
    def consume_atp(self, amount: float) -> bool:
        """Consume ATP, return True if successful."""
        if self.atp >= amount:
            self.atp -= amount
            return True
        return False


class Cell:
    """
    Comprehensive cell simulation class.
    Models cell physiology, division, metabolism, and responses.
    """
    
    def __init__(self, name: str = "Cell", cell_type: CellType = CellType.BACTERIUM):
        self.name = name
        self.cell_type = cell_type
        
        # Identity
        self.id = random.randint(10000, 99999)
        self.lineage: List[str] = []
        
        # State
        self.state = CellState.QUIESCENT
        self.age: float = 0.0  # Hours
        self.cycle_position: float = 0.0  # 0-1 through cell cycle
        
        # Components
        self.genome: Optional[Genome] = None
        self.volume = CellVolume()
        self.membrane = MembraneTransport()
        self.proteins = ProteinPool()
        self.metabolism = MetabolicState()
        self.biochemical_network: Optional[BiochemicalNetwork] = None
        
        # Compartment structure
        self.compartments: Dict[CompartmentType, Compartment] = {}
        self._initialize_compartments()
        
        # Signaling
        self.signal_cache: Dict[str, float] = {}
        
        # Environmental
        self.environment: Dict[str, float] = {}
        
        # History
        self.history: List[Dict] = []
    
    def _initialize_compartments(self):
        """Initialize cell compartments."""
        # Cytoplasm
        self.compartments[CompartmentType.CYTOPLASM] = Compartment(
            name="cytoplasm",
            compartment_type=CompartmentType.CYTOPLASM,
            volume=0.8 * self.volume.current_volume,
            pH=7.2
        )
        
        # Cell membrane
        self.compartments[CompartmentType.MEMBRANE] = Compartment(
            name="membrane",
            compartment_type=CompartmentType.MEMBRANE,
            volume=0.01 * self.volume.current_volume,
            pH=7.4
        )
        
        if self.cell_type == CellType.EUKARYOTE:
            self.compartments[CompartmentType.NUCLEUS] = Compartment(
                name="nucleus",
                compartment_type=CompartmentType.NUCLEUS,
                volume=0.1 * self.volume.current_volume,
                pH=7.2
            )
    
    def set_genome(self, genome: Genome):
        """Set the cell's genome."""
        self.genome = genome
        self._initialize_gene_products()
    
    def _initialize_gene_products(self):
        """Initialize proteins from genome genes."""
        if not self.genome:
            return
        
        for gene in self.genome.genes:
            if gene.protein_sequence:
                # Initialize at low concentration
                self.proteins.add_protein(gene.name, 0.0)
    
    def set_environment(self, nutrients: Dict[str, float]):
        """Set external nutrient environment."""
        self.environment.update(nutrients)
    
    def step(self, delta_time: float):
        """
        Execute one simulation step.
        
        Args:
            delta_time: Time step in seconds
        """
        dt_hours = delta_time / 3600
        self.age += dt_hours
        
        # Update cell state
        if self.state == CellState.QUIESCENT:
            self._check_growth_conditions()
        elif self.state == CellState.GROWING:
            self._grow(delta_time)
        elif self.state == CellState.DIVIDING:
            self._complete_division()
        
        # Update biochemical network
        self._update_metabolism(delta_time)
        
        # Update gene expression
        self._update_gene_expression(delta_time)
        
        # Update signaling
        self._update_signaling(delta_time)
        
        # Record history
        if len(self.history) % 100 == 0:  # Sample every 100 steps
            self._record_state()
    
    def _check_growth_conditions(self):
        """Check if conditions are suitable for growth."""
        if self.environment.get('nutrients', 0) > 0.1:
            self.state = CellState.GROWING
            mu = self.volume.calculate_growth_rate(self.environment)
            self.volume.doubling_time = np.log(2) / mu if mu > 0 else 1.0
    
    def _grow(self, delta_time: float):
        """Grow the cell."""
        # Update volume
        self.volume.grow(delta_time)
        
        # Update cycle position
        if self.volume.doubling_time > 0:
            self.cycle_position = (self.age % self.volume.doubling_time) / self.volume.doubling_time
        
        # Check for division trigger
        if self.volume.will_divide():
            self.state = CellState.DIVIDING
    
    def _complete_division(self):
        """Complete cell division."""
        self._replicate_chromosome()
        self.volume.reset_after_division()
        self.state = CellState.GROWING
        self.cycle_position = 0.0
    
    def _replicate_chromosome(self):
        """Replicate the chromosome (simplified)."""
        if self.genome and self.genome.sequence:
            # Create a copy
            new_sequence = DNA(list(self.genome.sequence.sequence), circular=True)
            # In full implementation, this would trigger septum formation
    
    def _update_metabolism(self, delta_time: float):
        """Update metabolic state."""
        dt = delta_time  # in seconds
        
        # ATP maintenance
        atp_consumption = 0.001 * self.volume.current_volume * 1e12  # mM/s
        if self.metabolism.consume_atp(atp_consumption * dt):
            # ATP available
            pass
        else:
            # Energy stress
            self._respond_to_energy_stress()
        
        # Recharge NADH -> NAD+
        nadh_oxidation = 0.01 * dt
        self.metabolism.nadh = max(0.0, self.metabolism.nadh - nadh_oxidation)
        self.metabolism.nad = min(10.0, self.metabolism.nad + nadh_oxidation)
    
    def _respond_to_energy_stress(self):
        """Respond to ATP depletion."""
        # Reduce growth
        self.volume.growth_rate *= 0.5
        
        # Activate stress responses
        stress_genes = ['sos_response', 'heat_shock', 'stringent_response']
        for gene_name in stress_genes:
            current = self.proteins.get_protein(gene_name)
            self.proteins.add_protein(gene_name, current + 0.1)
    
    def _update_gene_expression(self, delta_time: float):
        """Update gene expression levels."""
        if not self.genome:
            return
        
        dt = delta_time / 3600  # Convert to hours
        
        for gene in self.genome.genes:
            if not gene.promoter:
                continue
            
            # Get transcription factors
            tf_levels = {}
            for tf_name in gene.transcription_factors:
                tf_levels[tf_name] = self.proteins.get_protein(tf_name)
            
            # Calculate promoter activity
            transcription = gene.promoter.calculate_activity(tf_levels)
            
            # Update mRNA
            transcription_rate = transcription * gene.transcriptional_regulation
            self.proteins.update_mrna(gene.name, transcription_rate)
            
            # Translate to protein
            translation_rate = gene.rbs.calculate_translation_initiation_rate() if gene.rbs else 0.5
            degradation = gene.protein_stability * 0.1
            
            self.proteins.synthesize_protein(gene.name, translation_rate, degradation)
    
    def _update_signaling(self, delta_time: float):
        """Update cell signaling pathways."""
        # Check for signaling molecules
        signal_keys = [k for k in self.environment.keys() if k.startswith('signal_')]
        
        for signal in signal_keys:
            signal_name = signal.replace('signal_', '')
            concentration = self.environment[signal]
            
            # Signal transduction (simplified)
            self.signal_cache[signal_name] = concentration
    
    def _record_state(self):
        """Record cell state to history."""
        record = {
            'time': self.age,
            'volume': self.volume.current_volume,
            'state': self.state.value,
            'atp': self.metabolism.atp,
            'total_protein': self.proteins.get_total_protein(),
            'expressed_genes': sum(1 for g in self.genome.genes if g.is_expressed()) if self.genome else 0
        }
        self.history.append(record)
    
    def receive_signal(self, signal_name: str, strength: float):
        """Receive an external signal."""
        self.signal_cache[signal_name] = strength
    
    def respond_to_signal(self, signal_name: str) -> float:
        """Get cell response to a signal."""
        return self.signal_cache.get(signal_name, 0.0)
    
    def add_molecule(self, compartment: CompartmentType, molecule: Molecule, concentration: float):
        """Add molecule to compartment."""
        if compartment in self.compartments:
            self.compartments[compartment].add_molecule(molecule, concentration)
    
    def get_concentration(self, compartment: CompartmentType, molecule: Molecule) -> float:
        """Get molecule concentration in compartment."""
        if compartment in self.compartments:
            return self.compartments[compartment].get_concentration(molecule)
        return 0.0
    
    def transport_molecule(self, molecule: Molecule, from_compartment: CompartmentType,
                          to_compartment: CompartmentType):
        """Transport molecule between compartments."""
        if from_compartment in self.compartments and to_compartment in self.compartments:
            conc = self.compartments[from_compartment].get_concentration(molecule)
            if conc > 0:
                self.compartments[from_compartment].remove_molecule(molecule)
                self.compartments[to_compartment].add_molecule(molecule, conc)
    
    def get_cell_stats(self) -> Dict:
        """Get comprehensive cell statistics."""
        stats = {
            'id': self.id,
            'name': self.name,
            'type': self.cell_type.value,
            'state': self.state.value,
            'age_hours': self.age,
            'volume_pL': self.volume.current_volume * 1e12,
            'growth_rate': self.volume.growth_rate,
            'doubling_time_h': self.volume.doubling_time,
            'cycle_position': self.cycle_position,
            'atp_mM': self.metabolism.atp,
            'energy_charge': self.metabolism.energy_charge,
            'nadh_nad_ratio': self.metabolism.nadh_nad_ratio,
            'total_protein_mM': self.proteins.get_total_protein(),
        }
        
        if self.genome:
            stats['gene_count'] = len(self.genome.genes)
            stats['expressed_genes'] = sum(1 for g in self.genome.genes if g.is_expressed())
        
        return stats
    
    def is_alive(self) -> bool:
        """Check if cell is alive."""
        return self.state not in [CellState.DEAD, CellState.DYING] and self.metabolism.atp > 0.1
    
    def trigger_apoptosis(self):
        """Trigger programmed cell death."""
        self.state = CellState.DYING
        # Initiate death cascade
        death_genes = ['caspase_3', 'caspase_7', 'apaf1']
        for gene in death_genes:
            self.proteins.add_protein(gene, 1.0)
    
    def lyse(self):
        """Trigger cell lysis."""
        self.state = CellState.DEAD
        self.volume.current_volume = 0
    
    def __repr__(self):
        return f"Cell({self.name}, {self.cell_type.value}, state={self.state.value})"


class Organism:
    """
    Multi-cellular organism simulation.
    """
    
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species
        
        self.cells: List[Cell] = []
        self.tissues: Dict[str, List[Cell]] = {}
        
        self.age: float = 0.0
        self.development_stage: str = "adult"
    
    def add_cell(self, cell: Cell, tissue: str = "default"):
        """Add cell to organism."""
        self.cells.append(cell)
        
        if tissue not in self.tissues:
            self.tissues[tissue] = []
        self.tissues[tissue].append(cell)
    
    def get_cell_count(self) -> int:
        """Get total cell count."""
        return len(self.cells)
    
    def get_tissue_cell_count(self, tissue: str) -> int:
        """Get cell count for a tissue."""
        return len(self.tissues.get(tissue, []))
    
    def step(self, delta_time: float):
        """Simulate organism for one time step."""
        self.age += delta_time / 3600
        
        # Step all cells
        for cell in self.cells:
            cell.step(delta_time)
        
        # Remove dead cells
        self.cells = [c for c in self.cells if c.is_alive()]
    
    def get_organism_stats(self) -> Dict:
        """Get organism statistics."""
        alive_cells = [c for c in self.cells if c.is_alive()]
        
        return {
            'name': self.name,
            'species': self.species,
            'age_hours': self.age,
            'total_cells': len(alive_cells),
            'tissues': {t: len(cells) for t, cells in self.tissues.items()},
            'development_stage': self.development_stage
        }


class Population:
    """
    Cell population simulation with dynamics.
    """
    
    def __init__(self, name: str, initial_count: int = 100):
        self.name = name
        self.cells: List[Cell] = []
        
        # Initialize population
        for _ in range(initial_count):
            cell = Cell(name=f"Cell_{len(self.cells)}")
            self.cells.append(cell)
        
        # Population statistics
        self.history: List[Dict] = []
        self.generation: int = 0
    
    def step(self, delta_time: float):
        """Simulate population for one time step."""
        # Step all cells
        for cell in self.cells:
            cell.step(delta_time)
        
        # Remove dead cells
        dead_cells = [c for c in self.cells if not c.is_alive()]
        for cell in dead_cells:
            self.cells.remove(cell)
        
        # Check for division and add daughter cells
        new_cells = []
        for cell in self.cells[:]:  # Copy list to allow modification
            if cell.state == CellState.DIVIDING:
                # Create daughter cell
                daughter = Cell(name=f"{cell.name}_d{len(new_cells)}")
                daughter.genome = cell.genome  # Share genome reference
                daughter.set_environment(cell.environment)
                new_cells.append(daughter)
        
        self.cells.extend(new_cells)
        
        # Record statistics
        self._record_population_stats()
    
    def _record_population_stats(self):
        """Record population statistics."""
        if self.cells:
            avg_volume = np.mean([c.volume.current_volume for c in self.cells])
            avg_atp = np.mean([c.metabolism.atp for c in self.cells])
            alive = sum(1 for c in self.cells if c.is_alive())
        else:
            avg_volume = 0
            avg_atp = 0
            alive = 0
        
        self.history.append({
            'generation': self.generation,
            'population': alive,
            'avg_volume': avg_volume,
            'avg_atp': avg_atp
        })
    
    def get_population_growth_rate(self) -> float:
        """Calculate population growth rate."""
        if len(self.history) < 2:
            return 0.0
        
        # Use last 10 points
        recent = self.history[-10:]
        if len(recent) < 2:
            return 0.0
        
        times = [h['generation'] for h in recent]
        populations = [h['population'] for h in recent]
        
        # Log transform for exponential growth
        log_pop = [np.log(max(1, p)) for p in populations]
        
        # Linear regression
        if len(times) > 1:
            mean_t = np.mean(times)
            mean_lp = np.mean(log_pop)
            
            numerator = sum((t - mean_t) * (lp - mean_lp) for t, lp in zip(times, log_pop))
            denominator = sum((t - mean_t) ** 2 for t in times)
            
            if denominator > 0:
                return numerator / denominator
        
        return 0.0
    
    def get_population_stats(self) -> Dict:
        """Get population statistics."""
        if not self.cells:
            return {
                'population': 0,
                'generations': self.generation,
                'growth_rate': 0.0
            }
        
        return {
            'population': len(self.cells),
            'generations': self.generation,
            'growth_rate': self.get_population_growth_rate(),
            'avg_cell_age': np.mean([c.age for c in self.cells]),
            'avg_cell_volume': np.mean([c.volume.current_volume for c in self.cells])
        }
