"""
Gene Regulatory Network simulation for Synthia.
Models transcriptional regulation, gene networks, and circuit dynamics.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
import networkx as nx

from ..core.genome import Gene, Promoter, PromoterType
from ..core.sequence import Protein


class InteractionType(Enum):
    """Types of regulatory interactions."""
    ACTIVATION = "activation"
    REPRESSION = "repression"
    DUAL = "dual"  # Can be both
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"


@dataclass
class RegulatoryInteraction:
    """
    Represents a regulatory interaction between transcription factors and genes.
    """
    regulator: str  # TF gene name
    target: str  # Target gene name
    interaction_type: InteractionType
    strength: float = 1.0  # Interaction strength (0-1)
    hill_coefficient: float = 1.0  # Cooperativity
    kd: float = 0.5  # Dissociation constant
    
    # Context
    binding_sites: int = 1  # Number of TF binding sites
    direct: bool = True  # Direct or indirect
    
    def calculate_effect(self, tf_concentration: float) -> float:
        """
        Calculate regulatory effect based on TF concentration.
        
        Args:
            tf_concentration: Concentration of the transcription factor
            
        Returns:
            Effect factor (0-1 for repression, >1 for activation)
        """
        if self.interaction_type == InteractionType.ACTIVATION:
            # Hill equation for activation
            effect = (tf_concentration ** self.hill_coefficient) / (
                self.kd ** self.hill_coefficient + tf_concentration ** self.hill_coefficient
            )
            return 1.0 + effect * (self.strength - 1.0)
        
        elif self.interaction_type == InteractionType.REPRESSION:
            # Hill equation for repression (inverse)
            effect = (tf_concentration ** self.hill_coefficient) / (
                self.kd ** self.hill_coefficient + tf_concentration ** self.hill_coefficient
            )
            return 1.0 - effect * self.strength
        
        elif self.interaction_type == InteractionType.DUAL:
            # Can both activate and repress depending on context
            effect = (tf_concentration ** self.hill_coefficient) / (
                self.kd ** self.hill_coefficient + tf_concentration ** self.hill_coefficient
            )
            return 1.0 + (effect - 0.5) * 2 * self.strength
        
        return 1.0


@dataclass
class RegulatoryModule:
    """
    A module of co-regulated genes.
    """
    name: str
    genes: List[str]
    master_regulator: Optional[str] = None
    function: str = ""
    
    def get_core_genes(self) -> List[str]:
        """Get core genes of the module."""
        if self.master_regulator:
            return [self.master_regulator] + self.genes[:5]
        return self.genes[:5]


class GeneRegulatoryNetwork:
    """
    Gene Regulatory Network (GRN) with dynamics and analysis.
    """
    
    def __init__(self, name: str = "GRN"):
        self.name = name
        
        # Network structure
        self.interactions: List[RegulatoryInteraction] = []
        self.genes: Set[str] = set()
        
        # Networkx graph for analysis
        self.graph = nx.DiGraph()
        
        # Expression levels
        self.expression: Dict[str, float] = {}  # {gene: expression_level}
        self.basal_expression: Dict[str, float] = {}
        
        # Time series
        self.history: List[Dict[str, float]] = []
        
        # Modules
        self.modules: List[RegulatoryModule] = []
        
        # Parameters
        self.mRNA_degradation_rate: float = 0.1  # per hour
        self.protein_degradation_rate: float = 0.01  # per hour
        self.transcription_rate: float = 1.0
        self.translation_rate: float = 1.0
        
        # Noise
        self.noise_level: float = 0.05
        
        self._build_graph()
    
    def add_interaction(self, interaction: RegulatoryInteraction):
        """Add regulatory interaction to network."""
        self.interactions.append(interaction)
        self.genes.add(interaction.regulator)
        self.genes.add(interaction.target)
        self._build_graph()
    
    def add_gene(self, gene_name: str, basal_expression: float = 0.0):
        """Add gene to network."""
        self.genes.add(gene_name)
        self.basal_expression[gene_name] = basal_expression
        self.expression[gene_name] = basal_expression
    
    def add_module(self, module: RegulatoryModule):
        """Add regulatory module."""
        self.modules.append(module)
        for gene in module.genes:
            self.genes.add(gene)
    
    def _build_graph(self):
        """Build NetworkX graph from interactions."""
        self.graph = nx.DiGraph()
        
        for gene in self.genes:
            self.graph.add_node(gene, expression=self.expression.get(gene, 0.0))
        
        for interaction in self.interactions:
            self.graph.add_edge(
                interaction.regulator,
                interaction.target,
                interaction_type=interaction.interaction_type.value,
                strength=interaction.strength
            )
    
    def get_regulators(self, gene: str) -> List[str]:
        """Get all regulators of a gene."""
        return list(self.graph.predecessors(gene))
    
    def get_targets(self, gene: str) -> List[str]:
        """Get all targets of a gene."""
        return list(self.graph.successors(gene))
    
    def get_network_statistics(self) -> Dict:
        """Calculate network statistics."""
        stats = {
            'num_genes': len(self.genes),
            'num_interactions': len(self.interactions),
            'num_modules': len(self.modules),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if len(self.genes) > 0 else 0,
            'density': nx.density(self.graph),
        }
        
        # Calculate centrality measures
        if len(self.genes) > 0:
            try:
                betweenness = nx.betweenness_centrality(self.graph)
                stats['hub_genes'] = sorted(betweenness.items(), key=lambda x: -x[1])[:5]
            except:
                stats['hub_genes'] = []
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            stats['feedback_loops'] = len(cycles)
        except:
            stats['feedback_loops'] = 0
        
        return stats
    
    def simulate_step(self, delta_time: float, 
                    external_inputs: Dict[str, float] = None):
        """
        Simulate one step of GRN dynamics.
        
        Args:
            delta_time: Time step in hours
            external_inputs: External signals affecting gene expression
        """
        dt = delta_time
        
        # New expression levels
        new_expression = {}
        
        for gene in self.genes:
            # Get regulators
            regulators = self.get_regulators(gene)
            
            # Calculate regulatory input
            regulatory_input = 1.0
            for reg_gene in regulators:
                interaction = self._get_interaction(reg_gene, gene)
                if interaction:
                    reg_expr = self.expression.get(reg_gene, 0.0)
                    effect = interaction.calculate_effect(reg_expr)
                    regulatory_input *= effect
            
            # Add external inputs
            if external_inputs and gene in external_inputs:
                regulatory_input *= (1.0 + external_inputs[gene])
            
            # Basal expression
            basal = self.basal_expression.get(gene, 0.1)
            
            # Transcription
            transcription = basal * regulatory_input * self.transcription_rate
            
            # Degradation
            current_expr = self.expression.get(gene, 0.0)
            degradation = self.mRNA_degradation_rate * current_expr
            
            # Add noise
            noise = random.gauss(0, self.noise_level)
            
            # Update
            new_expr = current_expr + (transcription - degradation) * dt
            new_expr += noise * dt
            new_expression[gene] = max(0.0, new_expr)
        
        self.expression.update(new_expression)
        
        # Record history
        if len(self.history) % 10 == 0:
            self.history.append(self.expression.copy())
    
    def _get_interaction(self, regulator: str, target: str) -> Optional[RegulatoryInteraction]:
        """Get interaction between regulator and target."""
        for interaction in self.interactions:
            if interaction.regulator == regulator and interaction.target == target:
                return interaction
        return None
    
    def run_simulation(self, duration: float, delta_time: float = 0.01,
                      external_inputs: Dict[str, float] = None) -> List[Dict[str, float]]:
        """
        Run full simulation.
        
        Args:
            duration: Simulation duration in hours
            delta_time: Time step in hours
            external_inputs: External inputs (can be time-varying dict)
            
        Returns:
            Time series of expression levels
        """
        time = 0.0
        while time < duration:
            # Check for time-varying inputs
            if external_inputs and callable(external_inputs):
                inputs = external_inputs(time)
            else:
                inputs = external_inputs
            
            self.simulate_step(delta_time, inputs)
            time += delta_time
        
        return self.history
    
    def find_attractors(self, steps: int = 100) -> List[List[str]]:
        """
        Find attractors (stable states) in the network.
        Uses exhaustive simulation to find steady states.
        """
        # Simplified attractor detection
        states = []
        
        for _ in range(steps):
            self.simulate_step(0.1)
            state = tuple(sorted((g, round(e, 2)) for g, e in self.expression.items()))
            states.append(state)
        
        # Find repeated states (attractors)
        from collections import Counter
        state_counts = Counter(states)
        
        attractors = []
        for state, count in state_counts.items():
            if count >= 3:  # Threshold for attractor
                genes = [g for g, _ in state]
                attractors.append(genes)
        
        return attractors
    
    def identify_network_motifs(self) -> Dict[str, int]:
        """
        Identify network motifs ( Feed-Forward Loops, etc.)
        """
        motifs = {
            'ffl': 0,  # Feed-forward loop
            'bffl': 0,  # Bifan
            'sim': 0,  # Single input module
            'dense': 0,  # Dense regulon
        }
        
        # Find feed-forward loops (A->B->C, A->C)
        for node_a in self.genes:
            targets = self.get_targets(node_a)
            for node_b in targets:
                for node_c in self.get_targets(node_b):
                    if node_a in self.get_targets(node_c) and node_c in targets:
                        motifs['ffl'] += 1
        
        # Find bifans (A,C both regulate B,D)
        for node_a in self.genes:
            for node_c in self.get_targets(node_a):
                targets_a = set(self.get_targets(node_c))
                targets_c = set(self.get_targets(node_a))
                common_targets = targets_a & targets_c
                if len(common_targets) >= 2:
                    motifs['bffl'] += 1
        
        return motifs
    
    def calculate_dynamics_stability(self) -> Dict:
        """Analyze the stability of network dynamics."""
        # Calculate Jacobian matrix (simplified)
        n = len(self.genes)
        gene_list = list(self.genes)
        
        if n == 0:
            return {'stable': True, 'eigenvalues': []}
        
        jacobian = np.zeros((n, n))
        
        for i, gene_i in enumerate(gene_list):
            for j, gene_j in enumerate(gene_list):
                interaction = self._get_interaction(gene_j, gene_i)
                if interaction:
                    # Partial derivative approximation
                    base_effect = interaction.calculate_effect(self.expression.get(gene_j, 0.0))
                    delta = 0.01
                    perturbed_effect = interaction.calculate_effect(
                        self.expression.get(gene_j, 0.0) + delta
                    )
                    jacobian[i, j] = (perturbed_effect - base_effect) / delta
        
        # Add degradation terms
        for i in range(n):
            jacobian[i, i] -= self.mRNA_degradation_rate
        
        # Calculate eigenvalues
        try:
            eigenvalues = np.linalg.eigvals(jacobian)
            max_real = max(e.real for e in eigenvalues)
            
            return {
                'stable': max_real < 0,
                'max_real_eigenvalue': max_real,
                'eigenvalues': eigenvalues.tolist()
            }
        except:
            return {'stable': False, 'error': 'Could not compute eigenvalues'}
    
    def design_synthetic_circuit(self, circuit_type: str) -> List[RegulatoryInteraction]:
        """
        Design a synthetic gene circuit.
        
        Args:
            circuit_type: Type of circuit ('toggle_switch', 'oscillator', 'repressor')
            
        Returns:
            List of interactions for the circuit
        """
        circuits = {
            'toggle_switch': [
                RegulatoryInteraction('repressor_a', 'repressor_b', 
                                    InteractionType.REPRESSION, strength=2.0),
                RegulatoryInteraction('repressor_b', 'repressor_a',
                                    InteractionType.REPRESSION, strength=2.0),
            ],
            'oscillator': [
                RegulatoryInteraction('activator', 'repressor_a',
                                    InteractionType.ACTIVATION, strength=1.5),
                RegulatoryInteraction('repressor_a', 'repressor_b',
                                    InteractionType.REPRESSION, strength=1.5),
                RegulatoryInteraction('repressor_b', 'activator',
                                    InteractionType.REPRESSION, strength=1.5),
            ],
            'repressor': [
                RegulatoryInteraction('repressor', 'target_gene',
                                    InteractionType.REPRESSION, strength=1.5),
            ],
        }
        
        circuit = circuits.get(circuit_type, [])
        
        # Add genes
        for interaction in circuit:
            if interaction.regulator not in self.genes:
                self.add_gene(interaction.regulator)
            if interaction.target not in self.genes:
                self.add_gene(interaction.target)
            self.add_interaction(interaction)
        
        return circuit


class RegulatorySimulation:
    """
    High-level simulation of gene regulatory networks.
    """
    
    def __init__(self, network: GeneRegulatoryNetwork = None):
        self.network = network or GeneRegulatoryNetwork()
        self.time_series: List[Dict] = []
    
    def run_with_pulse_input(self, gene: str, pulse_start: float,
                            pulse_end: float, pulse_strength: float,
                            duration: float = 10.0) -> Dict[str, List]:
        """
        Run simulation with a pulse input to a gene.
        
        Args:
            gene: Gene to apply pulse
            pulse_start: Start time of pulse (hours)
            pulse_end: End time of pulse (hours)
            pulse_strength: Multiplicative factor during pulse
            duration: Total simulation duration
            
        Returns:
            Time series data
        """
        def input_func(time):
            if pulse_start <= time <= pulse_end:
                return {gene: pulse_strength}
            return {}
        
        self.network.run_simulation(duration, external_inputs=input_func)
        
        return {
            'time': list(range(len(self.network.history))) if self.network.history else [],
            'expression': self.network.history.copy()
        }
    
    def run_knockout_simulation(self, gene: str, duration: float = 10.0) -> Dict:
        """
        Simulate gene knockout and measure effects.
        """
        # Store original expression
        original_expr = self.network.expression[gene]
        
        # Knock out gene
        self.network.expression[gene] = 0.0
        
        # Run simulation
        self.network.run_simulation(duration)
        
        # Calculate effects
        effects = {}
        for g, expr in self.network.expression.items():
            if g != gene:
                effects[g] = expr - self.network.basal_expression.get(g, 0.1)
        
        # Restore original
        self.network.expression[gene] = original_expr
        
        return effects
    
    def run_overexpression_simulation(self, gene: str, fold_change: float = 10.0,
                                     duration: float = 10.0) -> Dict:
        """
        Simulate gene overexpression.
        """
        original_expr = self.network.expression[gene]
        
        # Overexpress
        self.network.expression[gene] = original_expr * fold_change
        
        # Run simulation
        self.network.run_simulation(duration)
        
        # Calculate effects
        effects = {}
        for g, expr in self.network.expression.items():
            effects[g] = expr
        
        # Restore
        self.network.expression[gene] = original_expr
        
        return effects
    
    def get_final_state(self) -> Dict[str, float]:
        """Get final expression state."""
        return self.network.expression.copy()
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics."""
        return {
            'time_points': len(self.network.history),
            'network_stats': self.network.get_network_statistics(),
            'final_state': self.get_final_state()
        }
