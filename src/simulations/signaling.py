"""
Cell signaling pathway simulation for Synthia.
Models signal transduction, second messengers, and cellular responses.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random


class SignalingPathway(Enum):
    """Major signaling pathways."""
    MAPK = "mapk"
    PI3K_AKT = "pi3k_akt"
    JAK_STAT = "jak_stat"
    NOTCH = "notch"
    WNT = "wnt"
    HEDGEHOG = "hedgehog"
    TGF_BETA = "tgf_beta"
    NFKB = "nfkb"
    CALCIUM = "calcium"
    CAMP = "camp"


class SecondMessenger(Enum):
    """Second messenger molecules."""
    CAMP = "cAMP"
    CGMP = "cGMP"
    IP3 = "IP3"
    DAG = "DAG"
    CA2 = "Ca2+"
    NO = "NO"


@dataclass
class Receptor:
    """
    Cell surface receptor.
    """
    name: str
    ligand: str
    receptor_type: str  # RTK, GPCR, etc.
    
    # State
    active: bool = False
    bound_ligand: float = 0.0
    
    # Kinetics
    kd: float = 1.0  # Dissociation constant
    k_on: float = 1e6  # Association rate (M^-1 s^-1)
    k_off: float = 1.0  # Dissociation rate (s^-1)
    
    # Desensitization
    internalization_rate: float = 0.0
    desensitized: bool = False
    
    def bind_ligand(self, ligand_concentration: float, dt: float):
        """Simulate ligand binding."""
        # Binding
        binding = self.k_on * ligand_concentration * (1 - self.bound_ligand) * dt
        unbinding = self.k_off * self.bound_ligand * dt
        
        self.bound_ligand += binding - unbinding
        self.bound_ligand = max(0.0, min(1.0, self.bound_ligand))
        
        self.active = self.bound_ligand > 0.5
    
    def internalize(self, dt: float):
        """Receptor internalization."""
        if self.internalization_rate > 0:
            self.bound_ligand *= (1 - self.internalization_rate * dt)


@dataclass
class Kinase:
    """
    Protein kinase.
    """
    name: str
    substrate: str
    pathway: SignalingPathway
    
    # State
    active: bool = False
    phosphorylated: bool = False
    
    # Kinetics
    vmax: float = 1.0
    km: float = 0.1
    
    # Regulation
    auto_phosphorylation_rate: float = 0.0
    phosphatase_rate: float = 0.1
    
    def activate(self, kinase_activity: float = 1.0):
        """Activate kinase."""
        self.active = True
        self.phosphorylated = True
    
    def deactivate(self):
        """Deactivate kinase."""
        self.active = False
    
    def update(self, upstream_signal: float, dt: float):
        """Update kinase state."""
        # Phosphorylation by upstream kinase
        if upstream_signal > 0:
            phos = self.auto_phosphorylation_rate * upstream_signal * dt
            self.phosphorylated = True
            self.active = True
        else:
            # Dephosphorylation
            dephos = self.phosphatase_rate * dt
            self.phosphorylated = max(False, random.random() < dephos)
            self.active = self.phosphorylated


@dataclass
class SignalingComponent:
    """
    General signaling component.
    """
    name: str
    component_type: str  # kinase, phosphatase, adapter, effector
    pathway: SignalingPathway
    
    # State
    concentration: float = 1.0
    active_concentration: float = 0.0
    
    # Interactions
    activators: List[str] = field(default_factory=list)
    inhibitors: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    
    def get_activity(self) -> float:
        """Get fractional activity."""
        if self.concentration == 0:
            return 0.0
        return self.active_concentration / self.concentration
    
    def activate(self, amount: float):
        """Activate component."""
        self.active_concentration = min(self.concentration, 
                                       self.active_concentration + amount)
    
    def deactivate(self, amount: float):
        """Deactivate component."""
        self.active_concentration = max(0.0, self.active_concentration - amount)


class SignalingPathwaySimulation:
    """
    Complete signaling pathway simulation.
    """
    
    def __init__(self, name: str = "Signaling"):
        self.name = name
        
        self.pathway: Optional[SignalingPathway] = None
        
        # Components
        self.receptors: Dict[str, Receptor] = {}
        self.kinases: Dict[str, Kinase] = {}
        self.components: Dict[str, SignalingComponent] = {}
        
        # Second messengers
        self.second_messengers: Dict[SecondMessenger, float] = {}
        
        # Output effectors
        self.transcription_factors: Dict[str, float] = {}
        
        # Simulation state
        self.time: float = 0.0
        self.history: List[Dict] = []
        
        # Parameters
        self.signal_amplification: float = 10.0  # Fold amplification
        self.noise_level: float = 0.05
    
    def add_receptor(self, receptor: Receptor):
        """Add receptor."""
        self.receptors[receptor.name] = receptor
    
    def add_kinase(self, kinase: Kinase):
        """Add kinase."""
        self.kinases[kinase.name] = kinase
    
    def add_component(self, component: SignalingComponent):
        """Add signaling component."""
        self.components[component.name] = component
    
    def set_second_messenger(self, messenger: SecondMessenger, concentration: float):
        """Set second messenger level."""
        self.second_messengers[messenger] = concentration
    
    def ligand_binding(self, receptor_name: str, ligand_concentration: float, dt: float):
        """Simulate ligand binding to receptor."""
        if receptor_name in self.receptors:
            self.receptors[receptor_name].bind_ligand(ligand_concentration, dt)
    
    def simulate_step(self, dt: float, 
                     ligand_concentrations: Dict[str, float] = None):
        """
        Simulate one step of signaling.
        
        Args:
            dt: Time step (seconds)
            ligand_concentrations: External ligand concentrations
        """
        # Ligand binding to receptors
        if ligand_concentrations:
            for receptor_name, conc in ligand_concentrations.items():
                if receptor_name in self.receptors:
                    self.ligand_binding(receptor_name, conc, dt)
        
        # Signal transduction through pathway
        self._propagate_signal(dt)
        
        # Update second messengers
        self._update_second_messengers(dt)
        
        # Activate transcription factors
        self._activate_transcription_factors(dt)
        
        # Add noise
        self._add_noise(dt)
        
        self.time += dt
        
        # Record history
        if len(self.history) % 10 == 0:
            self._record_state()
    
    def _propagate_signal(self, dt: float):
        """Propagate signal through the pathway."""
        # Simplified signal cascade
        for kinase_name, kinase in self.kinases.items():
            # Find upstream activator
            upstream = self._get_upstream_kinase(kinase_name)
            if upstream:
                upstream_signal = upstream.active_concentration if upstream else 0.0
            else:
                # Check receptor activation
                upstream_signal = 0.0
                for receptor in self.receptors.values():
                    if receptor.active:
                        upstream_signal += receptor.bound_ligand
            
            kinase.update(upstream_signal * self.signal_amplification, dt)
    
    def _get_upstream_kinase(self, kinase_name: str) -> Optional[SignalingComponent]:
        """Get upstream kinase in cascade."""
        # Define cascade order
        cascades = {
            SignalingPathway.MAPK: ['Raf', 'MEK', 'ERK'],
            SignalingPathway.PI3K_AKT: ['PI3K', 'PDK1', 'Akt'],
            SignalingPathway.JAK_STAT: ['JAK', 'STAT'],
        }
        
        if self.pathway and self.pathway in cascades:
            cascade = cascades[self.pathway]
            if kinase_name in cascade:
                idx = cascade.index(kinase_name)
                if idx > 0:
                    prev_name = cascade[idx - 1]
                    return self.components.get(prev_name)
        
        return None
    
    def _update_second_messengers(self, dt: float):
        """Update second messenger levels."""
        # cAMP production
        if SecondMessenger.CAMP in self.second_messengers:
            # GPCR-mediated production
            gprotein_activity = sum(1 for r in self.receptors.values() if r.active)
            production = 0.1 * gprotein_activity
            degradation = 0.05 * self.second_messengers[SecondMessenger.CAMP]
            
            self.second_messengers[SecondMessenger.CAMP] += (production - degradation) * dt
            self.second_messengers[SecondMessenger.CAMP] = max(0.0, 
                self.second_messengers[SecondMessenger.CAMP])
        
        # Calcium dynamics
        if SecondMessenger.CA2 in self.second_messengers:
            # IP3-mediated release
            ip3 = self.second_messengers.get(SecondMessenger.IP3, 0.0)
            
            # Store release
            release = 0.5 * ip3
            # Pump back
            reuptake = 0.1 * self.second_messengers[SecondMessenger.CA2]
            
            self.second_messengers[SecondMessenger.CA2] += (release - reuptake) * dt
            self.second_messengers[SecondMessenger.CA2] = max(0.0,
                self.second_messengers[SecondMessenger.CA2])
    
    def _activate_transcription_factors(self, dt: float):
        """Activate transcription factors downstream of signaling."""
        # MAPK pathway activates ELK1, c-Fos
        if self.pathway == SignalingPathway.MAPK:
            erk = self.kinases.get('ERK')
            if erk and erk.active:
                self.transcription_factors['ELK1'] = min(1.0, 
                    self.transcription_factors.get('ELK1', 0.0) + 0.1 * dt)
                self.transcription_factors['c-Fos'] = min(1.0,
                    self.transcription_factors.get('c-Fos', 0.0) + 0.05 * dt)
        
        # PI3K-Akt activates NF-kB
        elif self.pathway == SignalingPathway.PI3K_AKT:
            akt = self.kinases.get('Akt')
            if akt and akt.active:
                self.transcription_factors['NFkB'] = min(1.0,
                    self.transcription_factors.get('NFkB', 0.0) + 0.1 * dt)
    
    def _add_noise(self, dt: float):
        """Add biological noise to signaling."""
        for tf_name in self.transcription_factors:
            noise = random.gauss(0, self.noise_level * dt)
            self.transcription_factors[tf_name] = max(0.0, min(1.0,
                self.transcription_factors.get(tf_name, 0.0) + noise))
    
    def _record_state(self):
        """Record current state."""
        state = {
            'time': self.time,
            'receptors': {name: r.active for name, r in self.receptors.items()},
            'kinases': {name: k.active for name, k in self.kinases.items()},
            'second_messengers': dict(self.second_messengers),
            'transcription_factors': dict(self.transcription_factors)
        }
        self.history.append(state)
    
    def run_simulation(self, duration: float, dt: float = 0.01,
                      ligand_pulse: Tuple[float, float, str] = None) -> List[Dict]:
        """
        Run signaling simulation.
        
        Args:
            duration: Duration in seconds
            dt: Time step
            ligand_pulse: (start, end, receptor_name) for pulse stimulus
        """
        time_steps = int(duration / dt)
        
        for _ in range(time_steps):
            # Check for ligand pulse
            ligand_concs = {}
            if ligand_pulse:
                start, end, receptor = ligand_pulse
                if start <= self.time <= end:
                    ligand_concs[receptor] = 1.0  # Saturating ligand
            
            self.simulate_step(dt, ligand_concs)
        
        return self.history
    
    def get_pathway_activity(self) -> Dict[str, float]:
        """Get overall pathway activity."""
        activity = {
            'receptor_activation': np.mean([r.active for r in self.receptors.values()]),
            'kinase_cascade': np.mean([k.active for k in self.kinases.values()]),
            'tf_activation': np.mean(list(self.transcription_factors.values())),
        }
        
        if self.second_messengers:
            activity['second_messengers'] = np.mean(
                list(self.second_messengers.values()))
        
        return activity


class SignalTransduction:
    """
    General signal transduction model.
    """
    
    def __init__(self):
        self.input_signals: Dict[str, float] = {}
        self.output_responses: Dict[str, float] = {}
        self.network: Dict[str, List[str]] = {}  # {node: downstream_nodes}
        
    def add_node(self, node: str, downstream: List[str] = None):
        """Add signaling node."""
        self.network[node] = downstream or []
    
    def set_input(self, signal: str, intensity: float):
        """Set input signal."""
        self.input_signals[signal] = intensity
    
    def propagate_signal(self, node: str, intensity: float,
                        visited: Set[str] = None) -> float:
        """
        Propagate signal through network.
        
        Args:
            node: Current node
            intensity: Signal intensity
            visited: Set of visited nodes (for cycle prevention)
            
        Returns:
            Total output response
        """
        if visited is None:
            visited = set()
        
        if node in visited:
            return 0.0
        
        visited.add(node)
        
        total_response = intensity
        
        # Propagate to downstream nodes
        for downstream in self.network.get(node, []):
            # Apply attenuation
            attenuated = intensity * 0.8
            total_response += self.propagate_signal(downstream, attenuated, visited)
        
        return total_response
    
    def calculate_response(self) -> Dict[str, float]:
        """Calculate response to all inputs."""
        responses = {}
        
        for signal, intensity in self.input_signals.items():
            response = self.propagate_signal(signal, intensity)
            responses[signal] = response
        
        return responses


class CalciumSignaling:
    """
    Specialized calcium signaling dynamics.
    """
    
    def __init__(self):
        self.resting_ca: float = 0.1  # µM
        self.ca_concentration: float = 0.1
        self.oscillation_frequency: float = 0.0  # Hz
        
        # Channels
        self.ip3_sensitive: float = 0.0
        self.voltage_gated: float = 0.0
        self.store_leak: float = 0.01
        
        # Pumps
        self.serca_rate: float = 0.5  # SERCA pump rate
        self.pmca_rate: float = 0.1  # Plasma membrane Ca ATPase
        
        # Buffers
        self.buffer_capacity: float = 50.0  # Buffer capacity
    
    def simulate(self, ip3_signal: float, dt: float = 0.01):
        """
        Simulate calcium dynamics.
        
        Args:
            ip3_signal: IP3 concentration (µM)
            dt: Time step (ms)
        """
        # IP3 receptor opening
        self.ip3_sensitive = ip3_signal / (ip3_signal + 0.5)
        
        # Store release
        store_release = self.ip3_sensitive * 0.5
        
        # Leak from store
        leak = self.store_leak * (self.resting_ca - self.ca_concentration)
        
        # SERCA pumping
        serca = self.serca_rate * self.ca_concentration
        
        # PMCA extrusion
        pmca = self.pmca_rate * self.ca_concentration
        
        # Net change
        d_ca = store_release + leak - serca - pmca
        
        # Update with buffer kinetics
        self.ca_concentration += d_ca * dt / self.buffer_capacity
        self.ca_concentration = max(0.01, self.ca_concentration)
        
        # Detect oscillations
        self._detect_oscillations()
    
    def _detect_oscillations(self):
        """Detect calcium oscillations."""
        # Simplified oscillation detection
        if self.ca_concentration > 1.0:
            self.oscillation_frequency = 0.1  # Approximate frequency
    
    def get_spark(self) -> Dict:
        """Get calcium spark parameters."""
        if self.ca_concentration > 0.5:
            return {
                'amplitude': self.ca_concentration,
                'duration': 50.0,  # ms
                'frequency': self.oscillation_frequency
            }
        return None


def design_crosstalk(pathway1: SignalingPathway, 
                    pathway2: SignalingPathway) -> List[str]:
    """
    Design crosstalk points between pathways.
    """
    # Known crosstalk points
    crosstalk_points = [
        (SignalingPathway.MAPK, SignalingPathway.PI3K_AKT, 'Raf-PI3K interaction'),
        (SignalingPathway.P13K_AKT, SignalingPathway.MAPK, 'Akt-GSK3-RAF cascade'),
        (SignalingPathway.CAMP, SignalingPathway.MAPK, 'PKA-RAF crosstalk'),
        (SignalingPathway.CA2, SignalingPathway.MAPK, 'Calmodulin-kinase interactions'),
    ]
    
    points = []
    for p1, p2, mechanism in crosstalk_points:
        if (p1 == pathway1 and p2 == pathway2) or (p1 == pathway2 and p2 == pathway1):
            points.append(mechanism)
    
    return points
