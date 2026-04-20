"""
Biochemical kinetics simulation for Synthia.
Implements Michaelis-Menten kinetics, Hill equations, and enzyme kinetics.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import random


@dataclass
class Enzyme:
    """
    Enzyme with kinetic parameters.
    """
    name: str
    substrate: str
    km: float = 0.1  # Michaelis constant (mM)
    vmax: float = 1.0  # Maximum velocity (mM/s)
    kcat: float = 1.0  # Catalytic constant (1/s)
    enzyme_concentration: float = 1.0  # nM
    
    # Regulation
    activators: Dict[str, float] = field(default_factory=dict)  # {activator: factor}
    inhibitors: Dict[str, float] = field(default_factory=dict)   # {inhibitor: factor}
    
    # Cooperativity
    hill_coefficient: float = 1.0
    
    # Temperature dependence
    activation_energy: float = 50000  # J/mol (Arrhenius)
    
    @property
    def turnover_number(self) -> float:
        """Calculate kcat/Km (specificity constant)."""
        if self.km > 0:
            return self.kcat / self.km
        return float('inf')
    
    def apply_temperature(self, temperature: float) -> float:
        """Adjust rate based on temperature (Arrhenius equation)."""
        R = 8.314  # J/(mol·K)
        T_ref = 298.15  # Reference temperature (25°C)
        
        ratio = np.exp(self.activation_energy / R * (1/T_ref - 1/temperature))
        return ratio
    
    def calculate_rate(self, substrate_conc: float, 
                      inhibitor_conc: float = 0.0,
                      activator_conc: float = 0.0,
                      temperature: float = 298.15) -> float:
        """
        Calculate enzyme-catalyzed reaction rate.
        
        Args:
            substrate_conc: Substrate concentration (mM)
            inhibitor_conc: Inhibitor concentration (mM)
            activator_conc: Activator concentration (mM)
            temperature: Temperature (K)
            
        Returns:
            Reaction rate (mM/s)
        """
        # Base Michaelis-Menten rate
        if self.hill_coefficient == 1.0:
            rate = self.vmax * substrate_conc / (self.km + substrate_conc)
        else:
            # Hill equation for cooperativity
            rate = self.vmax * (substrate_conc ** self.hill_coefficient) / (
                self.km ** self.hill_coefficient + substrate_conc ** self.hill_coefficient
            )
        
        # Apply inhibitors
        for inhibitor, ki in self.inhibitors.items():
            if inhibitor_conc > 0:
                # Competitive inhibition
                alpha = 1 + inhibitor_conc / ki
                rate = rate / alpha
        
        # Apply activators
        for activator, factor in self.activators.items():
            if activator_conc > 0:
                rate *= (1 + factor * activator_conc)
        
        # Apply temperature
        rate *= self.apply_temperature(temperature)
        
        return max(0.0, rate)


@dataclass
class ReactionKinetics:
    """
    Reaction with detailed kinetics.
    """
    name: str
    reactants: Dict[str, float]  # {metabolite: stoichiometry}
    products: Dict[str, float]   # {metabolite: stoichiometry}
    
    # Kinetic parameters
    rate_constant: float = 1.0
    order: int = 1  # Reaction order
    
    # Type
    reversible: bool = False
    equilibrium_constant: float = 1.0
    
    # Michaelis-Menten
    km: float = 0.0
    vmax: float = 0.0
    
    # Hill
    hill_n: float = 1.0
    kd: float = 0.0
    
    def calculate_rate(self, concentrations: Dict[str, float]) -> float:
        """Calculate reaction rate."""
        if self.km > 0 and self.vmax > 0:
            # Michaelis-Menten
            substrate = list(self.reactants.keys())[0] if self.reactants else None
            if substrate and substrate in concentrations:
                S = concentrations[substrate]
                return self.vmax * S / (self.km + S)
            return 0.0
        
        elif self.kd > 0 and self.hill_n > 1:
            # Hill equation
            substrate = list(self.reactants.keys())[0] if self.reactants else None
            if substrate and substrate in concentrations:
                S = concentrations[substrate]
                return self.rate_constant * (S ** self.hill_n) / (self.kd + S ** self.hill_n)
            return 0.0
        
        else:
            # Mass action
            rate = self.rate_constant
            for metabolite, stoich in self.reactants.items():
                conc = concentrations.get(metabolite, 0.0)
                rate *= (conc ** stoich)
            
            if self.reversible:
                # Calculate reverse rate
                reverse_rate = 0.0
                for metabolite, stoich in self.products.items():
                    conc = concentrations.get(metabolite, 0.0)
                    reverse_rate *= (conc ** stoich)
                
                rate = rate - reverse_rate / self.equilibrium_constant
            
            return max(0.0, rate)


class MichaelisMentenKinetics:
    """
    Michaelis-Menten enzyme kinetics simulation.
    """
    
    def __init__(self, name: str = "Michaelis-Menten"):
        self.name = name
        self.enzymes: Dict[str, Enzyme] = {}
        self.reactions: List[ReactionKinetics] = []
        self.concentrations: Dict[str, float] = {}
    
    def add_enzyme(self, enzyme: Enzyme):
        """Add enzyme to simulation."""
        self.enzymes[enzyme.name] = enzyme
    
    def add_reaction(self, reaction: ReactionKinetics):
        """Add reaction."""
        self.reactions.append(reaction)
    
    def set_concentration(self, metabolite: str, concentration: float):
        """Set metabolite concentration."""
        self.concentrations[metabolite] = concentration
    
    def simulate(self, duration: float, dt: float = 0.01) -> Dict[str, List[float]]:
        """
        Simulate enzyme kinetics.
        
        Args:
            duration: Simulation duration
            dt: Time step
            
        Returns:
            Time series of concentrations
        """
        time_points = int(duration / dt)
        history = {met: [conc] for met, conc in self.concentrations.items()}
        
        for _ in range(time_points):
            # Calculate rates
            rates = {}
            for reaction in self.reactions:
                rates[reaction.name] = reaction.calculate_rate(self.concentrations)
            
            # Update concentrations
            new_concs = self.concentrations.copy()
            for reaction in self.reactions:
                for reactant, stoich in reaction.reactants.items():
                    rate = rates.get(reaction.name, 0.0)
                    new_concs[reactant] = max(0.0, new_concs.get(reactant, 0.0) - stoich * rate * dt)
                for product, stoich in reaction.products.items():
                    rate = rates.get(reaction.name, 0.0)
                    new_concs[product] = new_concs.get(product, 0.0) + stoich * rate * dt
            
            self.concentrations.update(new_concs)
            
            # Record history
            for met in history:
                history[met].append(self.concentrations.get(met, 0.0))
        
        return history
    
    def calculate_velocity(self, substrate_conc: float, enzyme: Enzyme) -> float:
        """
        Calculate reaction velocity at given substrate concentration.
        """
        return enzyme.vmax * substrate_conc / (enzyme.km + substrate_conc)
    
    def calculate_km_effective(self, inhibitor_conc: float, 
                             ki: float, inhibition_type: str = 'competitive') -> float:
        """
        Calculate effective Km in presence of inhibitor.
        """
        if inhibition_type == 'competitive':
            return self.km * (1 + inhibitor_conc / ki)
        elif inhibition_type == 'uncompetitive':
            return self.km / (1 + inhibitor_conc / ki)
        elif inhibition_type == 'noncompetitive':
            return self.km * (1 + inhibitor_conc / ki)
        return self.km
    
    def lineweaver_burk_transform(self, substrate_concs: List[float],
                                 velocities: List[float]) -> Tuple[float, float]:
        """
        Perform Lineweaver-Burk linearization.
        
        Returns:
            (1/Vmax, Km/Vmax) - intercept and slope
        """
        # 1/v vs 1/[S]
        inv_s = [1/s for s in substrate_concs]
        inv_v = [1/v for v in velocities if v > 0]
        
        # Linear regression
        if len(inv_s) != len(inv_v) or len(inv_s) < 2:
            return (0.0, 0.0)
        
        n = len(inv_s)
        mean_x = sum(inv_s) / n
        mean_y = sum(inv_v) / n
        
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(inv_s, inv_v))
        ss_xx = sum((x - mean_x) ** 2 for x in inv_s)
        
        if ss_xx == 0:
            return (0.0, 0.0)
        
        slope = ss_xy / ss_xx
        intercept = mean_y - slope * mean_x
        
        return (intercept, slope)
    
    def predict_product_inhibition(self, product_conc: float, ki: float) -> float:
        """
        Predict rate reduction due to product inhibition.
        """
        return 1.0 / (1 + product_conc / ki)


class KineticsEngine:
    """
    Comprehensive biochemical kinetics simulation engine.
    """
    
    def __init__(self, name: str = "Kinetics Engine"):
        self.name = name
        self.species: Dict[str, float] = {}
        self.reactions: List[ReactionKinetics] = {}
        self.kinetics_functions: Dict[str, Callable] = {}
        
        # Parameters
        self.temperature: float = 298.15
        self.pH: float = 7.0
        self.ionic_strength: float = 0.15
        
        # Simulation state
        self.time: float = 0.0
        self.history: List[Dict[str, float]] = []
    
    def add_species(self, name: str, initial_concentration: float):
        """Add molecular species."""
        self.species[name] = initial_concentration
    
    def add_reaction(self, name: str, reaction: ReactionKinetics):
        """Add reaction."""
        self.reactions[name] = reaction
        for met in list(reaction.reactants.keys()) + list(reaction.products.keys()):
            if met not in self.species:
                self.species[met] = 0.0
    
    def set_rate_function(self, reaction_name: str, func: Callable):
        """
        Set custom rate function for reaction.
        
        Args:
            reaction_name: Name of reaction
            func: Function(species_dict, time) -> rate
        """
        self.kinetics_functions[reaction_name] = func
    
    def simulate(self, duration: float, dt: float = 0.01,
                method: str = 'euler') -> Dict[str, List[float]]:
        """
        Simulate kinetics.
        
        Args:
            duration: Duration in seconds
            dt: Time step
            method: Integration method ('euler', 'rk4')
        """
        time_points = int(duration / dt)
        
        # Initialize history
        history = {name: [conc] for name, conc in self.species.items()}
        
        for step in range(time_points):
            if method == 'euler':
                self._euler_step(dt)
            elif method == 'rk4':
                self._rk4_step(dt)
            elif method == 'gill':
                self._gill_step(dt)
            
            self.time += dt
            
            # Record
            for name in history:
                history[name].append(self.species.get(name, 0.0))
        
        return history
    
    def _euler_step(self, dt: float):
        """Euler integration step."""
        # Calculate rates
        rates = {}
        for name, reaction in self.reactions.items():
            if name in self.kinetics_functions:
                rates[name] = self.kinetics_functions[name](self.species, self.time)
            else:
                rates[name] = reaction.calculate_rate(self.species)
        
        # Calculate changes
        changes = {name: 0.0 for name in self.species}
        
        for name, reaction in self.reactions.items():
            rate = rates[name]
            
            for reactant, stoich in reaction.reactants.items():
                changes[reactant] -= stoich * rate
            for product, stoich in reaction.products.items():
                changes[product] += stoich * rate
        
        # Update
        for name in self.species:
            self.species[name] = max(0.0, self.species[name] + changes[name] * dt)
    
    def _rk4_step(self, dt: float):
        """Runge-Kutta 4th order integration."""
        # k1
        k1 = self._calculate_derivatives()
        
        # k2
        state_backup = self.species.copy()
        for name in self.species:
            self.species[name] += 0.5 * dt * k1.get(name, 0.0)
        k2 = self._calculate_derivatives()
        
        # k3
        for name in self.species:
            self.species[name] = state_backup[name] + 0.5 * dt * k2.get(name, 0.0)
        k3 = self._calculate_derivatives()
        
        # k4
        for name in self.species:
            self.species[name] = state_backup[name] + dt * k3.get(name, 0.0)
        k4 = self._calculate_derivatives()
        
        # Combine
        for name in self.species:
            derivative = (k1.get(name, 0.0) + 2*k2.get(name, 0.0) + 
                        2*k3.get(name, 0.0) + k4.get(name, 0.0)) / 6
            self.species[name] = max(0.0, state_backup[name] + derivative * dt)
    
    def _gill_step(self, dt: float):
        """Gill's method (modified RK4)."""
        state_backup = self.species.copy()
        
        # Stage 1
        k1 = self._calculate_derivatives()
        
        # Stage 2
        for name in self.species:
            self.species[name] = state_backup[name] + 0.5 * dt * k1.get(name, 0.0)
        k2 = self._calculate_derivatives()
        
        # Stage 3
        for name in self.species:
            self.species[name] = (state_backup[name] + 
                                 dt * (-0.5 * k1.get(name, 0.0) + k2.get(name, 0.0)))
        k3 = self._calculate_derivatives()
        
        # Stage 4
        for name in self.species:
            self.species[name] = (state_backup[name] + 
                                 dt * (k1.get(name, 0.0) - k2.get(name, 0.0) + k3.get(name, 0.0)))
        k4 = self._calculate_derivatives()
        
        # Update
        for name in self.species:
            self.species[name] = max(0.0, state_backup[name] + 
                                    dt * (k1.get(name, 0.0) + 3*k2.get(name, 0.0) + 
                                         3*k3.get(name, 0.0) + k4.get(name, 0.0)) / 8)
    
    def _calculate_derivatives(self) -> Dict[str, float]:
        """Calculate derivatives for all species."""
        derivatives = {name: 0.0 for name in self.species}
        
        for name, reaction in self.reactions.items():
            if name in self.kinetics_functions:
                rate = self.kinetics_functions[name](self.species, self.time)
            else:
                rate = reaction.calculate_rate(self.species)
            
            for reactant, stoich in reaction.reactants.items():
                derivatives[reactant] -= stoich * rate
            for product, stoich in reaction.products.items():
                derivatives[product] += stoich * rate
        
        return derivatives
    
    def equilibrium_concentrations(self, tolerance: float = 1e-6,
                                  max_iterations: int = 1000) -> Dict[str, float]:
        """
        Calculate equilibrium concentrations.
        """
        concentrations = self.species.copy()
        
        for _ in range(max_iterations):
            changes = self._calculate_derivatives()
            
            max_change = max(abs(c) for c in changes.values())
            if max_change < tolerance:
                break
            
            # Step toward equilibrium
            for name in concentrations:
                concentrations[name] += changes[name] * 0.1
                concentrations[name] = max(0.0, concentrations[name])
        
        return concentrations
    
    def sensitivity_analysis(self, parameter: str) -> Dict[str, float]:
        """
        Calculate sensitivity coefficients (d[species]/d[parameter]).
        """
        sensitivities = {}
        
        # Store baseline
        baseline = self.species.copy()
        
        # Perturb parameter
        epsilon = 0.01
        if parameter in self.species:
            self.species[parameter] *= (1 + epsilon)
        elif parameter in self.reactions:
            self.reactions[parameter].rate_constant *= (1 + epsilon)
        
        # Simulate briefly
        self.simulate(0.1, dt=0.01)
        perturbed = self.species.copy()
        
        # Calculate sensitivities
        for name in baseline:
            if baseline[name] > 0:
                sensitivities[name] = (perturbed[name] - baseline[name]) / (epsilon * baseline[name])
            else:
                sensitivities[name] = 0.0
        
        # Restore
        self.species = baseline
        self.time = 0.0
        
        return sensitivities


class DrugInteraction:
    """
    Drug-target interaction kinetics.
    """
    
    def __init__(self, drug_name: str, target: str):
        self.drug_name = drug_name
        self.target = target
        self.kd: float = 0.0  # Dissociation constant
        self.ic50: float = 0.0  # Half-maximal inhibitory concentration
        self.ec50: float = 0.0  # Half-maximal effective concentration
    
    def calculate_occupancy(self, drug_concentration: float) -> float:
        """
        Calculate target occupancy by drug.
        """
        if self.kd > 0:
            return drug_concentration / (self.kd + drug_concentration)
        return 0.0
    
    def calculate_inhibition(self, substrate_conc: float, 
                           drug_concentration: float) -> float:
        """
        Calculate inhibition of enzymatic reaction.
        """
        occupancy = self.calculate_occupancy(drug_concentration)
        base_rate = substrate_conc / (self.ic50 + substrate_conc) if self.ic50 > 0 else 1.0
        return base_rate * (1 - occupancy)
    
    def calculate_synergy(self, other_drug: 'DrugInteraction',
                         drug_conc: float, other_conc: float) -> float:
        """
        Calculate synergy factor with another drug.
        """
        occupancy1 = self.calculate_occupancy(drug_conc)
        occupancy2 = other_drug.calculate_occupancy(other_conc)
        
        # Bliss independence model
        expected = occupancy1 + occupancy2 - occupancy1 * occupancy2
        
        return expected


def hill_equation(ligand_conc: float, kd: float, n: float = 1.0) -> float:
    """
    Hill equation for cooperative binding.
    
    Args:
        ligand_conc: Ligand concentration
        kd: Dissociation constant
        n: Hill coefficient
        
    Returns:
        Fractional saturation (0-1)
    """
    return (ligand_conc ** n) / (kd + ligand_conc ** n)


def michaelis_menten(substrate_conc: float, km: float, vmax: float) -> float:
    """
    Michaelis-Menten equation.
    """
    return vmax * substrate_conc / (km + substrate_conc)


def enzyme_efficiency(kcat: float, km: float) -> float:
    """
    Calculate enzyme efficiency (kcat/Km).
    """
    return kcat / km if km > 0 else float('inf')


def calculate_delta_g_standard(delta_h: float, delta_s: float, 
                              temperature: float = 298.15) -> float:
    """
    Calculate standard Gibbs free energy change.
    
    Args:
        delta_h: Enthalpy change (J/mol)
        delta_s: Entropy change (J/(mol·K))
        temperature: Temperature (K)
        
    Returns:
        Gibbs free energy (J/mol)
    """
    return delta_h - temperature * delta_s


def arrhenius_rate(temperature: float, pre_exponential: float,
                  activation_energy: float) -> float:
    """
    Arrhenius equation for temperature-dependent rate.
    """
    R = 8.314  # J/(mol·K)
    return pre_exponential * np.exp(-activation_energy / (R * temperature))
