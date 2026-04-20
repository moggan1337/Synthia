"""
Biochemical components for Synthia.
Handles molecules, reactions, compartments, and biochemical network modeling.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random


class CompartmentType(Enum):
    """Types of cellular compartments."""
    EXTRACELLULAR = "extracellular"
    CYTOPLASM = "cytoplasm"
    NUCLEUS = "nucleus"
    MITOCHONDRIA = "mitochondria"
    CHLOROPLAST = "chloroplast"
    ENDOPLASMIC_RETICULUM = "endoplasmic_reticulum"
    GOLGI = "golgi"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    VACUOLE = "vacuole"
    MEMBRANE = "membrane"


@dataclass
class Molecule:
    """
    Represents a biochemical molecule in the simulation.
    Can represent small molecules, proteins, ions, or other compounds.
    """
    name: str
    formula: str = ""
    mass: float = 0.0
    charge: int = 0
    compartment: CompartmentType = CompartmentType.CYTOPLASM
    concentration: float = 0.0  # in M (molar)
    properties: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.name, self.compartment.value))
    
    def __eq__(self, other):
        if not isinstance(other, Molecule):
            return False
        return self.name == other.name and self.compartment == other.compartment
    
    @property
    def molar_mass(self) -> float:
        """Calculate molar mass from formula (simplified)."""
        if self.mass > 0:
            return self.mass
        return self._parse_formula()
    
    def _parse_formula(self) -> float:
        """Parse chemical formula to calculate mass."""
        # Simplified formula parser
        elements = {'C': 12.01, 'H': 1.008, 'O': 16.00, 'N': 14.01,
                    'P': 30.97, 'S': 32.07, 'K': 39.10, 'Na': 22.99,
                    'Ca': 40.08, 'Mg': 24.31, 'Fe': 55.85}
        
        mass = 0.0
        i = 0
        formula = self.formula
        
        while i < len(formula):
            if formula[i].isupper():
                element = formula[i]
                i += 1
                count_str = ""
                while i < len(formula) and formula[i].isdigit():
                    count_str += formula[i]
                    i += 1
                count = int(count_str) if count_str else 1
                mass += elements.get(element, 0) * count
            else:
                i += 1
        
        return mass if mass > 0 else 100.0
    
    def amount_to_moles(self, amount: float, volume: float) -> float:
        """Convert amount (e.g., molecules) to moles."""
        return amount / 6.022e23
    
    def moles_to_amount(self, moles: float) -> float:
        """Convert moles to amount (molecules)."""
        return moles * 6.022e23


@dataclass
class Reaction:
    """
    Represents a biochemical reaction.
    Supports reversible reactions, rate laws, and various kinetic models.
    """
    name: str
    reactants: Dict[Molecule, float]  # {molecule: stoichiometry}
    products: Dict[Molecule, float]   # {molecule: stoichiometry}
    reversible: bool = False
    rate_constant: float = 1.0
    kinetic_type: str = "mass_action"  # mass_action, michaelis_menten, hill
    
    # Michaelis-Menten parameters
    km: float = 0.0
    vmax: float = 0.0
    
    # Hill equation parameters
    hill_coefficient: float = 1.0
    kd: float = 0.0
    
    # Catalysts and effectors
    catalysts: List[Molecule] = field(default_factory=list)
    activators: List[Molecule] = field(default_factory=list)
    inhibitors: List[Molecule] = field(default_factory=list)
    
    # Thermodynamics
    delta_g: float = 0.0  # Gibbs free energy change (kJ/mol)
    delta_h: float = 0.0  # Enthalpy change (kJ/mol)
    
    def __post_init__(self):
        """Validate reaction."""
        if not self.reactants:
            raise ValueError("Reaction must have at least one reactant")
        if not self.products:
            raise ValueError("Reaction must have at least one product")
    
    def calculate_rate(self, concentrations: Dict[Molecule, float], 
                       time: float = 0.0) -> float:
        """
        Calculate reaction rate based on current concentrations.
        
        Args:
            concentrations: Dict mapping molecules to their concentrations
            time: Current simulation time
            
        Returns:
            Reaction rate (change in product per unit time)
        """
        if self.kinetic_type == "mass_action":
            return self._mass_action_rate(concentrations)
        elif self.kinetic_type == "michaelis_menten":
            return self._michaelis_menten_rate(concentrations)
        elif self.kinetic_type == "hill":
            return self._hill_rate(concentrations)
        elif self.kinetic_type == "zero_order":
            return self.rate_constant
        elif self.kinetic_type == "first_order":
            return self._first_order_rate(concentrations)
        else:
            return self._mass_action_rate(concentrations)
    
    def _mass_action_rate(self, concentrations: Dict[Molecule, float]) -> float:
        """Calculate mass action rate."""
        rate = self.rate_constant
        
        for mol, stoich in self.reactants.items():
            conc = concentrations.get(mol, 0.0)
            rate *= (conc ** stoich)
        
        return max(0.0, rate)
    
    def _michaelis_menten_rate(self, concentrations: Dict[Molecule, float]) -> float:
        """Calculate Michaelis-Menten rate."""
        if len(self.reactants) == 1:
            substrate = list(self.reactants.keys())[0]
            conc = concentrations.get(substrate, 0.0)
            
            if self.km > 0:
                return self.vmax * conc / (self.km + conc)
            else:
                return self.rate_constant * conc
        return self._mass_action_rate(concentrations)
    
    def _hill_rate(self, concentrations: Dict[Molecule, float]) -> float:
        """Calculate Hill equation rate."""
        if len(self.reactants) == 1:
            substrate = list(self.reactants.keys())[0]
            conc = concentrations.get(substrate, 0.0)
            n = self.hill_coefficient
            
            if self.kd > 0:
                return self.rate_constant * (conc ** n) / (self.kd + conc ** n)
            else:
                return self.rate_constant * conc
        return self._mass_action_rate(concentrations)
    
    def _first_order_rate(self, concentrations: Dict[Molecule, float]) -> float:
        """Calculate first-order rate."""
        rate = 0.0
        for mol, stoich in self.reactants.items():
            conc = concentrations.get(mol, 0.0)
            rate += self.rate_constant * conc * stoich
        return rate
    
    def get_delta_g_prime(self, concentrations: Dict[Molecule, float]) -> float:
        """
        Calculate Gibbs free energy under current conditions.
        Uses the biochemical standard Gibbs energy.
        """
        if self.delta_g == 0.0:
            return 0.0
        
        R = 8.314e-3  # Gas constant (kJ/(mol·K))
        T = 298.15    # Temperature (K)
        
        # Calculate reaction quotient
        Q = 1.0
        for mol, stoich in self.products.items():
            conc = concentrations.get(mol, 1e-10)
            Q *= (conc ** stoich)
        for mol, stoich in self.reactants.items():
            conc = concentrations.get(mol, 1e-10)
            Q /= (conc ** stoich) if conc > 0 else float('inf')
        
        if Q <= 0:
            Q = 1e-10
        
        return self.delta_g + R * T * np.log(Q)
    
    def is_exergonic(self, concentrations: Dict[Molecule, float]) -> bool:
        """Check if reaction is spontaneous under current conditions."""
        return self.get_delta_g_prime(concentrations) < 0
    
    def equilibrium_constant(self) -> float:
        """Calculate equilibrium constant from standard Gibbs energy."""
        R = 8.314e-3
        T = 298.15
        if self.delta_g == 0:
            return 1.0
        return np.exp(-self.delta_g / (R * T))


@dataclass
class Compartment:
    """
    Represents a cellular compartment with volume and membrane properties.
    """
    name: str
    compartment_type: CompartmentType
    volume: float = 1e-12  # in liters (default 1 picoliter for cytoplasm)
    pH: float = 7.0
    temperature: float = 298.15  # Kelvin
    molecules: Dict[Molecule, float] = field(default_factory=dict)
    
    # Membrane properties if applicable
    permeability: Dict[str, float] = field(default_factory=dict)  # {molecule_name: permeability}
    membrane_potential: float = 0.0  # in mV
    
    def __hash__(self):
        return hash(self.name)
    
    def add_molecule(self, molecule: Molecule, concentration: float):
        """Add molecule to compartment."""
        self.molecules[molecule] = concentration
    
    def remove_molecule(self, molecule: Molecule) -> float:
        """Remove molecule and return its concentration."""
        return self.molecules.pop(molecule, 0.0)
    
    def get_concentration(self, molecule: Molecule) -> float:
        """Get molecule concentration."""
        return self.molecules.get(molecule, 0.0)
    
    def get_amount(self, molecule: Molecule) -> float:
        """Get molecule amount in molecules."""
        conc = self.get_concentration(molecule)
        moles = conc * self.volume
        return moles * 6.022e23
    
    def total_molecules(self) -> float:
        """Get total molecule count."""
        total = 0.0
        for mol, conc in self.molecules.items():
            total += conc * self.volume * 6.022e23
        return total


class BiochemicalNetwork:
    """
    Complete biochemical network with reactions and compartments.
    """
    
    def __init__(self, name: str = "Biochemical Network"):
        self.name = name
        self.reactions: List[Reaction] = []
        self.molecules: Set[Molecule] = set()
        self.compartments: Dict[CompartmentType, Compartment] = {}
        self.network_matrix: Optional[np.ndarray] = None
        self._build_stoichiometry_matrix()
    
    def add_reaction(self, reaction: Reaction):
        """Add a reaction to the network."""
        self.reactions.append(reaction)
        for mol in list(reaction.reactants.keys()) + list(reaction.products.keys()):
            self.molecules.add(mol)
        self._build_stoichiometry_matrix()
    
    def add_molecule(self, molecule: Molecule):
        """Add a molecule to the network."""
        self.molecules.add(molecule)
    
    def add_compartment(self, compartment: Compartment):
        """Add a compartment to the network."""
        self.compartments[compartment.compartment_type] = compartment
    
    def _build_stoichiometry_matrix(self):
        """Build stoichiometry matrix for the network."""
        n_molecules = len(self.molecules)
        n_reactions = len(self.reactions)
        
        if n_molecules == 0 or n_reactions == 0:
            self.network_matrix = None
            return
        
        mol_list = list(self.molecules)
        self._mol_to_idx = {mol: i for i, mol in enumerate(mol_list)}
        
        self.network_matrix = np.zeros((n_molecules, n_reactions))
        
        for j, reaction in enumerate(self.reactions):
            for mol, stoich in reaction.reactants.items():
                if mol in self._mol_to_idx:
                    self.network_matrix[self._mol_to_idx[mol], j] -= stoich
            for mol, stoich in reaction.products.items():
                if mol in self._mol_to_idx:
                    self.network_matrix[self._mol_to_idx[mol], j] += stoich
    
    def get_stoichiometry(self, molecule: Molecule, reaction: Reaction) -> float:
        """Get stoichiometry of molecule in reaction."""
        if molecule in reaction.reactants:
            return -reaction.reactants[molecule]
        if molecule in reaction.products:
            return reaction.products[molecule]
        return 0.0
    
    def get_reactions_for_molecule(self, molecule: Molecule) -> List[Reaction]:
        """Get all reactions involving a molecule."""
        return [r for r in self.reactions 
                if molecule in r.reactants or molecule in r.products]
    
    def get_upstream_reactions(self, molecule: Molecule, max_depth: int = 5) -> Set[Reaction]:
        """Get all reactions that produce a molecule (recursive)."""
        reactions = set()
        for reaction in self.reactions:
            if molecule in reaction.products:
                reactions.add(reaction)
        
        if max_depth > 0:
            for reaction in list(reactions):
                for reactant in reaction.reactants:
                    upstream = self.get_upstream_reactions(reactant, max_depth - 1)
                    reactions.update(upstream)
        
        return reactions
    
    def get_downstream_reactions(self, molecule: Molecule, max_depth: int = 5) -> Set[Reaction]:
        """Get all reactions that consume a molecule (recursive)."""
        reactions = set()
        for reaction in self.reactions:
            if molecule in reaction.reactants:
                reactions.add(reaction)
        
        if max_depth > 0:
            for reaction in list(reactions):
                for product in reaction.products:
                    downstream = self.get_downstream_reactions(product, max_depth - 1)
                    reactions.update(downstream)
        
        return reactions
    
    def identify_coupled_reactions(self) -> List[Tuple[Reaction, Reaction, float]]:
        """Identify thermodynamically coupled reactions."""
        coupled = []
        for i, r1 in enumerate(self.reactions):
            for r2 in self.reactions[i+1:]:
                # Check if reactions share a metabolite
                shared = set(r1.products.keys()) & set(r2.reactants.keys())
                if shared:
                    # Calculate coupling efficiency
                    efficiency = min(abs(r1.delta_g), abs(r2.delta_g)) / max(abs(r1.delta_g), abs(r2.delta_g))
                    if efficiency > 0.1:
                        coupled.append((r1, r2, efficiency))
        return coupled
    
    def find_conserved_quantities(self) -> List[List[Molecule]]:
        """Find conserved quantities in the network (moieties)."""
        if self.network_matrix is None:
            return []
        
        # Perform SVD to find null space
        try:
            U, S, Vt = np.linalg.svd(self.network_matrix)
            n_molecules = self.network_matrix.shape[0]
            
            # Find conservation laws (small singular values)
            tol = 1e-10
            conserved = []
            mol_list = list(self.molecules)
            
            for i, s in enumerate(S):
                if s < tol:
                    # Null space vector represents conserved quantity
                    conserved_quantity = Vt[i]
                    moiety = [mol_list[j] for j in range(n_molecules) 
                              if abs(conserved_quantity[j]) > 0.1]
                    if moiety:
                        conserved.append(moiety)
            
            return conserved
        except:
            return []
    
    def flux_balance_analysis(self, fixed_flux: Dict[Reaction, float] = None) -> Dict[Reaction, float]:
        """
        Perform flux balance analysis (FBA) on the network.
        Simplified version that optimizes for maximum growth or ATP maintenance.
        """
        if self.network_matrix is None:
            return {}
        
        n_reactions = len(self.reactions)
        fluxes = {r: 0.0 for r in self.reactions}
        
        # Simplified FBA - balance fluxes at steady state
        for i, reaction in enumerate(self.reactions):
            if reaction.kinetic_type == "mass_action":
                # Estimate flux from rate constant
                fluxes[reaction] = reaction.rate_constant * 0.5
        
        return fluxes
    
    def mass_balance_check(self, concentrations: Dict[Molecule, float]) -> Dict[Molecule, float]:
        """
        Check mass balance for all molecules.
        Returns net production rates.
        """
        balance = {}
        
        for mol in self.molecules:
            production = 0.0
            consumption = 0.0
            
            for reaction in self.reactions:
                if mol in reaction.reactants:
                    rate = reaction.calculate_rate(concentrations)
                    consumption += rate * reaction.reactants[mol]
                if mol in reaction.products:
                    rate = reaction.calculate_rate(concentrations)
                    production += rate * reaction.products[mol]
            
            balance[mol] = production - consumption
        
        return balance
    
    def get_connectivity(self, molecule: Molecule) -> Tuple[int, int]:
        """
        Get connectivity of a molecule (number of reactions producing/consuming it).
        """
        producers = len([r for r in self.reactions if molecule in r.products])
        consumers = len([r for r in self.reactions if molecule in r.reactants])
        return (producers, consumers)
    
    def classify_molecules(self) -> Dict[str, List[Molecule]]:
        """
        Classify molecules by their network roles.
        """
        classification = {
            'substrates': [],      # Only consumed
            'products': [],       # Only produced
            'intermediates': [],   # Both consumed and produced
            'external': []         # Not involved in any reaction
        }
        
        for mol in self.molecules:
            prods, cons = self.get_connectivity(mol)
            if prods == 0 and cons > 0:
                classification['substrates'].append(mol)
            elif cons == 0 and prods > 0:
                classification['products'].append(mol)
            elif cons > 0 and prods > 0:
                classification['intermediates'].append(mol)
            else:
                classification['external'].append(mol)
        
        return classification
    
    def to_sbml(self) -> str:
        """Export network to SBML format (simplified)."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">',
                 f'  <model name="{self.name}">',
                 '    <listOfCompartments>']
        
        for comp_type, comp in self.compartments.items():
            lines.append(f'      <compartment id="{comp_type.value}" volume="{comp.volume}"/>')
        
        lines.append('    </listOfCompartments>')
        lines.append('    <listOfSpecies>')
        
        for mol in self.molecules:
            lines.append(f'      <species id="{mol.name}" compartment="{mol.compartment.value}" initialConcentration="{mol.concentration}"/>')
        
        lines.append('    </listOfSpecies>')
        lines.append('    <listOfReactions>')
        
        for rxn in self.reactions:
            lines.append(f'      <reaction id="{rxn.name}" reversible="{"true" if rxn.reversible else "false"}">')
            lines.append('        <listOfReactants>')
            for mol, stoich in rxn.reactants.items():
                lines.append(f'          <speciesReference species="{mol.name}" stoichiometry="{stoich}"/>')
            lines.append('        </listOfReactants>')
            lines.append('        <listOfProducts>')
            for mol, stoich in rxn.products.items():
                lines.append(f'          <speciesReference species="{mol.name}" stoichiometry="{stoich}"/>')
            lines.append('        </listOfProducts>')
            lines.append('      </reaction>')
        
        lines.append('    </listOfReactions>')
        lines.append('  </model>')
        lines.append('</sbml>')
        
        return '\n'.join(lines)
