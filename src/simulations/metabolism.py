"""
Metabolic pathway simulation for Synthia.
Models metabolic networks, flux analysis, and pathway dynamics.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random

from ..core.biochemistry import Molecule, Reaction, BiochemicalNetwork


class PathwayType(Enum):
    """Types of metabolic pathways."""
    GLYCOLYSIS = "glycolysis"
    TCA_CYCLE = "tca_cycle"
    OXIDATIVE_PHOSPHORYLATION = "oxidative_phosphorylation"
    PHOTOSYNTHESIS = "photosynthesis"
    PENTOSE_PHOSPHATE = "pentose_phosphate"
    UREA_CYCLE = "urea_cycle"
    Fatty_acid_synthesis = "fatty_acid_synthesis"
    Fatty_acid_oxidation = "fatty_acid_oxidation"
    AMINO_ACID_BIOSYNTHESIS = "amino_acid_biosynthesis"
    NUCLEOTIDE_BIOSYNTHESIS = "nucleotide_biosynthesis"


@dataclass
class MetabolicPathway:
    """
    Represents a metabolic pathway with connected reactions.
    """
    name: str
    pathway_type: PathwayType
    reactions: List[Reaction] = field(default_factory=list)
    enzymes: Dict[str, str] = field(default_factory=dict)  # {enzyme_name: reaction_name}
    regulatory_effectors: Dict[str, List[str]] = field(default_factory=dict)
    
    # Pathway properties
    energy_yield: float = 0.0  # ATP yield
    carbon_yield: float = 0.0  # Carbon atoms conserved
    
    def add_reaction(self, reaction: Reaction, enzyme: str = None):
        """Add reaction to pathway."""
        self.reactions.append(reaction)
        if enzyme:
            self.enzymes[enzyme] = reaction.name
    
    def get_pathway_length(self) -> int:
        """Get number of reactions in pathway."""
        return len(self.reactions)
    
    def get_intermediates(self) -> List[Molecule]:
        """Get intermediate metabolites."""
        intermediates = []
        for reaction in self.reactions:
            for mol in reaction.reactants:
                if mol not in reaction.products:
                    intermediates.append(mol)
            for mol in reaction.products:
                if mol not in reaction.reactants:
                    intermediates.append(mol)
        return list(set(intermediates))
    
    def get_ rate_limiting_step(self) -> Optional[Reaction]:
        """Identify rate-limiting step (simplified)."""
        if not self.reactions:
            return None
        # Return reaction with lowest rate constant as proxy
        return min(self.reactions, key=lambda r: r.rate_constant)


@dataclass
class FluxDistribution:
    """
    Represents flux through metabolic network.
    """
    fluxes: Dict[str, float] = field(default_factory=dict)  # {reaction_name: flux}
    objective_value: float = 0.0
    
    def get_flux(self, reaction: str) -> float:
        """Get flux for a reaction."""
        return self.fluxes.get(reaction, 0.0)
    
    def set_flux(self, reaction: str, flux: float):
        """Set flux for a reaction."""
        self.fluxes[reaction] = flux
    
    def total_flux(self) -> float:
        """Get total absolute flux."""
        return sum(abs(f) for f in self.fluxes.values())
    
    def carbon_flux(self, carbon_atoms: Dict[str, int]) -> float:
        """Calculate carbon flux."""
        total_carbon = 0.0
        for reaction, flux in self.fluxes.items():
            if reaction in carbon_atoms:
                total_carbon += abs(flux * carbon_atoms[reaction])
        return total_carbon


class MetabolicSimulation:
    """
    Metabolic network simulation with kinetic and constraint-based modeling.
    """
    
    def __init__(self, name: str = "Metabolism"):
        self.name = name
        
        # Network
        self.network = BiochemicalNetwork(name)
        self.pathways: Dict[PathwayType, MetabolicPathway] = {}
        
        # State
        self.concentrations: Dict[Molecule, float] = {}
        self.flux_distribution: FluxDistribution = FluxDistribution()
        
        # Parameters
        self.time: float = 0.0
        self.temperature: float = 298.15  # K
        
        # History
        self.concentration_history: List[Dict[Molecule, float]] = []
        self.flux_history: List[Dict[str, float]] = []
        
        # Constraints
        self.upper_bounds: Dict[str, float] = {}
        self.lower_bounds: Dict[str, float] = {}
    
    def add_pathway(self, pathway: MetabolicPathway):
        """Add metabolic pathway."""
        self.pathways[pathway.pathway_type] = pathway
        for reaction in pathway.reactions:
            self.network.add_reaction(reaction)
    
    def set_concentration(self, molecule: Molecule, concentration: float):
        """Set molecule concentration."""
        self.concentrations[molecule] = concentration
    
    def set_flux_bounds(self, reaction_name: str, lower: float, upper: float):
        """Set flux bounds for a reaction."""
        self.lower_bounds[reaction_name] = lower
        self.upper_bounds[reaction_name] = upper
    
    def simulate_kinetics(self, delta_time: float = 0.01, duration: float = 1.0):
        """
        Simulate metabolic dynamics using kinetic equations.
        
        Args:
            delta_time: Time step in seconds
            duration: Simulation duration in seconds
        """
        dt = delta_time
        
        while self.time < duration:
            # Calculate reaction rates
            rates = {}
            for reaction in self.network.reactions:
                rate = reaction.calculate_rate(self.concentrations, self.time)
                rates[reaction.name] = rate
            
            # Update concentrations
            for reaction in self.network.reactions:
                rate = rates[reaction.name]
                
                # Update reactants
                for mol, stoich in reaction.reactants.items():
                    delta = -rate * stoich * dt
                    current = self.concentrations.get(mol, 0.0)
                    self.concentrations[mol] = max(0.0, current + delta)
                
                # Update products
                for mol, stoich in reaction.products.items():
                    delta = rate * stoich * dt
                    current = self.concentrations.get(mol, 0.0)
                    self.concentrations[mol] = current + delta
            
            self.time += dt
            
            # Record history periodically
            if len(self.concentration_history) % 100 == 0:
                self.concentration_history.append(self.concentrations.copy())
                self.flux_history.append(rates.copy())
    
    def flux_balance_analysis(self, objective: str = "biomass",
                            medium: Dict[str, float] = None) -> FluxDistribution:
        """
        Perform Flux Balance Analysis (FBA).
        
        Args:
            objective: Reaction to optimize (default: biomass)
            medium: Nutrient uptake rates
            
        Returns:
            FluxDistribution with optimal fluxes
        """
        # Simplified FBA implementation
        # Full implementation would use linear programming
        
        fluxes = {}
        
        for reaction in self.network.reactions:
            name = reaction.name
            
            # Check bounds
            lower = self.lower_bounds.get(name, -1000)
            upper = self.upper_bounds.get(name, 1000)
            
            # Simple heuristic: maximize ATP production
            if 'ATP' in name or 'biomass' in name.lower():
                flux = upper * 0.8
            elif 'glucose' in name.lower() or 'uptake' in name.lower():
                flux = medium.get(name, 10.0) if medium else 10.0
            else:
                # Default flux
                flux = (lower + upper) / 2
            
            fluxes[name] = flux
        
        self.flux_distribution = FluxDistribution(
            fluxes=fluxes,
            objective_value=fluxes.get(objective, 0.0)
        )
        
        return self.flux_distribution
    
    def flux_variance_analysis(self, samples: int = 100) -> Dict[str, Tuple[float, float]]:
        """
        Analyze flux variance using Monte Carlo sampling.
        
        Args:
            samples: Number of random samples
            
        Returns:
            Dict of {reaction: (mean, std)} flux statistics
        """
        flux_samples = {r.name: [] for r in self.network.reactions}
        
        for _ in range(samples):
            # Perturb bounds randomly
            for reaction in self.network.reactions:
                lower = self.lower_bounds.get(reaction.name, -1000)
                upper = self.upper_bounds.get(reaction.name, 1000)
                
                # Sample from uniform distribution
                flux = random.uniform(lower, upper)
                flux_samples[reaction.name].append(flux)
        
        # Calculate statistics
        stats = {}
        for name, samples in flux_samples.items():
            if samples:
                stats[name] = (np.mean(samples), np.std(samples))
        
        return stats
    
    def identify_essential_reactions(self, gene_knockouts: List[str] = None) -> List[str]:
        """
        Identify essential reactions (required for viability).
        """
        # Simplified essential reaction identification
        essential = []
        
        for reaction in self.network.reactions:
            # Check if reaction produces essential metabolites
            for product in reaction.products:
                if 'ATP' in product.name or 'biomass' in product.name:
                    essential.append(reaction.name)
                    break
        
        return list(set(essential))
    
    def calculate_yield(self, substrate: str, product: str) -> float:
        """
        Calculate yield of product from substrate.
        """
        substrate_flux = abs(self.flux_distribution.get_flux(substrate))
        product_flux = abs(self.flux_distribution.get_flux(product))
        
        if substrate_flux == 0:
            return 0.0
        
        return product_flux / substrate_flux
    
    def predict_growth_rate(self, glucose_uptake: float = 10.0) -> float:
        """
        Predict growth rate based on glucose uptake.
        Uses empirical relationship.
        """
        # Yields per g glucose (theoretical maximum)
        Y_atp = 32  # ATP per glucose
        Y_biomass = 0.5  # g biomass per g glucose
        
        # Maintenance ATP
        maintenance = 7.6  # mmol ATP/gDW/h
        
        # Calculate ATP from glucose
        atp_from_glucose = glucose_uptake * Y_atp
        
        # Net ATP for growth
        net_atp = max(0, atp_from_glucose - maintenance)
        
        # Growth rate (per hour)
        mu_max = 0.5  # Maximum growth rate
        yield_factor = Y_biomass
        
        growth_rate = min(mu_max, net_atp * yield_factor * 0.1)
        
        return max(0.0, growth_rate)
    
    def simulate_catabolite_repression(self, carbon_sources: Dict[str, float]) -> str:
        """
        Simulate carbon catabolite repression.
        Returns which carbon source is preferred.
        """
        # Preference order (simplified)
        preferences = {
            'glucose': 1.0,
            'fructose': 0.9,
            'maltose': 0.8,
            'lactose': 0.7,
            'acetate': 0.5,
            'ethanol': 0.4,
        }
        
        best_source = None
        best_score = -1
        
        for source, concentration in carbon_sources.items():
            if concentration > 0:
                preference = preferences.get(source.lower(), 0.5)
                score = preference * concentration
                if score > best_score:
                    best_score = score
                    best_source = source
        
        return best_source or "none"
    
    def calculate_atp_yield(self, substrate: str) -> float:
        """
        Calculate theoretical ATP yield from substrate.
        """
        atp_yields = {
            'glucose': 32,  # Per glycolysis + oxidative phosphorylation
            'pyruvate': 12.5,
            'acetate': 10,
            'fatty_acid': 100,  # Per beta-oxidation cycle
        }
        
        return atp_yields.get(substrate.lower(), 0.0)
    
    def get_metabolic_state(self) -> Dict:
        """Get current metabolic state summary."""
        return {
            'time': self.time,
            'num_reactions': len(self.network.reactions),
            'num_metabolites': len(self.network.molecules),
            'total_flux': self.flux_distribution.total_flux(),
            'pathways': [p.name for p in self.pathways.values()]
        }
    
    def build_glycolysis(self) -> MetabolicPathway:
        """Build glycolysis pathway (Embden-Meyerhof-Parnas)."""
        pathway = MetabolicPathway(
            name="Glycolysis",
            pathway_type=PathwayType.GLYCOLYSIS,
            energy_yield=2.0  # Net ATP
        )
        
        # Create key metabolites
        glucose = Molecule("glucose", formula="C6H12O6", concentration=5.0)
        g6p = Molecule("G6P", formula="C6H13O9P", concentration=0.0)
        f6p = Molecule("F6P", formula="C6H13O9P", concentration=0.0)
        f16bp = Molecule("F16BP", formula="C6H14O12P2", concentration=0.0)
        g3p = Molecule("G3P", formula="C3H5O6P", concentration=0.0)
        pyruvate = Molecule("pyruvate", formula="C3H3O3", concentration=0.0)
        atp = Molecule("ATP", concentration=2.0)
        adp = Molecule("ADP", concentration=1.0)
        nad = Molecule("NAD+", concentration=2.0)
        nadh = Molecule("NADH", concentration=0.0)
        
        # Hexokinase
        hexokinase = Reaction(
            name="hexokinase",
            reactants={glucose: 1, atp: 1},
            products={g6p: 1, adp: 1},
            kinetic_type="michaelis_menten",
            km=0.1,
            vmax=1.0,
            delta_g=-16.7  # kJ/mol
        )
        
        # Phosphofructokinase (rate limiting)
        pfk = Reaction(
            name="phosphofructokinase",
            reactants={g6p: 1, atp: 1},
            products={f16bp: 1, adp: 1},
            kinetic_type="michaelis_menten",
            km=0.1,
            vmax=0.5,
            delta_g=-14.2
        )
        
        # Aldolase
        aldolase = Reaction(
            name="aldolase",
            reactants={f16bp: 1},
            products={g3p: 2},
            kinetic_type="mass_action",
            rate_constant=0.5,
            delta_g=23.9
        )
        
        # GAP dehydrogenase
        gapdh = Reaction(
            name="GAP_dehydrogenase",
            reactants={g3p: 1, nad: 1},
            products={nadh: 1},
            kinetic_type="mass_action",
            rate_constant=0.5,
            delta_g=6.3
        )
        
        # Pyruvate kinase
        pk = Reaction(
            name="pyruvate_kinase",
            reactants={g3p: 1, adp: 1},
            products={pyruvate: 1, atp: 1},
            kinetic_type="michaelis_menten",
            km=0.2,
            vmax=1.0,
            delta_g=-31.4
        )
        
        pathway.add_reaction(hexokinase, "HK")
        pathway.add_reaction(pfk, "PFK")
        pathway.add_reaction(aldolase, "ALD")
        pathway.add_reaction(gapdh, "GAPDH")
        pathway.add_reaction(pk, "PK")
        
        self.add_pathway(pathway)
        
        # Set initial concentrations
        self.set_concentration(glucose, 5.0)
        self.set_concentration(g6p, 0.2)
        self.set_concentration(f6p, 0.1)
        self.set_concentration(f16bp, 0.1)
        self.set_concentration(g3p, 0.1)
        self.set_concentration(pyruvate, 0.5)
        self.set_concentration(atp, 2.0)
        self.set_concentration(adp, 1.0)
        self.set_concentration(nad, 2.0)
        self.set_concentration(nadh, 0.0)
        
        return pathway
    
    def build_tca_cycle(self) -> MetabolicPathway:
        """Build TCA (Krebs) cycle."""
        pathway = MetabolicPathway(
            name="TCA Cycle",
            pathway_type=PathwayType.TCA_CYCLE,
            energy_yield=10.0  # NADH, FADH2 yield
        )
        
        # Key metabolites
        oaa = Molecule("oxaloacetate", formula="C4H4O5", concentration=0.0)
        citrate = Molecule("citrate", formula="C6H8O7", concentration=0.0)
        akg = Molecule("alpha_ketoglutarate", formula="C5H6O5", concentration=0.0)
        succoa = Molecule("succinyl-CoA", formula="C25H40N7O19P3S", concentration=0.0)
        succinate = Molecule("succinate", formula="C4H6O4", concentration=0.0)
        fumarate = Molecule("fumarate", formula="C4H4O4", concentration=0.0)
        malate = Molecule("malate", formula="C4H6O5", concentration=0.0)
        co2 = Molecule("CO2", concentration=0.0)
        nadh = Molecule("NADH", concentration=0.0)
        nad = Molecule("NAD+", concentration=2.0)
        fadh2 = Molecule("FADH2", concentration=0.0)
        fad = Molecule("FAD", concentration=1.0)
        gtp = Molecule("GTP", concentration=0.0)
        gdp = Molecule("GDP", concentration=1.0)
        
        # Citrate synthase
        cs = Reaction(
            name="citrate_synthase",
            reactants={oaa: 1, acetyl_coa: 1, h2o: 1},
            products={citrate: 1, coa: 1},
            kinetic_type="mass_action",
            rate_constant=0.3,
            delta_g=-35.5
        )
        
        # Isocitrate dehydrogenase
        idh = Reaction(
            name="isocitrate_dehydrogenase",
            reactants={citrate: 1, nad: 1},
            products={akg: 1, nadh: 1, co2: 1},
            kinetic_type="michaelis_menten",
            km=0.05,
            vmax=0.2,
            delta_g=-8.4
        )
        
        # Alpha-ketoglutarate dehydrogenase
        akgd = Reaction(
            name="alpha_kg_dehydrogenase",
            reactants={akg: 1, nad: 1, coa: 1},
            products={succoa: 1, nadh: 1, co2: 1},
            kinetic_type="mass_action",
            rate_constant=0.1,
            delta_g=-30.1
        )
        
        # Succinyl-CoA synthetase
        scs = Reaction(
            name="succinyl_coa_synthetase",
            reactants={succoa: 1, gdp: 1, pi: 1},
            products={succinate: 1, gtp: 1, coa: 1},
            kinetic_type="mass_action",
            rate_constant=0.2,
            delta_g=-3.4
        )
        
        # Succinate dehydrogenase
        sd = Reaction(
            name="succinate_dehydrogenase",
            reactants={succinate: 1, fad: 1},
            products={fumarate: 1, fadh2: 1},
            kinetic_type="mass_action",
            rate_constant=0.2,
            delta_g=0.0
        )
        
        # Fumarase
        fh = Reaction(
            name="fumarase",
            reactants={fumarate: 1, h2o: 1},
            products={malate: 1},
            kinetic_type="mass_action",
            rate_constant=0.3,
            delta_g=-3.8
        )
        
        # Malate dehydrogenase
        mdh = Reaction(
            name="malate_dehydrogenase",
            reactants={malate: 1, nad: 1},
            products={oaa: 1, nadh: 1},
            kinetic_type="mass_action",
            rate_constant=0.3,
            delta_g=29.7
        )
        
        # Add placeholder molecules if not defined
        acetyl_coa = Molecule("acetyl-CoA", concentration=1.0)
        h2o = Molecule("H2O", concentration=100.0)
        coa = Molecule("CoA", concentration=1.0)
        pi = Molecule("Pi", concentration=10.0)
        
        pathway.add_reaction(cs, "CS")
        pathway.add_reaction(idh, "IDH")
        pathway.add_reaction(akgd, "AKGD")
        pathway.add_reaction(scs, "SCS")
        pathway.add_reaction(sd, "SDH")
        pathway.add_reaction(fh, "FH")
        pathway.add_reaction(mdh, "MDH")
        
        self.add_pathway(pathway)
        
        return pathway


# Helper molecules for TCA
acetyl_coa = Molecule("acetyl-CoA", concentration=1.0)
h2o = Molecule("H2O", concentration=100.0)
coa = Molecule("CoA", concentration=1.0)
pi = Molecule("Pi", concentration=10.0)
