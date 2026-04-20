# Synthia - Synthetic Biology Simulator

**Synthia** is a comprehensive computational biology framework for simulating synthetic biology systems. It provides tools for modeling DNA/RNA sequences, protein structures, gene regulatory networks, metabolic pathways, cell signaling, and evolutionary dynamics.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Features](#core-features)
   - [Biological Sequences](#biological-sequences)
   - [Genome Modeling](#genome-modeling)
   - [Cell Simulation](#cell-simulation)
   - [Biochemical Networks](#biochemical-networks)
5. [Simulation Modules](#simulation-modules)
   - [Gene Regulatory Networks](#gene-regulatory-networks)
   - [Metabolic Pathways](#metabolic-pathways)
   - [Biochemical Kinetics](#biochemical-kinetics)
   - [Cell Signaling](#cell-signaling)
   - [Population Dynamics](#population-dynamics)
   - [Cell Division](#cell-division)
6. [Analysis Tools](#analysis-tools)
7. [Visualization](#visualization)
8. [Examples](#examples)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

Synthia is designed for researchers and students in computational biology, synthetic biology, and systems biology. It provides a flexible, extensible framework for:

- **Sequence Analysis**: DNA, RNA, and protein sequence manipulation and analysis
- **Gene Regulatory Network Modeling**: Simulate transcriptional regulation and genetic circuits
- **Metabolic Pathway Simulation**: Model metabolic networks with kinetics
- **Cell Signaling**: Phosphorylation cascades and second messenger dynamics
- **Population Dynamics**: Evolutionary simulations and genetic algorithms
- **Cell Division**: Cell cycle, mitosis, and differentiation

### Key Features

- Comprehensive biological sequence classes with advanced operations
- Modular architecture for easy extension
- Multiple simulation methods (deterministic, stochastic-ready)
- Network analysis tools
- Visualization capabilities
- Drug interaction modeling
- Evolutionary algorithms for synthetic circuit design

---

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0 (optional, for visualization)
- NetworkX >= 2.6.0 (optional, for network analysis)
- Pandas >= 1.3.0 (optional, for data handling)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/moggan1337/Synthia.git
cd Synthia

# Install in development mode
pip install -e .
```

### Install Dependencies

```bash
pip install numpy scipy matplotlib networkx pandas biopython
```

---

## Quick Start

### Basic Usage

```python
from src.core.sequence import DNA, RNA, Protein

# Create DNA sequence
dna = DNA("ATGCGATCGATCGATCG")

# Transcribe to RNA
rna = dna.transcribe()

# Translate to protein
protein = rna.translate()

print(f"DNA: {dna}")
print(f"RNA: {rna}")
print(f"Protein: {protein}")
print(f"GC Content: {dna.gc_content():.2f}%")
```

### Cell Simulation

```python
from src.core.cell import Cell, CellType
from src.core.genome import Genome, Gene, Promoter

# Create a cell
cell = Cell(name="E. coli", cell_type=CellType.BACTERIUM)

# Simulate growth
cell.set_environment({'glucose': 10.0, 'oxygen': 5.0})

# Run simulation
for step in range(100):
    cell.step(delta_time=1.0)
    
print(cell.get_cell_stats())
```

### Gene Regulatory Network

```python
from src.simulations.gene_regulation import GeneRegulatoryNetwork, RegulatorySimulation

# Create network
grn = GeneRegulatoryNetwork(name="Lac Operon Model")

# Add genes
grn.add_gene("lacI", basal_expression=0.8)  # Repressor
grn.add_gene("lacZ", basal_expression=0.1)  # Beta-galactosidase
grn.add_gene("lacY", basal_expression=0.1)  # Permease

# Simulate
sim = RegulatorySimulation(grn)
results = sim.run_with_pulse_input("lacI", pulse_start=0, pulse_end=5, 
                                   pulse_strength=0.1, duration=20.0)
```

---

## Core Features

### Biological Sequences

Synthia provides comprehensive classes for biological sequence manipulation:

#### DNA Class

The `DNA` class supports:

- **Basic Operations**: Creation, slicing, complement, reverse complement
- **Transcription**: Convert to RNA
- **Translation**: Convert to protein
- **Motif Finding**: Search for restriction sites, patterns
- **ORF Detection**: Find open reading frames
- **GC Content**: Calculate nucleotide composition
- **Mutagenesis**: Simulate random mutations

```python
from src.core.sequence import DNA

# Create DNA
dna = DNA("ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGTTGGTATCAAGGTTACAAGACAGGTTTAAGGAGACCAATAGAAACTGGGCTTGTCAG")

# Basic properties
print(f"Length: {len(dna)}")
print(f"GC Content: {dna.gc_content():.2f}%")
print(f"Hash: {dna.hash}")

# Find ORFs
orfs = dna.find_orfs(min_length=100)
print(f"Found {len(orfs)} ORFs")

# Restriction map
cut_sites = dna.restriction_map()
for pos, enzyme in cut_sites.items():
    print(f"{enzyme}: {pos}")
```

#### RNA Class

The `RNA` class provides:

- **Reverse Transcription**: Convert back to DNA
- **Translation**: Convert to protein
- **Secondary Structure**: Basic structure prediction
- **miRNA Binding Sites**: Find potential binding sites

```python
from src.core.sequence import RNA

rna = RNA("AUGGCCAUUGUAAUGGGCCGCUA")
protein = rna.translate()
print(f"Protein: {protein}")
```

#### Protein Class

The `Protein` class supports:

- **Molecular Weight**: Calculate from sequence
- **Isoelectric Point**: Estimate pI
- **Hydrophobicity Profile**: Kyte-Doolittle scale
- **Transmembrane Domain Prediction**: Identify TM helices
- **Secondary Structure Prediction**: Simplified Chou-Fasman
- **Signal Peptide Prediction**: N-terminal targeting

```python
from src.core.sequence import Protein

protein = Protein("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")

print(f"Molecular Weight: {protein.molecular_weight:.2f} Da")
print(f"Isoelectric Point: {protein.isoelectric_point:.2f}")
print(f"Transmembrane domains: {protein.transmembrane_domains()}")
```

---

## Genome Modeling

### Gene Class

```python
from src.core.genome import Gene, GeneType, Promoter, PromoterType, RBS

# Create promoter
promoter = Promoter(
    name="pLac",
    sequence=DNA("TTACGGTTATAATGCGATCGATCG"),
    promoter_type=PromoterType.INDUCIBLE
)

# Create RBS
rbs = RBS(
    name="RBS1",
    sequence=DNA("AGGAGG"),
    ribosome_binding_strength=0.8
)

# Create gene
gene = Gene(
    name="gfp",
    gene_type=GeneType.CODING,
    sequence=DNA("ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"),
    start=0,
    end=714,
    promoter=promoter,
    rbs=rbs
)

# Calculate expression
expression = gene.calculate_expression(
    transcription_factors={'CRP': 0.5},
    inducers=['IPTG']
)
print(f"Expression level: {expression:.4f}")
```

### Genome Class

```python
from src.core.genome import Genome, Gene, Operon

# Create genome
genome = Genome(name="Minimal Genome", species="E. coli K-12", total_length=4641652)

# Add operon
lac_operon = Operon(name="Lac Operon")
lac_operon.set_promoter(promoter)

# Add genes to operon
for gene_data in [("lacZ", 3573), ("lacY", 3685), ("lacA", 4579)]:
    gene = Gene(name=gene_data[0], sequence=DNA(""), start=gene_data[1], end=gene_data[1]+1000)
    lac_operon.add_gene(gene)

genome.add_operon(lac_operon)

# Get statistics
stats = genome.get_genome_stats()
print(f"Total genes: {stats['total_genes']}")
print(f"GC Content: {stats['gc_content']:.2f}%")
```

---

## Cell Simulation

### Cell Class

```python
from src.core.cell import Cell, CellType, CellVolume, MetabolicState

# Create cell
cell = Cell(name="E. coli W3110", cell_type=CellType.BACTERIUM)

# Set environment
cell.set_environment({
    'glucose': 10.0,
    'oxygen': 5.0,
    'nitrogen': 100.0,
    'phosphate': 50.0
})

# Simulate
for i in range(1000):
    cell.step(delta_time=1.0)  # 1 second steps
    
    # Check for division
    if cell.state == CellState.DIVIDING:
        print(f"Cell dividing at t={cell.age:.2f} hours")
        break

# Get statistics
stats = cell.get_cell_stats()
print(f"ATP: {stats['atp_mM']:.2f} mM")
print(f"Energy Charge: {stats['energy_charge']:.3f}")
print(f"Volume: {stats['volume_pL']:.2f} pL")
```

### Population Simulation

```python
from src.core.cell import Population

# Create population
pop = Population(name="Bacterial Culture", initial_count=100)

# Simulate growth
for i in range(100):
    pop.step(delta_time=1.0)
    
    if i % 10 == 0:
        stats = pop.get_population_stats()
        print(f"t={i}: N={stats['population']}, μ={stats['growth_rate']:.3f}")
```

---

## Biochemical Networks

### Molecule and Reaction Classes

```python
from src.core.biochemistry import Molecule, Reaction, Compartment, CompartmentType

# Create molecules
glucose = Molecule("glucose", formula="C6H12O6", concentration=5.0)
atp = Molecule("ATP", concentration=2.0)
g6p = Molecule("G6P", concentration=0.0)
adp = Molecule("ADP", concentration=1.0)

# Create reaction (Hexokinase)
hexokinase = Reaction(
    name="hexokinase",
    reactants={glucose: 1, atp: 1},
    products={g6p: 1, adp: 1},
    kinetic_type="michaelis_menten",
    km=0.1,
    vmax=1.0,
    delta_g=-16.7
)

# Calculate rate
concentrations = {glucose: 5.0, atp: 2.0, g6p: 0.1, adp: 1.0}
rate = hexokinase.calculate_rate(concentrations)
print(f"Reaction rate: {rate:.4f} mM/s")
```

### Biochemical Network

```python
from src.core.biochemistry import BiochemicalNetwork

# Create network
network = BiochemicalNetwork(name="Central Metabolism")

# Add molecules and reactions
network.add_molecule(glucose)
network.add_molecule(atp)
network.add_molecule(g6p)
network.add_molecule(adp)
network.add_reaction(hexokinase)

# Analyze network
stats = network.get_network_statistics()
print(f"Molecules: {stats['num_molecules']}")
print(f"Reactions: {stats['num_reactions']}")
```

---

## Simulation Modules

### Gene Regulatory Networks

```python
from src.simulations.gene_regulation import (
    GeneRegulatoryNetwork, 
    RegulatoryInteraction, 
    InteractionType
)

# Create network
grn = GeneRegulatoryNetwork(name="Toggle Switch")

# Add genes
grn.add_gene("repressor_A")
grn.add_gene("repressor_B")

# Add mutual repression (toggle switch)
grn.add_interaction(RegulatoryInteraction(
    regulator="repressor_A",
    target="repressor_B",
    interaction_type=InteractionType.REPRESSION,
    strength=2.0
))

grn.add_interaction(RegulatoryInteraction(
    regulator="repressor_B",
    target="repressor_A",
    interaction_type=InteractionType.REPRESSION,
    strength=2.0
))

# Run simulation
grn.run_simulation(duration=50.0, delta_time=0.1)

# Find attractors
attractors = grn.find_attractors()
print(f"Found {len(attractors)} attractors")
```

### Metabolic Pathways

```python
from src.simulations.metabolism import MetabolicSimulation, MetabolicPathway, PathwayType

# Create simulation
sim = MetabolicSimulation(name="Glycolysis Model")

# Build glycolysis
pathway = sim.build_glycolysis()

# Simulate kinetics
sim.simulate_kinetics(duration=60.0, delta_time=0.1)

# Run FBA
flux = sim.flux_balance_analysis(objective="ATP")
print(f"ATP production flux: {flux.objective_value:.2f}")
```

### Biochemical Kinetics

```python
from src.simulations.kinetics import Enzyme, MichaelisMentenKinetics, KineticsEngine

# Create enzyme
hexokinase_enzyme = Enzyme(
    name="Hexokinase",
    substrate="glucose",
    km=0.1,
    vmax=1.0,
    kcat=100.0
)

# Create kinetics simulation
km_sim = MichaelisMentenKinetics(name="Hexokinase Kinetics")
km_sim.add_enzyme(hexokinase_enzyme)

# Calculate velocity at various substrate concentrations
concs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
for conc in concs:
    v = km_sim.calculate_velocity(conc, hexokinase_enzyme)
    print(f"[S]={conc:.2f}mM, v={v:.4f} mM/s")
```

### Cell Signaling

```python
from src.simulations.signaling import (
    SignalingPathwaySimulation,
    SignalingPathway,
    Receptor,
    Kinase
)

# Create simulation
sig_sim = SignalingPathwaySimulation(name="MAPK Cascade")
sig_sim.pathway = SignalingPathway.MAPK

# Add receptors
receptor = Receptor(
    name="EGFR",
    ligand="EGF",
    receptor_type="RTK",
    kd=0.1
)
sig_sim.add_receptor(receptor)

# Add kinases
raf = Kinase(name="Raf", substrate="MEK", pathway=SignalingPathway.MAPK)
mek = Kinase(name="MEK", substrate="ERK", pathway=SignalingPathway.MAPK)
erk = Kinase(name="ERK", substrate="ELK1", pathway=SignalingPathway.MAPK)

sig_sim.add_kinase(raf)
sig_sim.add_kinase(mek)
sig_sim.add_kinase(erk)

# Simulate with EGF pulse
results = sig_sim.run_simulation(
    duration=30.0,
    ligand_pulse=(0.0, 5.0, "EGFR")
)

# Get pathway activity
activity = sig_sim.get_pathway_activity()
print(f"TF activation: {activity['tf_activation']:.3f}")
```

### Population Dynamics

```python
from src.simulations.population import PopulationDynamics, EvolutionarySimulation

# Exponential growth
dynamics = PopulationDynamics(name="Bacterial Growth")
dynamics.initial_population = 100
dynamics.growth_rate = 0.5
dynamics.set_model(GrowthModel.EXPONENTIAL)

dynamics.simulate(duration=24.0, dt=0.1)
print(f"Final population: {dynamics.population[-1]:.0f}")

# Logistic growth
dynamics_logistic = PopulationDynamics(name="Logistic Growth")
dynamics_logistic.initial_population = 100
dynamics_logistic.growth_rate = 0.5
dynamics_logistic.carrying_capacity = 10000
dynamics_logistic.set_model(GrowthModel.LOGISTIC)

dynamics_logistic.simulate(duration=24.0, dt=0.1)
print(f"Final population: {dynamics_logistic.population[-1]:.0f}")
```

### Cell Division

```python
from src.simulations.cell_division import CellDivision, DivisionSimulation

# Create division simulation
div_sim = DivisionSimulation(name="Cell Division")

# Add cells
for i in range(10):
    cell = CellDivision(cell_id=i)
    cell.initialize_chromosomes(number=1, length=5000000)
    div_sim.add_cell(cell)

# Run simulation
div_sim.run(duration=48.0, dt=1.0, nutrients=1.0)

# Get statistics
stats = div_sim.get_population_stats()
print(f"Total divisions: {stats['total_divisions']}")
print(f"Current cells: {stats['total_cells']}")
```

---

## Analysis Tools

### Sequence Analysis

```python
from src.analysis.sequence_analysis import SequenceAnalyzer

analyzer = SequenceAnalyzer()

# Calculate complexity
sequence = "ATATATATATATATAT"
complexity = analyzer.calculate_complexity(sequence)
print(f"Complexity: {complexity:.3f}")

# Find repeats
repeats = analyzer.find_repeats(sequence, min_length=3)
print(f"Repeats: {repeats}")

# Calculate dinucleotide bias
bias = analyzer.calculate_dinucleotide_bias(sequence)
print(f"Dinucleotide bias: {bias}")
```

### Network Analysis

```python
from src.analysis.network_analysis import NetworkAnalyzer
import networkx as nx

# Create network
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D')])

# Analyze
analyzer = NetworkAnalyzer(G)
summary = analyzer.get_network_summary()

print(f"Nodes: {summary['nodes']}")
print(f"Edges: {summary['edges']}")
print(f"Hub nodes: {summary['hub_nodes']}")
```

### Simulation Analysis

```python
from src.analysis.simulation_analysis import SimulationAnalyzer
import numpy as np

analyzer = SimulationAnalyzer()

# Load time series
time_series = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.randn(1000)*0.1
analyzer.load_time_series("Oscillation", time_series.tolist())

# Calculate oscillation parameters
params = analyzer.calculate_oscillation_parameters(time_series)
print(f"Oscillatory: {params['oscillatory']}")
print(f"Period: {params.get('period', 'N/A')}")
```

---

## Visualization

```python
from src.visualization.plots import (
    plot_time_series,
    plot_network,
    plot_population_dynamics
)

# Plot time series
time_points = list(range(100))
data = {
    'Glucose': [10 * np.exp(-0.1 * t) for t in time_points],
    'Pyruvate': [0.5 * (1 - np.exp(-0.1 * t)) for t in time_points]
}

plot_time_series(time_points, data, title="Metabolite Dynamics", 
                save_path="metabolites.png")

# Plot population dynamics
plot_population_dynamics(time_points, 
                        {'Wild-type': [100*np.exp(0.5*t) for t in time_points],
                         'Mutant': [100*np.exp(0.3*t) for t in time_points]},
                        title="Population Competition",
                        save_path="population.png")
```

---

## Examples

### Complete Example: Lac Operon Simulation

```python
from src.core.cell import Cell, CellType
from src.core.genome import Genome, Gene, Operon, Promoter, PromoterType
from src.simulations.gene_regulation import GeneRegulatoryNetwork
from src.simulations.metabolism import MetabolicSimulation

# Create cell with lac operon
cell = Cell(name="E. coli Lac", cell_type=CellType.BACTERIUM)

# Create lac operon
promoter = Promoter(
    name="pLac",
    sequence=DNA(""),
    promoter_type=PromoterType.CONSTITUTIVE,
    basal_activity=0.1
)

operon = Operon(name="Lac Operon")
operon.set_promoter(promoter)

# Add genes
for gene_name in ["lacI", "lacZ", "lacY", "lacA"]:
    gene = Gene(name=gene_name, sequence=DNA(""), start=0, end=1000)
    operon.add_gene(gene)

# Create genome
genome = Genome(name="E. coli")
genome.add_operon(operon)
cell.set_genome(genome)

# Simulate with/without lactose
for condition in ["glucose", "lactose"]:
    cell.state = CellState.GROWING
    cell.set_environment({
        'glucose': 10.0 if condition == 'glucose' else 0.0,
        'lactose': 5.0 if condition == 'lactose' else 0.0
    })
    
    for _ in range(100):
        cell.step(delta_time=1.0)
    
    print(f"{condition}: Expression = {cell.proteins.get_protein('lacZ'):.4f}")
```

---

## API Reference

### Core Modules

| Module | Description |
|--------|-------------|
| `src.core.sequence` | DNA, RNA, Protein sequence classes |
| `src.core.genome` | Gene, Genome, Operon classes |
| `src.core.cell` | Cell, Organism, Population classes |
| `src.core.biochemistry` | Molecule, Reaction, Network classes |

### Simulation Modules

| Module | Description |
|--------|-------------|
| `src.simulations.gene_regulation` | Gene regulatory network simulation |
| `src.simulations.metabolism` | Metabolic pathway simulation |
| `src.simulations.kinetics` | Biochemical kinetics |
| `src.simulations.signaling` | Cell signaling pathways |
| `src.simulations.population` | Population dynamics |
| `src.simulations.cell_division` | Cell division mechanics |

### Analysis Modules

| Module | Description |
|--------|-------------|
| `src.analysis.sequence_analysis` | Sequence analysis tools |
| `src.analysis.network_analysis` | Network analysis tools |
| `src.analysis.simulation_analysis` | Simulation result analysis |

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

### Development Setup

```bash
# Clone repository
git clone https://github.com/moggan1337/Synthia.git
cd Synthia

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use Synthia in your research, please cite:

```
Synthia: A Comprehensive Synthetic Biology Simulator
Computational Biology Framework v1.0
https://github.com/moggan1337/Synthia
```

---

## Acknowledgments

- The BioPython community for sequence handling inspiration
- Systems Biology community for metabolic network modeling concepts
- Synthetic Biology community for genetic circuit design patterns

## Contact

- **Project Lead**: moggan1337
- **GitHub**: https://github.com/moggan1337/Synthia
- **Email**: team@synthia.bio

---

## Roadmap

- [ ] Add stochastic simulation capabilities (Gillespie algorithm)
- [ ] Implement spatial simulation (reaction-diffusion)
- [ ] Add SBML import/export
- [ ] Implement cell-cell communication
- [ ] Add fluxomics analysis tools
- [ ] Implement proteomics simulation
- [ ] Add multi-scale modeling capabilities
- [ ] GPU acceleration for large networks

---

## FAQ

**Q: What is synthetic biology?**
A: Synthetic biology is an interdisciplinary field that applies engineering principles to biological systems, enabling the design and construction of new biological parts and systems.

**Q: What is computational biology?**
A: Computational biology uses mathematical models, computational simulation, and data analysis to understand biological systems.

**Q: How does Synthia compare to other simulators?**
A: Synthia provides a comprehensive Python-based framework with a focus on modularity and extensibility. While tools like COPASI and CellDesigner focus on specific aspects, Synthia aims to provide end-to-end capabilities from sequence analysis to population dynamics.

---

## Changelog

### v1.0.0 (2026-04-20)
- Initial release
- Core sequence classes (DNA, RNA, Protein)
- Gene and genome modeling
- Cell simulation framework
- Biochemical network modeling
- Gene regulatory network simulation
- Metabolic pathway simulation
- Cell signaling simulation
- Population dynamics
- Cell division mechanics
- Analysis and visualization tools

---

<div align="center">
  <strong>Synthia - Building the Future of Synthetic Biology</strong>
  <br>
  Made with ❤️ by the Synthia Team
</div>
