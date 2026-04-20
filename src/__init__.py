"""
Synthia - Synthetic Biology Simulator
A comprehensive framework for computational biology and synthetic biology research.
"""

__version__ = "1.0.0"
__author__ = "Synthia Team"

from .core.sequence import DNA, RNA, Protein
from .core.cell import Cell
from .core.genome import Genome
from .simulations.gene_regulation import Gene RegulatoryNetwork
from .simulations.metabolism import MetabolicPathway
from .simulations.kinetics import KineticsEngine

__all__ = [
    "DNA", "RNA", "Protein",
    "Cell", "Genome",
    "Gene RegulatoryNetwork",
    "MetabolicPathway",
    "KineticsEngine",
]
