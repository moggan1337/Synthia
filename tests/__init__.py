"""
Tests for Synthia.
"""

import pytest
from src.core.sequence import DNA, RNA, Protein


class TestSequence:
    """Test biological sequence classes."""
    
    def test_dna_creation(self):
        dna = DNA("ATCGATCG")
        assert len(dna) == 8
        assert str(dna) == "ATCGATCG"
    
    def test_dna_complement(self):
        dna = DNA("ATCG")
        comp = dna.complement()
        assert str(comp) == "TAGC"
    
    def test_dna_transcribe(self):
        dna = DNA("ATCG")
        rna = dna.transcribe()
        assert str(rna) == "AUCG"
    
    def test_dna_gc_content(self):
        dna = DNA("GCGCGCGC")
        assert dna.gc_content() == 100.0
        
        dna2 = DNA("ATATATAT")
        assert dna2.gc_content() == 0.0
    
    def test_rna_translate(self):
        rna = RNA("AUG")
        protein = rna.translate()
        assert str(protein) == "M"
    
    def test_protein_properties(self):
        protein = Protein("MVLSPADKT")
        assert protein.molecular_weight > 0
        
        pI = protein.isoelectric_point
        assert 0 < pI < 14


class TestBiochemistry:
    """Test biochemical components."""
    
    def test_molecule_creation(self):
        from src.core.biochemistry import Molecule
        mol = Molecule("glucose", formula="C6H12O6")
        assert mol.name == "glucose"
        assert mol.molar_mass > 0
    
    def test_reaction_rate(self):
        from src.core.biochemistry import Molecule, Reaction
        
        A = Molecule("A")
        B = Molecule("B")
        
        rxn = Reaction(
            name="test",
            reactants={A: 1},
            products={B: 1},
            kinetic_type="mass_action",
            rate_constant=1.0
        )
        
        rate = rxn.calculate_rate({A: 1.0})
        assert rate == 1.0


class TestSimulations:
    """Test simulation modules."""
    
    def test_grn_creation(self):
        from src.simulations.gene_regulation import GeneRegulatoryNetwork
        
        grn = GeneRegulatoryNetwork(name="Test GRN")
        grn.add_gene("Gene1")
        grn.add_gene("Gene2")
        
        assert len(grn.genes) == 2
    
    def test_population_dynamics(self):
        from src.simulations.population import PopulationDynamics, GrowthModel
        
        pop = PopulationDynamics()
        pop.initial_population = 100
        pop.growth_rate = 0.5
        pop.set_model(GrowthModel.EXPONENTIAL)
        
        pop.simulate(duration=1.0, dt=0.1)
        
        assert len(pop.population) > 0
        assert pop.population[-1] > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
