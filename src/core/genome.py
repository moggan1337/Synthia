"""
Genome and Gene classes for Synthia.
Handles genetic elements, regulatory regions, and genome structure.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from .sequence import DNA, RNA, Protein


class GeneType(Enum):
    """Types of genes."""
    CODING = "coding"
    NON_CODING = "non_coding"
    rRNA = "rRNA"
    tRNA = "tRNA"
    miRNA = "miRNA"
    siRNA = "siRNA"
    lncRNA = "lncRNA"
    PSEUDOGENE = "pseudogene"


class PromoterType(Enum):
    """Types of promoters."""
    CONSTITUTIVE = "constitutive"
    INDUCIBLE = "inducible"
    REPRESSIBLE = "repressible"
    ACTIVATABLE = "activatable"
    TATA_BOX = "tata_box"
    CpG = "cpg_island"
    ENHANCER = "enhancer"
    SILENCER = "silencer"


class RegulatoryElement:
    """
    Represents a regulatory DNA element (enhancer, silencer, insulator, etc.)
    """
    
    def __init__(self, name: str, element_type: str, sequence: DNA,
                 start: int, end: int, strand: str = '+'):
        self.name = name
        self.element_type = element_type
        self.sequence = sequence
        self.start = start
        self.end = end
        self.strand = strand
        self.bound_proteins: List[str] = []
        self.target_genes: List[str] = []
    
    def __repr__(self):
        return f"RegulatoryElement({self.name}, {self.element_type}, {self.start}-{self.end})"
    
    def get_sequence_string(self) -> str:
        """Get the regulatory sequence as string."""
        return str(self.sequence)


@dataclass
class Promoter:
    """
    Promoter element controlling gene transcription.
    """
    name: str
    sequence: DNA
    promoter_type: PromoterType = PromoterType.CONSTITUTIVE
    transcription_factor_binding_sites: Dict[str, Tuple[int, int, float]] = field(default_factory=dict)
    # {tf_name: (start, end, affinity)}
    
    # Core promoter elements
    has_tata_box: bool = False
    tata_box_position: Optional[int] = None
    has_inr: bool = False
    has_dpe: bool = False
    
    # Activity
    basal_activity: float = 0.1  # 0-1 scale
    max_activity: float = 1.0
    noise_level: float = 0.05
    
    # Response elements
    inducers: List[str] = field(default_factory=list)
    repressors: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Promoter({self.name}, type={self.promoter_type.value})"
    
    def find_tata_box(self, pattern: str = "TATAAA") -> Optional[int]:
        """Find TATA box in promoter sequence."""
        seq_str = str(self.sequence).upper()
        pos = seq_str.find(pattern)
        if pos >= 0:
            self.has_tata_box = True
            self.tata_box_position = pos
        return pos
    
    def add_tf_binding_site(self, tf_name: str, start: int, end: int, affinity: float = 1.0):
        """Add a transcription factor binding site."""
        self.transcription_factor_binding_sites[tf_name] = (start, end, affinity)
    
    def calculate_activity(self, transcription_factors: Dict[str, float],
                          inducers: List[str] = None,
                          repressors: List[str] = None) -> float:
        """
        Calculate promoter activity based on bound factors.
        
        Args:
            transcription_factors: Dict of {tf_name: concentration/activity}
            inducers: List of active inducers
            repressors: List of active repressors
            
        Returns:
            Promoter activity (0-1 scale)
        """
        activity = self.basal_activity
        
        # Add activation from bound TFs
        for tf_name, (start, end, affinity) in self.transcription_factor_binding_sites.items():
            tf_activity = transcription_factors.get(tf_name, 0.0)
            activity += affinity * tf_activity * (1 - activity)
        
        # Process inducers
        if inducers:
            for ind in self.inducers:
                if ind in inducers:
                    activity *= 1.5
                    activity = min(activity, self.max_activity)
        
        # Process repressors
        if repressors:
            for rep in self.repressors:
                if rep in repressors:
                    activity *= 0.3
        
        # Add noise
        import random
        noise = random.gauss(0, self.noise_level)
        activity += noise
        
        return max(0.0, min(1.0, activity))
    
    def get_consensus_sequence(self) -> str:
        """Get consensus sequence from binding sites."""
        if not self.transcription_factor_binding_sites:
            return ""
        
        sites = []
        for tf_name, (start, end, affinity) in self.transcription_factor_binding_sites.items():
            if end <= len(self.sequence):
                sites.append(str(self.sequence[start:end]))
        
        if not sites:
            return ""
        
        # Simple consensus calculation
        consensus = []
        min_len = min(len(s) for s in sites)
        for i in range(min_len):
            bases = [s[i] for s in sites]
            consensus.append(max(set(bases), key=bases.count))
        
        return ''.join(consensus)


@dataclass
class Terminator:
    """Transcription terminator."""
    name: str
    sequence: DNA
    termination_efficiency: float = 0.95  # 0-1 scale
    rho_dependent: bool = False
    
    def will_terminate(self) -> bool:
        """Determine if transcription will terminate at this site."""
        import random
        return random.random() < self.termination_efficiency


@dataclass
class RBS:
    """
    Ribosome Binding Site (Shine-Dalgarno sequence in bacteria).
    """
    name: str
    sequence: DNA
    ribosome_binding_strength: float = 0.5  # 0-1 scale
    spacer_length: int = 5
    accessibility: float = 1.0  # 0-1, reduced by secondary structure
    
    SD_SEQUENCE = "AGGAGG"
    
    def calculate_translation_initiation_rate(self) -> float:
        """Calculate translation initiation rate."""
        # Check for Shine-Dalgarno sequence
        seq_str = str(self.sequence).upper()
        has_sd = self.SD_SEQUENCE in seq_str
        
        if has_sd:
            base_rate = self.ribosome_binding_strength * 0.8
        else:
            base_rate = self.ribosome_binding_strength * 0.2
        
        return base_rate * self.accessibility
    
    def find_sd_sequence(self) -> Optional[int]:
        """Find Shine-Dalgarno sequence."""
        seq_str = str(self.sequence).upper()
        pos = seq_str.find(self.SD_SEQUENCE)
        return pos


@dataclass
class Gene:
    """
    Represents a gene with all its regulatory elements.
    """
    name: str
    gene_type: GeneType = GeneType.CODING
    sequence: DNA
    start: int
    end: int
    strand: str = '+'  # '+' or '-'
    
    # Gene components
    promoter: Optional[Promoter] = None
    rbs: Optional[RBS] = None
    terminator: Optional[Terminator] = None
    utr5: Optional[DNA] = None  # 5' UTR
    utr3: Optional[DNA] = None  # 3' UTR
    introns: List[DNA] = field(default_factory=list)
    
    # Coding sequence
    cds: Optional[DNA] = None  # Coding sequence (exons only)
    protein_sequence: Optional[Protein] = None
    
    # Regulatory
    regulatory_elements: List[RegulatoryElement] = field(default_factory=list)
    transcription_factors: List[str] = field(default_factory=list)
    operon: Optional[str] = None
    
    # Expression control
    expression_level: float = 0.0
    transcriptional_regulation: float = 1.0
    translational_regulation: float = 1.0
    protein_stability: float = 1.0  # half-life factor
    
    # Annotations
    gene_id: Optional[str] = None
    product: Optional[str] = None
    function: Optional[str] = None
    ontology_terms: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Gene({self.name}, type={self.gene_type.value}, {self.start}-{self.end})"
    
    def get_length(self) -> int:
        """Get gene length in bp."""
        return self.end - self.start
    
    def transcribe(self) -> RNA:
        """Transcribe gene to RNA."""
        if self.strand == '+':
            return self.sequence.transcribe()
        else:
            return self.sequence.reverse_complement().transcribe()
    
    def translate(self, rna: RNA = None) -> Protein:
        """Translate gene to protein."""
        if rna is None:
            rna = self.transcribe()
        
        protein = rna.translate()
        self.protein_sequence = protein
        return protein
    
    def get_coding_sequence(self) -> Optional[DNA]:
        """Get the coding sequence (without introns)."""
        if self.cds:
            return self.cds
        
        # Extract CDS from main sequence if no introns
        if not self.introns:
            return self.sequence[self.start:self.end]
        
        return None
    
    def calculate_expression(self, transcription_factors: Dict[str, float] = None,
                            inducers: List[str] = None,
                            repressors: List[str] = None) -> float:
        """
        Calculate gene expression level.
        
        Returns:
            Protein production rate
        """
        # Transcriptional regulation
        if self.promoter and transcription_factors:
            transcription = self.promoter.calculate_activity(
                transcription_factors, inducers, repressors
            )
        else:
            transcription = self.promoter.basal_activity if self.promoter else 0.1
        
        transcription *= self.transcriptional_regulation
        
        # Translational regulation
        if self.rbs:
            translation = self.rbs.calculate_translation_initiation_rate()
        else:
            translation = 0.5
        
        translation *= self.translational_regulation
        
        # Combined expression
        self.expression_level = transcription * translation * self.protein_stability
        
        return self.expression_level
    
    def add_regulatory_element(self, element: RegulatoryElement):
        """Add a regulatory element to the gene."""
        self.regulatory_elements.append(element)
    
    def get_regulatory_domains(self, window: int = 1000) -> List[Tuple[int, int, str]]:
        """
        Get regulatory domains up/downstream of gene.
        
        Args:
            window: Window size in bp
            
        Returns:
            List of (start, end, type) tuples
        """
        domains = []
        
        if self.strand == '+':
            # Upstream regulatory region
            domains.append((max(0, self.start - window), self.start, 'upstream'))
            # Downstream region
            domains.append((self.end, self.end + window, 'downstream'))
        else:
            # For reverse strand
            domains.append((self.end, min(len(str(self.sequence)), self.end + window), 'upstream'))
            domains.append((max(0, self.start - window), self.start, 'downstream'))
        
        return domains
    
    def is_expressed(self, threshold: float = 0.01) -> bool:
        """Check if gene is expressed above threshold."""
        return self.expression_level > threshold
    
    def get_promoter_strength(self) -> float:
        """Get normalized promoter strength."""
        if self.promoter:
            return self.promoter.max_activity
        return 0.0
    
    def find_binding_sites(self, motif: str) -> List[int]:
        """Find transcription factor binding sites."""
        return self.sequence.find_motif(motif)
    
    def annotate_ORFs(self, min_length: int = 100) -> List[Dict]:
        """Find open reading frames within gene."""
        return self.sequence.find_orfs(min_length)


class Operon:
    """
    Bacterial operon - group of genes transcribed together.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.genes: List[Gene] = []
        self.promoter: Optional[Promoter] = None
        self.terminator: Optional[Terminator] = None
        self.operator: Optional[DNA] = None  # Operator site for repressor binding
        self.regulators: List[str] = []  # Names of regulator proteins
        
    def add_gene(self, gene: Gene):
        """Add gene to operon."""
        gene.operon = self.name
        self.genes.append(gene)
    
    def set_promoter(self, promoter: Promoter):
        """Set operon promoter."""
        self.promoter = promoter
        for gene in self.genes:
            gene.promoter = promoter
    
    def get_polycistronic_rna(self) -> RNA:
        """Get the polycistronic mRNA transcript."""
        sequences = [str(gene.sequence) for gene in self.genes]
        combined = ''.join(sequences)
        return RNA(combined)
    
    def calculate_operon_expression(self, 
                                    transcription_factors: Dict[str, float] = None,
                                    inducers: List[str] = None,
                                    repressors: List[str] = None) -> Dict[Gene, float]:
        """
        Calculate expression for all genes in operon.
        """
        expression = {}
        
        if self.promoter:
            promoter_activity = self.promoter.calculate_activity(
                transcription_factors, inducers, repressors
            )
        else:
            promoter_activity = 0.1
        
        for gene in self.genes:
            expression[gene] = promoter_activity * gene.translational_regulation
        
        return expression


class Genome:
    """
    Complete genome with genes, regulatory elements, and chromosome structure.
    """
    
    def __init__(self, name: str, species: str = "", total_length: int = 0):
        self.name = name
        self.species = species
        self.total_length = total_length
        
        self.sequence: Optional[DNA] = None
        self.genes: List[Gene] = []
        self.operons: List[Operon] = []
        self.chromosomes: Dict[str, DNA] = {}
        
        # Index structures
        self._gene_index: Dict[str, Gene] = {}
        self._position_index: Dict[str, List[Gene]] = {}  # chromosome -> sorted genes
        
        # Metadata
        self.gc_content: float = 0.0
        self.gene_density: float = 0.0
        
        # Annotations
        self.annotation_version: str = "1.0"
        self.organism: Optional[str] = None
        self.taxonomy: List[str] = []
    
    def add_gene(self, gene: Gene):
        """Add gene to genome."""
        self.genes.append(gene)
        self._gene_index[gene.name] = gene
        
        # Update position index
        chrom = getattr(gene, 'chromosome', 'main')
        if chrom not in self._position_index:
            self._position_index[chrom] = []
        self._position_index[chrom].append(gene)
        self._position_index[chrom].sort(key=lambda g: g.start)
    
    def add_operon(self, operon: Operon):
        """Add operon to genome."""
        self.operons.append(operon)
        for gene in operon.genes:
            self.add_gene(gene)
    
    def add_chromosome(self, chrom_id: str, sequence: DNA):
        """Add a chromosome."""
        self.chromosomes[chrom_id] = sequence
        if self.sequence is None:
            self.sequence = sequence
    
    def get_gene_by_name(self, name: str) -> Optional[Gene]:
        """Get gene by name."""
        return self._gene_index.get(name)
    
    def get_genes_in_region(self, chrom: str, start: int, end: int) -> List[Gene]:
        """Get all genes in a genomic region."""
        genes_in_region = []
        for gene in self.genes:
            if gene.start >= start and gene.end <= end:
                genes_in_region.append(gene)
        return genes_in_region
    
    def get_genes_upstream(self, gene: Gene, window: int = 5000) -> List[Gene]:
        """Get genes upstream of given gene."""
        upstream_genes = []
        for g in self.genes:
            if g != gene:
                if gene.strand == '+':
                    if g.end < gene.start and gene.start - g.end < window:
                        upstream_genes.append(g)
                else:
                    if g.start > gene.end and g.start - gene.end < window:
                        upstream_genes.append(g)
        return upstream_genes
    
    def calculate_gc_content(self):
        """Calculate genome GC content."""
        if self.sequence:
            self.gc_content = self.sequence.gc_content()
    
    def calculate_gene_density(self):
        """Calculate gene density (genes per kb)."""
        if self.total_length > 0:
            self.gene_density = len(self.genes) / (self.total_length / 1000)
    
    def get_genome_stats(self) -> Dict:
        """Get comprehensive genome statistics."""
        stats = {
            'total_genes': len(self.genes),
            'total_operons': len(self.operons),
            'total_chromosomes': len(self.chromosomes),
            'total_length': self.total_length,
            'gc_content': self.gc_content,
            'gene_density': self.gene_density,
        }
        
        # Gene type distribution
        type_counts = {}
        for gene in self.genes:
            type_counts[gene.gene_type.value] = type_counts.get(gene.gene_type.value, 0) + 1
        stats['gene_type_distribution'] = type_counts
        
        # Expression statistics
        expressed = sum(1 for g in self.genes if g.is_expressed())
        stats['expressed_genes'] = expressed
        stats['expression_ratio'] = expressed / len(self.genes) if self.genes else 0
        
        return stats
    
    def to_genbank(self) -> str:
        """Export genome to GenBank format (simplified)."""
        lines = []
        lines.append(f"LOCUS       {self.name}")
        lines.append(f"DEFINITION  {self.species} genome, {len(self.genes)} genes.")
        lines.append(f"ACCESSION   {self.name}")
        lines.append(f"VERSION     {self.name}.1")
        lines.append("FEATURES             Location/Qualifiers")
        
        for gene in sorted(self.genes, key=lambda g: g.start):
            lines.append(f"     gene            {gene.start}..{gene.end}")
            lines.append(f"                     /gene=\"{gene.name}\"")
            if gene.product:
                lines.append(f"                     /product=\"{gene.product}\"")
            if gene.gene_id:
                lines.append(f"                     /locus_tag=\"{gene.gene_id}\"")
            
            if gene.gene_type == GeneType.CODING:
                lines.append(f"     CDS             {gene.start}..{gene.end}")
                lines.append(f"                     /gene=\"{gene.name}\"")
                if gene.product:
                    lines.append(f"                     /product=\"{gene.product}\"")
        
        return '\n'.join(lines)
    
    def find_orthologs(self, other_genome: 'Genome') -> Dict[Gene, Gene]:
        """Find orthologous gene pairs with another genome."""
        orthologs = {}
        
        for gene in self.genes:
            if gene.protein_sequence:
                best_match = None
                best_identity = 0.0
                
                for other_gene in other_genome.genes:
                    if other_gene.protein_sequence:
                        identity = self._calculate_sequence_identity(
                            str(gene.protein_sequence),
                            str(other_gene.protein_sequence)
                        )
                        if identity > best_identity and identity > 0.3:
                            best_identity = identity
                            best_match = other_gene
                
                if best_match:
                    orthologs[gene] = best_match
        
        return orthologs
    
    @staticmethod
    def _calculate_sequence_identity(seq1: str, seq2: str) -> float:
        """Calculate simple sequence identity."""
        if len(seq1) != len(seq2) or len(seq1) == 0:
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def find_motif_conservation(self, motif: str) -> Dict[str, List[int]]:
        """Find conservation of motifs across chromosomes."""
        conservation = {}
        for chrom_id, chrom in self.chromosomes.items():
            positions = chrom.find_motif(motif)
            conservation[chrom_id] = positions
        return conservation
    
    def identify_crispr_arrays(self, repeat_length: int = 28,
                               min_spacers: int = 3) -> List[Dict]:
        """Identify CRISPR arrays (simplified detection)."""
        arrays = []
        
        # Simplified CRISPR detection
        # Real implementation would use more sophisticated pattern matching
        
        return arrays
