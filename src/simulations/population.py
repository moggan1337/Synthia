"""
Population dynamics and evolutionary simulation for Synthia.
Models cell populations, evolutionary dynamics, and genetic circuits.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import random
import networkx as nx


class GrowthModel(Enum):
    """Population growth models."""
    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"
    GOMPERTZ = "gompertz"
    RICHARDS = "richards"
    BARanyi = "baranyi"


@dataclass
class Individual:
    """
    Individual in a population with genetic information.
    """
    id: int
    genome: Dict[str, any]  # Genetic parameters
    fitness: float = 1.0
    age: float = 0.0
    generation: int = 0
    
    # Phenotype
    traits: Dict[str, float] = field(default_factory=dict)
    
    def mutate(self, mutation_rate: float = 0.001):
        """Mutate genome."""
        for key in self.genome:
            if random.random() < mutation_rate:
                # Gaussian mutation
                self.genome[key] *= random.gauss(1.0, 0.1)
    
    def reproduce(self, offspring_id: int) -> 'Individual':
        """Create offspring with inheritance."""
        offspring = Individual(
            id=offspring_id,
            genome=self.genome.copy(),
            generation=self.generation + 1
        )
        
        # Inherit traits
        offspring.traits = self.traits.copy()
        
        # Calculate fitness
        offspring.fitness = self._calculate_fitness()
        
        return offspring
    
    def _calculate_fitness(self) -> float:
        """Calculate fitness from traits."""
        base = 1.0
        for trait, value in self.traits.items():
            base *= (1 + value * 0.1)
        return base


@dataclass
class Population:
    """
    Population of individuals.
    """
    name: str
    individuals: List[Individual] = field(default_factory=list)
    
    # Parameters
    carrying_capacity: float = 10000
    mutation_rate: float = 0.001
    crossover_rate: float = 0.7
    
    # Statistics
    generation: int = 0
    history: List[Dict] = []
    
    @property
    def size(self) -> int:
        return len(self.individuals)
    
    @property
    def avg_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return np.mean([ind.fitness for ind in self.individuals])
    
    @property
    def max_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return max(ind.fitness for ind in self.individuals)
    
    def add_individual(self, individual: Individual):
        """Add individual to population."""
        self.individuals.append(individual)
    
    def remove_individual(self, individual: Individual):
        """Remove individual from population."""
        if individual in self.individuals:
            self.individuals.remove(individual)
    
    def selection(self, method: str = "tournament", k: int = 3):
        """Select individuals for reproduction."""
        if method == "tournament":
            return self._tournament_selection(k)
        elif method == "roulette":
            return self._roulette_selection()
        elif method == "rank":
            return self._rank_selection()
        return self.individuals[0]
    
    def _tournament_selection(self, k: int = 3) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.individuals, min(k, len(self.individuals)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _roulette_selection(self) -> Individual:
        """Roulette wheel selection."""
        total_fitness = sum(ind.fitness for ind in self.individuals)
        if total_fitness == 0:
            return random.choice(self.individuals)
        
        pick = random.uniform(0, total_fitness)
        cumulative = 0
        for ind in self.individuals:
            cumulative += ind.fitness
            if cumulative >= pick:
                return ind
        return self.individuals[-1]
    
    def _rank_selection(self) -> Individual:
        """Rank selection."""
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness)
        n = len(sorted_inds)
        ranks = list(range(1, n + 1))
        total_rank = sum(ranks)
        
        pick = random.uniform(0, total_rank)
        cumulative = 0
        for ind, rank in zip(sorted_inds, ranks):
            cumulative += rank
            if cumulative >= pick:
                return ind
        return sorted_inds[-1]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover two individuals."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1_genome = {}
        child2_genome = {}
        
        for key in parent1.genome:
            if key in parent2.genome:
                if random.random() < 0.5:
                    child1_genome[key] = parent1.genome[key]
                    child2_genome[key] = parent2.genome[key]
                else:
                    child1_genome[key] = parent2.genome[key]
                    child2_genome[key] = parent1.genome[key]
        
        child1 = Individual(id=0, genome=child1_genome, generation=self.generation + 1)
        child2 = Individual(id=0, genome=child2_genome, generation=self.generation + 1)
        
        return child1, child2
    
    def evolve_one_generation(self) -> int:
        """Evolve population for one generation."""
        new_individuals = []
        next_id = max((ind.id for ind in self.individuals), default=0) + 1
        
        while len(new_individuals) < len(self.individuals):
            # Select parents
            parent1 = self.selection()
            parent2 = self.selection()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            
            # Set IDs
            child1.id = next_id
            next_id += 1
            child2.id = next_id
            next_id += 1
            
            # Calculate fitness
            child1.fitness = child1._calculate_fitness()
            child2.fitness = child2._calculate_fitness()
            
            new_individuals.extend([child1, child2])
        
        # Apply carrying capacity
        if len(new_individuals) > self.carrying_capacity:
            # Fitness-proportionate culling
            new_individuals.sort(key=lambda x: x.fitness, reverse=True)
            new_individuals = new_individuals[:int(self.carrying_capacity)]
        
        self.individuals = new_individuals
        self.generation += 1
        
        # Record statistics
        self.history.append({
            'generation': self.generation,
            'size': len(self.individuals),
            'avg_fitness': self.avg_fitness,
            'max_fitness': self.max_fitness
        })
        
        return len(new_individuals)
    
    def get_statistics(self) -> Dict:
        """Get population statistics."""
        if not self.individuals:
            return {}
        
        fitness_values = [ind.fitness for ind in self.individuals]
        
        return {
            'size': len(self.individuals),
            'avg_fitness': np.mean(fitness_values),
            'max_fitness': max(fitness_values),
            'min_fitness': min(fitness_values),
            'fitness_std': np.std(fitness_values),
            'generation': self.generation,
            'diversity': self._calculate_diversity()
        }
    
    def _calculate_diversity(self) -> float:
        """Calculate genetic diversity."""
        if len(self.individuals) < 2:
            return 0.0
        
        traits = list(self.individuals[0].traits.keys())
        if not traits:
            return 0.0
        
        diversities = []
        for trait in traits:
            values = [ind.traits.get(trait, 0) for ind in self.individuals]
            diversities.append(np.std(values))
        
        return np.mean(diversities)


class PopulationDynamics:
    """
    Population dynamics with various growth models.
    """
    
    def __init__(self, name: str = "Population"):
        self.name = name
        self.model = GrowthModel.EXPONENTIAL
        
        # Population state
        self.population: List[float] = []
        self.time_points: List[float] = []
        
        # Parameters
        self.initial_population: float = 100
        self.current_population: float = 100
        self.growth_rate: float = 0.5  # per hour
        self.carrying_capacity: float = 10000
        
        # Gompertz parameters
        self.gompertz_b: float = 1.0
        self.gompertz_c: float = 1.0
        
        # Time
        self.time: float = 0.0
    
    def set_model(self, model: GrowthModel):
        """Set growth model."""
        self.model = model
    
    def exponential_growth(self, dt: float) -> float:
        """Calculate exponential growth."""
        return self.current_population * np.exp(self.growth_rate * dt)
    
    def logistic_growth(self, dt: float) -> float:
        """Calculate logistic growth."""
        K = self.carrying_capacity
        N = self.current_population
        r = self.growth_rate
        
        dN = r * N * (1 - N / K)
        return N + dN * dt
    
    def gompertz_growth(self, dt: float) -> float:
        """Calculate Gompertz growth."""
        K = self.carrying_capacity
        N = self.current_population
        b = self.gompertz_b
        c = self.gompertz_c
        
        dN = N * c * np.log(K / N) * np.exp(-b * self.time)
        return N + dN * dt
    
    def simulate(self, duration: float, dt: float = 0.1):
        """
        Simulate population growth.
        
        Args:
            duration: Duration in hours
            dt: Time step
        """
        self.population = [self.initial_population]
        self.time_points = [0.0]
        self.current_population = self.initial_population
        self.time = 0.0
        
        steps = int(duration / dt)
        
        for _ in range(steps):
            if self.model == GrowthModel.EXPONENTIAL:
                self.current_population = self.exponential_growth(dt)
            elif self.model == GrowthModel.LOGISTIC:
                self.current_population = self.logistic_growth(dt)
            elif self.model == GrowthModel.GOMPERTZ:
                self.current_population = self.gompertz_growth(dt)
            
            # Ensure non-negative
            self.current_population = max(0, self.current_population)
            
            self.population.append(self.current_population)
            self.time_points.append(self.time)
            self.time += dt
    
    def lotka_volterra(self, prey: float, predator: float, 
                     alpha: float, beta: float, gamma: float, delta: float) -> Tuple[float, float]:
        """
        Lotka-Volterra predator-prey dynamics.
        
        Returns:
            (dPrey/dt, dPredator/dt)
        """
        d_prey = alpha * prey - beta * prey * predator
        d_predator = delta * prey * predator - gamma * predator
        return d_prey, d_predator
    
    def simulate_predator_prey(self, duration: float, dt: float = 0.01,
                             prey0: float = 100, predator0: float = 10):
        """
        Simulate predator-prey dynamics.
        """
        prey_pop = [prey0]
        predator_pop = [predator0]
        times = [0.0]
        
        prey = prey0
        predator = predator0
        t = 0.0
        
        # Lotka-Volterra parameters
        alpha = 1.0  # Prey growth rate
        beta = 0.1   # Predation rate
        gamma = 1.0  # Predator death rate
        delta = 0.1  # Predator reproduction rate
        
        steps = int(duration / dt)
        
        for _ in range(steps):
            d_prey, d_predator = self.lotka_volterra(prey, predator, 
                                                      alpha, beta, gamma, delta)
            
            prey = max(0, prey + d_prey * dt)
            predator = max(0, predator + d_predator * dt)
            
            prey_pop.append(prey)
            predator_pop.append(predator)
            times.append(t)
            t += dt
        
        return {
            'time': times,
            'prey': prey_pop,
            'predator': predator_pop
        }


class EvolutionarySimulation:
    """
    Evolutionary simulation with genetic algorithms and circuits.
    """
    
    def __init__(self, name: str = "Evolution"):
        self.name = name
        self.population = None
        
        # Genetic circuit parameters
        self.circuit_parameters: Dict[str, Tuple[float, float]] = {}
        
        # Optimization target
        self.target_function: Optional[Callable] = None
        
        # Evolution history
        self.best_individuals: List[Dict] = []
    
    def define_circuit(self, parameters: Dict[str, Tuple[float, float]]):
        """
        Define genetic circuit parameters with bounds.
        
        Args:
            parameters: {name: (min, max)} bounds
        """
        self.circuit_parameters = parameters
    
    def fitness_function(self, genome: Dict[str, float]) -> float:
        """
        Evaluate fitness of a genome.
        Override this for specific optimization problems.
        """
        if self.target_function:
            return self.target_function(genome)
        
        # Default: maximize all parameters
        return sum(genome.values())
    
    def create_initial_population(self, size: int = 100):
        """Create initial random population."""
        individuals = []
        
        for i in range(size):
            genome = {}
            for param, (min_val, max_val) in self.circuit_parameters.items():
                genome[param] = random.uniform(min_val, max_val)
            
            ind = Individual(id=i, genome=genome)
            ind.fitness = self.fitness_function(genome)
            individuals.append(ind)
        
        self.population = Population(
            name="evolution_pop",
            individuals=individuals,
            carrying_capacity=size
        )
    
    def evolve(self, generations: int = 100, verbose: bool = True):
        """
        Evolve population.
        
        Args:
            generations: Number of generations
            verbose: Print progress
        """
        if not self.population:
            self.create_initial_population()
        
        for g in range(generations):
            # Evolve one generation
            self.population.evolve_one_generation()
            
            # Update fitness
            for ind in self.population.individuals:
                ind.fitness = self.fitness_function(ind.genome)
            
            # Record best
            best = max(self.population.individuals, key=lambda x: x.fitness)
            self.best_individuals.append({
                'generation': g,
                'fitness': best.fitness,
                'genome': best.genome.copy()
            })
            
            if verbose and g % 10 == 0:
                stats = self.population.get_statistics()
                print(f"Gen {g}: Best={best.fitness:.4f}, Avg={stats['avg_fitness']:.4f}")
    
    def get_best_individual(self) -> Individual:
        """Get the best individual."""
        if not self.best_individuals:
            return None
        best = max(self.best_individuals, key=lambda x: x['fitness'])
        return Individual(id=0, genome=best['genome'], fitness=best['fitness'])
    
    def design_synthetic_oscillator(self) -> Dict:
        """
        Design synthetic oscillator using evolutionary algorithm.
        """
        # Define oscillator parameters
        self.define_circuit({
            'activation_strength': (0.1, 2.0),
            'repression_strength': (0.1, 2.0),
            'degradation_rate': (0.01, 0.2),
            'hill_coefficient': (1.0, 4.0),
            'delay_time': (0.1, 1.0)
        })
        
        # Target: sinusoidal oscillation
        def oscillator_fitness(genome):
            # Simulate simple oscillator dynamics
            T = 10  # Period
            t_points = np.linspace(0, T, 100)
            
            x = [1.0]  # Initial
            for t in t_points[1:]:
                dx = (genome['activation_strength'] * 
                     np.sin(2 * np.pi * t / genome['delay_time']) -
                     genome['repression_strength'] * x[-1])
                x.append(x[-1] + dx * 0.1)
            
            # Fitness: maximize amplitude and regularity
            x = np.array(x)
            amplitude = np.max(x) - np.min(x)
            regularity = 1.0 / (1.0 + np.std(np.diff(x)))
            
            return amplitude * regularity
        
        self.target_function = oscillator_fitness
        self.create_initial_population(50)
        self.evolve(50, verbose=False)
        
        return self.get_best_individual().genome
    
    def design_synthetic_toggle_switch(self) -> Dict:
        """
        Design synthetic toggle switch.
        """
        self.define_circuit({
            'repression_a_to_b': (0.1, 3.0),
            'repression_b_to_a': (0.1, 3.0),
            'basal_expression': (0.01, 0.1),
            'cooperativity': (1.0, 4.0)
        })
        
        def toggle_fitness(genome):
            # Simulate toggle dynamics
            A, B = 1.0, 0.0  # Initial state
            
            for _ in range(100):
                # Toggle equations
                dA = (genome['basal_expression'] + 
                     genome['repression_a_to_b'] / (1 + B ** genome['cooperativity']))
                dB = (genome['basal_expression'] + 
                     genome['repression_b_to_a'] / (1 + A ** genome['cooperativity']))
                
                A = max(0, A + (dA - A) * 0.1)
                B = max(0, B + (dB - B) * 0.1)
            
            # Fitness: bistability (A and B should be different)
            bistability = abs(A - B)
            
            # Also penalize both off
            if A < 0.1 and B < 0.1:
                bistability *= 0.1
            
            return bistability
        
        self.target_function = toggle_fitness
        self.create_initial_population(50)
        self.evolve(50, verbose=False)
        
        return self.get_best_individual().genome


class GeneticCircuitEvolution:
    """
    Evolve genetic circuits for specific functions.
    """
    
    def __init__(self):
        self.circuit_library: Dict[str, Dict] = {}
    
    def evolve_and_gate(self) -> Dict:
        """
        Evolve AND gate genetic circuit.
        """
        # AND gate: output only when both inputs are present
        def and_gate_fitness(genome):
            inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
            outputs = [0, 0, 0, 1]
            
            fitness = 0
            for (a, b), expected in zip(inputs, outputs):
                # Simple model
                output = genome['a_weight'] * a * genome['b_weight'] * b
                output = 1 if output > genome['threshold'] else 0
                
                if output == expected:
                    fitness += 1
            
            return fitness / len(inputs)
        
        return self._evolve_circuit(and_gate_fitness, ['a_weight', 'b_weight', 'threshold'])
    
    def evolve_or_gate(self) -> Dict:
        """Evolve OR gate genetic circuit."""
        def or_gate_fitness(genome):
            inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
            outputs = [0, 1, 1, 1]
            
            fitness = 0
            for (a, b), expected in zip(inputs, outputs):
                output = genome['a_weight'] * a + genome['b_weight'] * b
                output = 1 if output > genome['threshold'] else 0
                
                if output == expected:
                    fitness += 1
            
            return fitness / len(inputs)
        
        return self._evolve_circuit(or_gate_fitness, ['a_weight', 'b_weight', 'threshold'])
    
    def _evolve_circuit(self, fitness_fn: Callable, params: List[str]) -> Dict:
        """Generic circuit evolution."""
        bounds = {p: (0.1, 2.0) for p in params}
        bounds['threshold'] = (0.1, 1.5)
        
        # Simple evolution
        best_genome = None
        best_fitness = 0
        
        for _ in range(100):
            genome = {p: random.uniform(*bounds[p]) for p in params}
            fitness = fitness_fn(genome)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome
        
        return best_genome
