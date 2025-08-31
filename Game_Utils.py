"""
Shared utilities for the Liar's Dice NEAT experiment.
Contains common functions used across multiple files.
"""
import math
import neat
import pygame
import pandas as pd
import numpy as np
from typing import List, Tuple, Any, Optional


def probability_r(required: int, n: int = 5, p: float = 1/6) -> float:
    """
    Calculate probability of getting at least 'required' successes in 'n' trials 
    with success probability 'p' (binomial distribution).
    
    Args:
        required: Minimum number of successes needed
        n: Number of trials (dice)  
        p: Probability of success per trial (1/6 for standard dice)
    
    Returns:
        Probability as float between 0 and 1
    """
    probability = 0
    if required < 0:
        probability = 1
    else:
        for k in range(required, n + 1):
            probability += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return probability


def test_best_network(best_net: neat.nn.FeedForwardNetwork, dice_game_class, n_dice: int = 5) -> int:
    """
    Test a neural network against probability strategy over multiple games.
    
    Args:
        best_net: Trained neural network to test
        dice_game_class: DiceGame class to use for testing (injected dependency)
        n_dice: Number of dice per player (default 5, can be 3 for other variants)
    
    Returns:
        Number of wins for the neural network
    """
    from Game_Constants import TEST_GAMES_TOTAL, GAMES_PER_STARTING_POSITION, WINDOW_WIDTH, WINDOW_HEIGHT_TEST
    
    pygame_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT_TEST))
    pygame.display.set_caption("Test AI")
    
    net_wins = 0
    for i in range(TEST_GAMES_TOTAL):
        dice_game = dice_game_class(pygame_window, WINDOW_WIDTH, WINDOW_HEIGHT_TEST)
        
        # Alternate starting player
        starting_player = 1 if i < GAMES_PER_STARTING_POSITION else 2
        result = dice_game.arena(best_net, starting_player=starting_player)
        
        if result == 1:
            net_wins += 1

    print(f"Test results: {net_wins}/{TEST_GAMES_TOTAL} wins")
    return net_wins


class MetricsReporter(neat.reporting.BaseReporter):
    """
    Reporter class to track training metrics across generations.
    """
    def __init__(self) -> None:
        self.generations: List[int] = []
        self.highest_fitnesses: List[float] = []
        self.average_fitnesses: List[float] = []
        self.std_deviations: List[float] = []
        self.num_species: List[int] = []
        self.prev_champion_rounds: List[int] = []
        self.generation_count: int = 0

    def post_evaluate(self, config: neat.Config, population: dict, species: Any, best_genome: neat.DefaultGenome) -> None:
        # Compute highest fitness
        highest_fitness = (
            best_genome.fitness if best_genome.fitness is not None else float("-inf")
        )

        # Compute average fitness of the current population
        fitnesses = [
            genome.fitness
            for genome in population.values()
            if genome.fitness is not None
        ]
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0

        # Compute standard deviation of fitnesses
        std_dev = pd.Series(fitnesses).std() if fitnesses else 0

        # Count the number of species
        num_species = len(species.species)

        # Retrieve the round the previous champion made it to
        # Note: This will need to be passed in or accessed differently to avoid circular imports
        prev_champion_round = 0  # Placeholder - should be updated by caller

        # Append the metrics to the corresponding lists
        self.generations.append(self.generation_count)
        self.highest_fitnesses.append(highest_fitness)
        self.average_fitnesses.append(avg_fitness)
        self.std_deviations.append(std_dev)
        self.num_species.append(num_species)
        self.prev_champion_rounds.append(prev_champion_round)

        self.generation_count += 1


class BestNetworkTestReporter(neat.reporting.BaseReporter):
    """
    Reporter class to periodically test and save the best network.
    """
    def __init__(self, config: neat.Config, dice_game_class, test_interval: int = 10, file_prefix: str = "best_network") -> None:
        self.generation = 0
        self.config = config
        self.dice_game_class = dice_game_class
        self.test_results: List[Tuple[int, int]] = []
        self.test_interval = test_interval
        self.file_prefix = file_prefix

    def post_evaluate(self, config: neat.Config, population: dict, species: Any, best_genome: neat.DefaultGenome) -> None:
        self.generation += 1
        if self.generation % self.test_interval == 0:
            print(f"Testing best network at generation {self.generation}...")
            best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            score = test_best_network(best_net, self.dice_game_class)
            self.test_results.append((self.generation, score))

            filename = f"{self.file_prefix}_gen_{self.generation}.pickle"
            import pickle
            with open(filename, "wb") as f:
                pickle.dump(best_net, f)
            print(f"Saved best network for generation {self.generation} to {filename}")