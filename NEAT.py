from Liars_Dice_Game import Game
import glob
import pygame
import neat
import os
import pickle
import random
import numpy as np
from collections import Counter
import re
from typing import List, Tuple, Optional, Dict, Any
from Game_Utils import (
    probability_r,
    test_best_network,
    MetricsReporter,
    BestNetworkTestReporter,
)
from Game_Constants import *
from Champion_Tracker import ChampionTracker

# Initialize champion tracker for this experiment
champion_tracker = ChampionTracker()


class DiceGame:
    def __init__(
        self, window: pygame.Surface, window_width: int, window_height: int
    ) -> None:
        self.game = Game(window, window_width, window_height)

    def _make_network_decision(
        self, network: neat.nn.FeedForwardNetwork, is_player_1: bool
    ) -> None:
        """Shared logic for making network-based betting decisions.

        Args:
            network: Neural network to use for decision making
            is_player_1: True for player 1, False for player 2
        """
        # Get current game state based on player
        dice_values = self.game.p1_hand if is_player_1 else self.game.p2_hand
        d1, d2, d3 = dice_values
        random_input = random.uniform(RANDOM_INPUT_MIN, RANDOM_INPUT_MAX)
        bet_quantity, bet_value = (
            self.game.current_bet if self.game.current_bet else (0, 0)
        )
        previous_bet = self.game.bet_history[-1] if self.game.bet_history else (0, 0)
        previous_quantity, previous_value = previous_bet

        network_inputs = [
            d1,
            d2,
            d3,
            bet_quantity,
            bet_value,
            previous_quantity,
            previous_value,
            random_input,
            1,
        ]
        output = network.activate(network_inputs)

        # Convert network outputs to whole numbers within valid game ranges
        proposed_quantity = (
            round(output[0] * NETWORK_OUTPUT_SCALING) + BET_QUANTITY_OFFSET
        )
        proposed_value = round(output[1] * NETWORK_OUTPUT_SCALING) + BET_QUANTITY_OFFSET

        # Make game decision based on network output
        max_bet = (MAX_BET_QUANTITY_3DICE, MAX_DICE_VALUE)
        if self.game.current_bet == max_bet:
            self.game.call_bluff(is_player_1=is_player_1)
        elif self.game.current_bet is None:
            self.game.make_bet(
                proposed_quantity, proposed_value, is_player_1=is_player_1
            )
        elif output[2] > BLUFF_CALL_THRESHOLD:
            self.game.call_bluff(is_player_1=is_player_1)
        else:
            self.game.make_bet(
                proposed_quantity, proposed_value, is_player_1=is_player_1
            )

    # This method runs a game between the best network and the probability strategy.
    def arena(
        self,
        best_net: neat.nn.FeedForwardNetwork,
        draw: bool = False,
        show_both_hands: bool = True,
        starting_player: int = 1,
    ) -> int:
        run = True

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            if starting_player == 1:
                self.bet_or_bluff_p1(best_net)
                if self.game.round_over:
                    return self.game.round_winner
                self.probability_strat()
                if self.game.round_over:
                    return self.game.round_winner
            else:
                self.probability_strat()
                if self.game.round_over:
                    return self.game.round_winner
                self.bet_or_bluff_p1(best_net)
                if self.game.round_over:
                    return self.game.round_winner

            if draw:
                self.game.draw(show_both_hands)

            pygame.display.update()

    # This method runs a game between two AI networks for training.
    def train_ai(
        self,
        net1: neat.nn.FeedForwardNetwork,
        net2: neat.nn.FeedForwardNetwork,
        draw: bool = False,
        show_both_hands: bool = True,
        starting_player: int = 1,
    ) -> int:
        run = True

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            if starting_player == 1:
                self.bet_or_bluff_p1(net1)
                if self.game.round_over:
                    return self.game.round_winner
                self.bet_or_bluff_p2(net2)
                if self.game.round_over:
                    return self.game.round_winner
            else:
                self.bet_or_bluff_p2(net2)
                if self.game.round_over:
                    return self.game.round_winner
                self.bet_or_bluff_p1(net1)
                if self.game.round_over:
                    return self.game.round_winner

            if draw:
                self.game.draw(show_both_hands)

            pygame.display.update()

    # This method determines the game action for player 1 based on the neural network's output.
    def bet_or_bluff_p1(self, network: neat.nn.FeedForwardNetwork) -> None:
        """Make betting decision for player 1 using neural network output."""
        self._make_network_decision(network, is_player_1=True)

    # This method determines the game action for player 2 based on the neural network's output.
    def bet_or_bluff_p2(self, network: neat.nn.FeedForwardNetwork) -> None:
        """Make betting decision for player 2 using neural network output."""
        self._make_network_decision(network, is_player_1=False)

    def probability_strat(self) -> None:
        """Probability-based strategy for player 2 (opponent)."""
        player_hand = self.game.p2_hand
        current_bet = self.game.current_bet
        if current_bet:
            bet_quantity, bet_value = current_bet
        else:
            bet_quantity, bet_value = (0, 0)

        # Analyze hand composition
        dice_matching_bet = player_hand.count(bet_value)
        dice_counts = Counter(player_hand)
        most_common_value, most_common_count = dice_counts.most_common(1)[0]

        # Find dice with values above current bet
        higher_dice = [die_value for die_value in player_hand if die_value > bet_value]
        if higher_dice:
            higher_counts = Counter(higher_dice)
            best_higher_value, best_higher_count = higher_counts.most_common(1)[0]
        else:
            best_higher_value, best_higher_count = bet_value, 0

        # Calculate challenge probability (unused but kept for research consistency)
        challenge_prob = max(
            probability_r(bet_quantity - dice_matching_bet),
            probability_r(bet_quantity - best_higher_count),
            probability_r(bet_quantity + 1 - most_common_count),
        )

        # Make decision based on probability strategy rules
        if bet_quantity >= PROB_STRAT_BLUFF_THRESHOLD_3DICE:
            self.game.call_bluff(is_player_1=False)
        elif current_bet is None:
            self.game.make_bet(MIN_BET_QUANTITY, most_common_value, is_player_1=False)
        elif bet_quantity == dice_matching_bet:
            if higher_dice and (bet_quantity - best_higher_count) <= (
                bet_quantity + 1 - most_common_count
            ):
                self.game.make_bet(bet_quantity, best_higher_value, is_player_1=False)
            else:
                self.game.make_bet(
                    bet_quantity + 1, most_common_value, is_player_1=False
                )
        else:
            self.game.call_bluff(is_player_1=False)


# This function runs the NEAT algorithm to evolve neural networks for playing Liar's Dice.
def run_seeding_round(
    genome_list: List[Tuple[int, neat.DefaultGenome]], config: neat.Config, pygame_window: pygame.Surface
) -> Tuple[List[Tuple[int, neat.DefaultGenome]], Dict[int, int]]:
    """Evaluate all genomes against probability strategy to determine tournament seeding."""
    first_round_results = []
    seeding_wins = {}

    for genome_id, genome in genome_list:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        wins = 0

        for game_index in range(SEEDING_GAMES):
            dice_game = DiceGame(pygame_window, WINDOW_WIDTH, WINDOW_HEIGHT_GAME)

            # Alternate starting player
            starting_player = 1 if game_index < GAMES_PER_STARTING_PLAYER else 2
            result = dice_game.arena(network, starting_player=starting_player)

            if result == 1:  # Network wins
                wins += 1

        first_round_results.append((genome_id, genome, wins))
        seeding_wins[genome_id] = wins

    # Sort by performance and keep top performers
    first_round_results.sort(key=lambda x: x[2], reverse=True)
    cutoff = int(len(first_round_results) * TOP_PERFORMERS_CUTOFF)

    # Track statistics
    top_wins = [wins for _, _, wins in first_round_results[:cutoff]]
    avg_top_wins = sum(top_wins) / len(top_wins) if top_wins else 0
    std_dev = np.std(top_wins, ddof=1)

    print(
        f"Top {int(TOP_PERFORMERS_CUTOFF * 100)}% average wins in seeding round: {avg_top_wins}"
    )
    print(f"Standard deviation: {std_dev}")

    # Return qualified genomes and seeding performance
    qualified_genomes = [
        (gid, genome) for gid, genome, wins in first_round_results[:cutoff]
    ]
    return qualified_genomes, seeding_wins


def run_elimination_tournament(
    genome_list: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    pygame_window: pygame.Surface,
) -> Tuple[Optional[Tuple[int, neat.DefaultGenome]], Dict[int, int], Dict[int, int]]:
    """Run elimination tournament between qualified genomes."""
    current_genomes = genome_list[:] 
    rounds_reached = {genome_id: 0 for genome_id, _ in genome_list}
    extra_wins = {genome_id: 0 for genome_id, _ in genome_list}

    round_num = 1
    while len(current_genomes) > 1:
        # Track that all remaining genomes reached this round
        for genome_id, _ in current_genomes:
            rounds_reached[genome_id] = round_num

        next_round = []
        i = 0

        while i < len(current_genomes):
            if i + 1 < len(current_genomes):
                # Pair up two genomes for head-to-head competition
                genome_id_1, genome_1 = current_genomes[i]
                genome_id_2, genome_2 = current_genomes[i + 1]

                network_1 = neat.nn.FeedForwardNetwork.create(genome_1, config)
                network_2 = neat.nn.FeedForwardNetwork.create(genome_2, config)
                wins_1 = 0
                wins_2 = 0

                # Play tournament games between the two networks
                for game_index in range(TOURNAMENT_GAMES_PER_MATCHUP):
                    dice_game = DiceGame(
                        pygame_window, WINDOW_WIDTH, WINDOW_HEIGHT_GAME
                    )

                    # Alternate starting player
                    if game_index < GAMES_PER_STARTING_PLAYER:
                        result = dice_game.train_ai(
                            network_1, network_2, starting_player=1
                        )
                    else:
                        result = dice_game.train_ai(
                            network_1, network_2, starting_player=2
                        )

                    if result == 1:
                        wins_1 += 1
                    elif result == 2:
                        wins_2 += 1

                # Award extra fitness for wins above baseline
                extra_wins[genome_id_1] += max(0, wins_1 - FITNESS_BASELINE_WINS)
                extra_wins[genome_id_2] += max(0, wins_2 - FITNESS_BASELINE_WINS)

                # Advance the winner to next round
                if wins_1 > wins_2:
                    next_round.append((genome_id_1, genome_1))
                elif wins_2 > wins_1:
                    next_round.append((genome_id_2, genome_2))
                # Note: In case of tie, neither advances (rare but possible)

                i += 2
            else:
                # Odd genome gets a bye to next round
                next_round.append(current_genomes[i])
                i += 1

        current_genomes = next_round
        round_num += 1

    # Determine tournament champion
    if len(current_genomes) == 1:
        champion_id, champion_genome = current_genomes[0]
        rounds_reached[champion_id] = rounds_reached.get(champion_id, 0) + 1
        return (champion_id, champion_genome), rounds_reached, extra_wins
    else:
        # No clear winner - find genome that progressed furthest
        max_round = max(rounds_reached.values())
        best_candidates = [
            (gid, genome)
            for gid, genome in genome_list
            if rounds_reached.get(gid, 0) == max_round
        ]
        if best_candidates:
            champion_id, champion_genome = random.choice(best_candidates)
            return (champion_id, champion_genome), rounds_reached, extra_wins
        else:
            return None, rounds_reached, extra_wins


def calculate_fitness_scores(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    rounds_reached: Dict[int, int],
    extra_wins: Dict[int, int],
    seeding_wins: Dict[int, int],
) -> None:
    """Calculate fitness scores for all genomes based on tournament performance."""
    # Use champion tracker instead of global variables

    # Initialize preserved champion info if needed
    champion_id, champion_score, champion_genome, champion_round = (
        champion_tracker.get_champion_info()
    )
    if champion_id is None:
        champion_score = 0
        champion_round = 0
    else:
        champion_round = rounds_reached.get(champion_id, 0)

    # Calculate fitness for each genome
    for genome_id, genome in genomes:
        # Base fitness from tournament progress
        round_diff = rounds_reached.get(genome_id, 0) - champion_round
        genome.fitness = champion_score + (round_diff * FITNESS_ROUND_MULTIPLIER)

        # Bonus fitness from extra wins in tournament games
        genome.fitness += extra_wins.get(genome_id, 0) / FITNESS_EXTRA_WINS_DIVISOR

        # Bonus fitness from seeding round performance
        if genome_id in seeding_wins:
            genome.fitness += FITNESS_SEEDING_MULTIPLIER * seeding_wins[genome_id]

        # Store round reached for tracking
        genome.round_reached = rounds_reached.get(genome_id, 0)



def run_neat(config: neat.Config) -> None:

    def tournament_evaluation(genomes, config):
        """Coordinate tournament evaluation using modular functions.

        Args:
            genomes: List of (genome_id, genome) tuples to evaluate
            config: NEAT configuration object

        Returns:
            Champion genome from tournament
        """
        # Set up pygame window
        pygame_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT_GAME))
        pygame.display.set_caption("Liar's Dice Tournament")

        # Prepare genome list
        genome_list = genomes[:]
        random.shuffle(genome_list)

        # Phase 1: Run seeding round to rank genomes
        top_genomes, seeding_wins = run_seeding_round(
            genome_list, config, pygame_window
        )

        # Phase 2: Run elimination tournament
        tournament_result = run_elimination_tournament(
            top_genomes, config, pygame_window
        )
        champion_info, rounds_reached, extra_wins = tournament_result

        # Handle case where no champion emerged
        if champion_info is None:
            print("No genome progressed. Resetting champion tracking.")
            champion_tracker.reset()
            for gid, genome in genomes:
                genome.fitness = 0
                genome.round_reached = 0
            return None

        champion_id, champion_genome = champion_info

        # Phase 3: Calculate fitness scores for all genomes
        calculate_fitness_scores(genomes, rounds_reached, extra_wins, seeding_wins)

        # Phase 4: Update champion tracking
        champion_tracker.update_if_better(
            champion_id, champion_genome.fitness, champion_genome, 0
        )

        return champion_genome

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-199')
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(CHECKPOINT_FREQUENCY))

    metrics_reporter = MetricsReporter()
    population.add_reporter(metrics_reporter)

    best_network_test = BestNetworkTestReporter(
        config, DiceGame, file_prefix=NETWORK_SAVE_PREFIX_3DICE
    )
    population.add_reporter(best_network_test)

    winner = population.run(tournament_evaluation, MAX_GENERATIONS)

    print("\n--- Best Network Test Results ---")
    for generation, score in best_network_test.test_results:
        print(f"Generation {generation}: {score} wins")

    # Save the winning genome and network
    with open(GENOME_SAVE_NAME, "wb") as genome_file:
        pickle.dump(winner, genome_file)

    best_network = neat.nn.FeedForwardNetwork.create(winner, config)
    with open(NETWORK_SAVE_NAME, "wb") as network_file:
        pickle.dump(best_network, network_file)

    print(f"Saved network file size: {os.stat(NETWORK_SAVE_NAME).st_size} bytes")


def load_saved_networks(pattern="1v1_best_network_gen_*.pickle"):
    """Load saved networks and extract the generation number as identifier."""
    network_files = glob.glob(pattern)
    networks = []
    for file in network_files:
        match = re.search(r"1v1_best_network_gen_(\d+)\.pickle", file)
        if match:
            generation = int(match.group(1))
        else:
            continue
        with open(file, "rb") as f:
            net = pickle.load(f)
        networks.append((generation, net))
    return networks


def test_saved_networks_against_each_other(networks, num_games=100):
    """Run round-robin tournament between saved network generations."""
    pygame_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT_TEST))
    pygame.display.set_caption("Saved Networks Tournament")

    total_wins = {generation: 0 for generation, network in networks}

    for i in range(len(networks)):
        for j in range(i + 1, len(networks)):
            generation_1, network_1 = networks[i]
            generation_2, network_2 = networks[j]
            wins_1 = 0
            wins_2 = 0

            for game_index in range(num_games):
                dice_game = DiceGame(pygame_window, WINDOW_WIDTH, WINDOW_HEIGHT_TEST)

                if game_index % 2 == 0:
                    # Network 1 goes first
                    result = dice_game.arena(network_1, starting_player=1)
                    if result == 1:
                        wins_1 += 1
                    elif result == 2:
                        wins_2 += 1
                else:
                    # Network 2 goes first (same as network_1 being player 2)
                    result = dice_game.arena(network_2, starting_player=1)
                    if result == 1:
                        wins_2 += 1
                    elif result == 2:
                        wins_1 += 1

            print(
                f"Generation {generation_1} vs Generation {generation_2}: {wins_1}-{wins_2}"
            )
            total_wins[generation_1] += wins_1
            total_wins[generation_2] += wins_2

    return total_wins


def run_saved_tournament():
    """Run tournament between all saved network generations."""
    pygame.init()
    networks = load_saved_networks()
    results = test_saved_networks_against_each_other(networks, num_games=1000)

    print("\n--- Tournament Results ---")
    for generation, total_wins in sorted(results.items()):
        print(f"Generation {generation}: {total_wins} total wins")


if __name__ == "__main__":
    pygame.init()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs", "NEAT_raw_config.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    run_neat(config)
