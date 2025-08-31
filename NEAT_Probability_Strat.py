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
        self.game = Game(window, window_width, window_height, dice_per_player=DICE_PER_PLAYER_5DICE)

    def arena(
        self,
        best_net: neat.nn.FeedForwardNetwork,
        draw: bool = False,
        show_both_hands: bool = False,
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
                self.probability_strat_p2()
                if self.game.round_over:
                    return self.game.round_winner
            else:
                self.probability_strat_p2()
                if self.game.round_over:
                    return self.game.round_winner
                self.bet_or_bluff_p1(best_net)
                if self.game.round_over:
                    return self.game.round_winner

            if draw:
                self.game.draw(show_both_hands)

            pygame.display.update()

    def train_ai(
        self,
        net1: neat.nn.FeedForwardNetwork,
        net2: neat.nn.FeedForwardNetwork,
        draw: bool = False,
        show_both_hands: bool = False,
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

    def bet_or_bluff_p1(self, network: neat.nn.FeedForwardNetwork) -> None:
        """Make betting decision for player 1 using neural network with probability features."""
        # Get current game state
        dice_values = self.game.p1_hand
        d1, d2, d3, d4, d5 = dice_values

        current_bet = self.game.current_bet
        bet_quantity, bet_value = current_bet if current_bet else (0, 0)
        previous_bet = self.game.bet_history[-1] if self.game.bet_history else (0, 0)
        previous_quantity, previous_value = previous_bet

        # Analyze hand composition
        dice_counts = Counter(dice_values)
        most_common_value, most_common_count = dice_counts.most_common(1)[0]

        # Find dice with values above current bet
        higher_dice = [die_value for die_value in dice_values if die_value > bet_value]
        if higher_dice:
            higher_counts = Counter(higher_dice)
            best_higher_value, best_higher_count = higher_counts.most_common(1)[0]
        else:
            best_higher_value, best_higher_count = 0, 0

        # Calculate probability features for network
        dice_matching_bet = dice_values.count(bet_value)
        current_bet_probability = probability_r(
            bet_quantity - dice_matching_bet, n=DICE_PER_PLAYER_5DICE
        )
        next_bet_probability = max(
            probability_r(bet_quantity - best_higher_count, n=DICE_PER_PLAYER_5DICE),
            probability_r(
                bet_quantity + 1 - most_common_count, n=DICE_PER_PLAYER_5DICE
            ),
        )

        # Prepare neural network inputs with probability features
        network_inputs = [
            d1,
            d2,
            d3,
            d4,
            d5,
            bet_quantity,
            bet_value,
            previous_quantity,
            previous_value,
            most_common_count,
            best_higher_count,
            next_bet_probability,
            current_bet_probability,
            1,  # Player indicator
        ]
        output = network.activate(network_inputs)

        # Make decision based on network outputs
        random_threshold = random.uniform(0, 1)

        if current_bet is None:
            # First bet of round
            self.game.make_bet(MIN_BET_QUANTITY, most_common_value, is_player_1=True)
        elif bet_quantity == MAX_BET_QUANTITY_5DICE:
            # Maximum bet reached - must call bluff
            self.game.call_bluff(is_player_1=True)
        elif random_threshold <= output[0]:
            # Network chooses to call bluff
            self.game.call_bluff(is_player_1=True)
        elif random_threshold <= output[1] and higher_dice:
            # Network chooses to raise with higher value
            self.game.make_bet(bet_quantity, best_higher_value, is_player_1=True)
        else:
            # Network chooses to raise quantity
            self.game.make_bet(
                min(MAX_BET_QUANTITY_5DICE, bet_quantity + 1),
                most_common_value,
                is_player_1=True,
            )

    def bet_or_bluff_p2(self, network: neat.nn.FeedForwardNetwork) -> None:
        """Make betting decision for player 2 using neural network with probability features."""
        # Get current game state
        dice_values = self.game.p2_hand
        d1, d2, d3, d4, d5 = dice_values

        current_bet = self.game.current_bet
        bet_quantity, bet_value = current_bet if current_bet else (0, 0)
        previous_bet = self.game.bet_history[-1] if self.game.bet_history else (0, 0)
        previous_quantity, previous_value = previous_bet

        # Analyze hand composition
        dice_counts = Counter(dice_values)
        most_common_value, most_common_count = dice_counts.most_common(1)[0]

        # Find dice with values above current bet
        higher_dice = [die_value for die_value in dice_values if die_value > bet_value]
        if higher_dice:
            higher_counts = Counter(higher_dice)
            best_higher_value, best_higher_count = higher_counts.most_common(1)[0]
        else:
            best_higher_value, best_higher_count = 0, 0

        # Calculate probability features for network
        dice_matching_bet = dice_values.count(bet_value)
        current_bet_probability = probability_r(
            bet_quantity - dice_matching_bet, n=DICE_PER_PLAYER_5DICE
        )
        next_bet_probability = max(
            probability_r(bet_quantity - best_higher_count, n=DICE_PER_PLAYER_5DICE),
            probability_r(
                bet_quantity + 1 - most_common_count, n=DICE_PER_PLAYER_5DICE
            ),
        )

        # Prepare neural network inputs with probability features
        network_inputs = [
            d1,
            d2,
            d3,
            d4,
            d5,
            bet_quantity,
            bet_value,
            previous_quantity,
            previous_value,
            most_common_count,
            best_higher_count,
            next_bet_probability,
            current_bet_probability,
            1,  # Player indicator
        ]
        output = network.activate(network_inputs)

        # Make decision based on network outputs
        random_threshold = random.uniform(0, 1)

        if current_bet is None:
            # First bet of round
            self.game.make_bet(MIN_BET_QUANTITY, most_common_value, is_player_1=False)
        elif bet_quantity == MAX_BET_QUANTITY_5DICE:
            # Maximum bet reached - must call bluff
            self.game.call_bluff(is_player_1=False)
        elif random_threshold <= output[0]:
            # Network chooses to call bluff
            self.game.call_bluff(is_player_1=False)
        elif random_threshold <= output[1] and higher_dice:
            # Network chooses to raise with higher value
            self.game.make_bet(bet_quantity, best_higher_value, is_player_1=False)
        else:
            # Network chooses to raise quantity
            self.game.make_bet(
                min(MAX_BET_QUANTITY_5DICE, bet_quantity + 1),
                most_common_value,
                is_player_1=False,
            )

    def probability_strat_p2(self) -> None:
        """Probability-based strategy for player 2 (5-dice variant)."""
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
        challenge_probability = max(
            probability_r(bet_quantity - dice_matching_bet, n=DICE_PER_PLAYER_5DICE),
            probability_r(bet_quantity - best_higher_count, n=DICE_PER_PLAYER_5DICE),
            probability_r(
                bet_quantity + 1 - most_common_count, n=DICE_PER_PLAYER_5DICE
            ),
        )

        # Make decision based on probability strategy rules
        if bet_quantity >= PROB_STRAT_BLUFF_THRESHOLD_5DICE:
            self.game.call_bluff(is_player_1=False)
        elif current_bet is None:
            self.game.make_bet(MIN_BET_QUANTITY, most_common_value, is_player_1=False)
        elif bet_quantity <= dice_matching_bet:
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


def run_seeding_round(
    genome_list: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    pygame_window: pygame.Surface,
) -> Tuple[List[Tuple[int, neat.DefaultGenome]], Dict[int, int]]:
    """Run seeding games against probability strategy to rank genomes.

    Args:
        genome_list: List of (genome_id, genome) tuples to evaluate
        pygame_window: Pygame window for game display

    Returns:
        List of top-performing genomes ready for tournament
    """
    global top50_avg_wins, top50_std

    # Initialize global tracking lists if not defined
    try:
        top50_avg_wins
    except NameError:
        top50_avg_wins = []
    try:
        top50_std
    except NameError:
        top50_std = []

    # Evaluate each genome against probability strategy
    first_round_results = []
    seeding_wins = {}
    for genome_id, genome in genome_list:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        wins = 0
        for game_index in range(SEEDING_GAMES):
            dice_game = DiceGame(pygame_window, WINDOW_WIDTH, WINDOW_HEIGHT_GAME)
            if game_index < GAMES_PER_STARTING_PLAYER:
                result = dice_game.arena(network, starting_player=1)
                if result == 1:
                    wins += 1
            else:
                result = dice_game.arena(network, starting_player=2)
                if result == 1:
                    wins += 1
        first_round_results.append((genome_id, genome, wins))
        seeding_wins[genome_id] = wins

    # Sort genomes by performance and select top performers
    first_round_results.sort(key=lambda x: x[2], reverse=True)
    cutoff = int(len(first_round_results) * TOP_PERFORMERS_CUTOFF)
    top_wins = [wins for _, _, wins in first_round_results[:cutoff]]
    avg_top_wins = sum(top_wins) / len(top_wins) if top_wins else 0
    std_dev = np.std(top_wins, ddof=1)
    top50_avg_wins.append(avg_top_wins)
    top50_std.append(std_dev)
    print(
        f"Top {int(TOP_PERFORMERS_CUTOFF * 100)}% average wins in seeding round: {avg_top_wins}"
    )
    print(f"Standard deviation: {std_dev}")

    # Return top performing genomes for tournament
    top_genomes = [
        (genome_id, genome) for genome_id, genome, wins in first_round_results[:cutoff]
    ]
    return top_genomes, seeding_wins


def run_elimination_tournament(
    genome_list: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    pygame_window: pygame.Surface,
) -> Tuple[Optional[Tuple[int, neat.DefaultGenome]], Dict[int, int], Dict[int, int]]:
    """Run elimination tournament between genomes.

    Args:
        genome_list: List of seeded (genome_id, genome) tuples
        config: NEAT configuration object
        pygame_window: Pygame window for game display

    Returns:
        Tuple of (champion_genome_or_None, rounds_reached, extra_wins)
    """
    # Dictionary to track how far each genome progresses
    rounds_reached = {genome_id: 0 for genome_id, _ in genome_list}
    # Track extra fitness for wins above baseline
    extra_wins = {genome_id: 0 for genome_id, _ in genome_list}

    round_num = 1
    current_genomes = genome_list[:]

    while len(current_genomes) > 1:
        # Update the round reached for all genomes in this round
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

                for game_index in range(TOURNAMENT_GAMES_PER_MATCHUP):
                    dice_game = DiceGame(
                        pygame_window, WINDOW_WIDTH, WINDOW_HEIGHT_GAME
                    )

                    # Alternate starting player for fairness
                    if game_index < GAMES_PER_STARTING_PLAYER:
                        result = dice_game.train_ai(
                            network_1, network_2, starting_player=1
                        )
                        if result == 1:
                            wins_1 += 1
                        elif result == 2:
                            wins_2 += 1
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

                # The genome with more wins progresses
                if wins_1 > wins_2:
                    next_round.append((genome_id_1, genome_1))
                elif wins_2 > wins_1:
                    next_round.append((genome_id_2, genome_2))
                # In case of a tie, neither genome progresses

                i += 2
            else:
                # Unpaired genome gets a bye
                next_round.append(current_genomes[i])
                i += 1

        current_genomes = next_round
        round_num += 1

    # Determine the champion
    if len(current_genomes) == 1:
        champion_id, champion_genome = current_genomes[0]
        # Record the champion's round reached
        rounds_reached[champion_id] = rounds_reached.get(champion_id, 0) + 1
        return (champion_id, champion_genome), rounds_reached, extra_wins
    else:
        # No clear champion found - handle edge case
        max_round = max(rounds_reached.values()) if rounds_reached else 0
        best_candidates = [
            (gid, genome)
            for gid, genome in current_genomes
            if rounds_reached.get(gid, 0) == max_round
        ]
        if best_candidates:
            champion_id, champion_genome = random.choice(best_candidates)
            rounds_reached[champion_id] = rounds_reached.get(champion_id, 0) + 1
            return (champion_id, champion_genome), rounds_reached, extra_wins
        else:
            return None, rounds_reached, extra_wins


def calculate_fitness_scores(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    rounds_reached: Dict[int, int],
    extra_wins: Dict[int, int],
    seeding_wins: Dict[int, int],
) -> None:
    """Calculate and assign fitness scores to all genomes.

    Args:
        genomes: List of all (genome_id, genome) tuples
        rounds_reached: Dictionary mapping genome_id to tournament round reached
        extra_wins: Dictionary mapping genome_id to bonus wins above baseline
        seeding_wins: Dictionary mapping genome_id to seeding round wins
    """
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
        # Base fitness from tournament progress relative to champion
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
        config, DiceGame, file_prefix=NETWORK_SAVE_PREFIX_5DICE
    )
    population.add_reporter(best_network_test)

    winner = population.run(tournament_evaluation, MAX_GENERATIONS)

    # Output test best network results.
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


def draw_net_layers(
    config,
    genome,
    view=False,
    filename="layered_net.gv",
    node_names=None,
    show_disabled=True,
    prune_unused=False,
    fmt="png",
):
    """
    Draws a NEAT network with 4 visual layers:
      - Layer 1: Input nodes (blue, box)
      - Layer 2: Hidden nodes, group 1 (red, circle)
      - Layer 3: Hidden nodes, group 2 (orange, circle)
      - Layer 4: Output nodes (seagreen, box)

    Hidden nodes are assigned to one of two groups:
      - Group 1 (if computed depth < 2)
      - Group 2 (if computed depth >= 2)

    Additionally, if a connection exists between two hidden nodes in the same group,
    it is drawn as a recursive (feedback) connection (using constraint=false and dir=back).

    Arguments:
        config       -- NEAT configuration.
        genome       -- The genome to visualize.
        view         -- If True, open the generated image with the default viewer.
        filename     -- Base filename for output (without extension).
        node_names   -- Optional dict mapping node IDs to labels.
        show_disabled-- If False, skip drawing disabled connections.
        prune_unused -- If True, prune unused nodes.
        fmt          -- Output format (e.g., 'png', 'pdf', 'svg').
    """
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    # Define input and output sets.
    input_keys = set(config.genome_config.input_keys)
    output_keys = set(config.genome_config.output_keys)

    # All nodes is the union of genome nodes plus input and output keys.
    all_nodes = set(genome.nodes.keys()) | input_keys | output_keys

    # Compute a basic feed-forward depth ignoring cycles.
    memo = {}

    def get_depth(node):
        if node in memo:
            return memo[node]
        if node in input_keys:
            memo[node] = 0
            return 0
        incoming = []
        for (u, v), conn in genome.connections.items():
            if conn.enabled and v == node:
                incoming.append(get_depth(u))
        d = max(incoming) + 1 if incoming else 0
        memo[node] = d
        return d

    depths = {n: get_depth(n) for n in all_nodes}

    # For hidden nodes (not input or output), assign them to one of two hidden groups.
    # We'll use a simple rule: if depth < 2, assign to hidden group 1, else group 2.
    hidden_groups = {}
    for n in all_nodes:
        if n not in input_keys and n not in output_keys:
            hidden_groups[n] = 1 if depths[n] < 2 else 2

    # Now assign a final visual layer:
    # Inputs: layer 1, hidden group 1: layer 2, hidden group 2: layer 3, outputs: layer 4.
    visual_layers = {}
    for n in all_nodes:
        if n in input_keys:
            visual_layers[n] = 1
        elif n in output_keys:
            visual_layers[n] = 4
        else:
            visual_layers[n] = 1 + hidden_groups[n]  # so group 1 -> 2, group 2 -> 3

    # Group nodes by visual layer.
    groups = {}
    for n, v in visual_layers.items():
        groups.setdefault(v, []).append(n)

    dot = Digraph(format=fmt)
    dot.attr(rankdir="LR", splines="line", nodesep=".05")

    # Create subgraphs for each visual layer.
    for layer_num in sorted(groups.keys()):
        with dot.subgraph(name=f"cluster_layer_{layer_num}") as s:
            s.attr(rank="same", label=f"Layer {layer_num}", color="white")
            for node in groups[layer_num]:
                label = node_names.get(node, str(node))
                if node in input_keys:
                    fillcolor = "blue"
                    shape = "box"
                elif node in output_keys:
                    fillcolor = "seagreen"
                    shape = "box"
                else:
                    # For hidden nodes, use different colors by group.
                    fillcolor = "red" if hidden_groups[node] == 1 else "orange"
                    shape = "circle"
                s.node(
                    str(node),
                    label=label,
                    style="filled",
                    fillcolor=fillcolor,
                    shape=shape,
                )

    # Add edges.
    for (u, v), conn in genome.connections.items():
        if not conn.enabled and not show_disabled:
            continue
        a = node_names.get(u, str(u))
        b = node_names.get(v, str(v))
        style = "solid" if conn.enabled else "dotted"
        color = "green" if conn.weight > 0 else "red"
        penwidth = str(max(0.1, abs(conn.weight) * 2))

        # If both u and v are hidden and are in the same hidden group, treat as a recursive edge.
        if (
            u not in input_keys
            and u not in output_keys
            and v not in input_keys
            and v not in output_keys
            and hidden_groups.get(u, 1) == hidden_groups.get(v, 1)
        ):
            dot.edge(
                str(u),
                str(v),
                label="{:.3f}".format(conn.weight),
                style=style,
                color=color,
                penwidth=penwidth,
                constraint="false",
                dir="back",
            )
        else:
            dot.edge(
                str(u),
                str(v),
                label="{:.3f}".format(conn.weight),
                style=style,
                color=color,
                penwidth=penwidth,
            )

    output_path = dot.render(filename, view=view)
    print("Layered network graph saved to:", output_path)
    return dot


def load_saved_networks(pattern="4v4_best_pnetwork_gen_*.pickle"):
    """Load saved networks and extract the generation number as identifier."""
    network_files = glob.glob(pattern)
    networks = []
    for file in network_files:
        # Extract generation number using regex.
        match = re.search(r"4v4_best_pnetwork_gen_(\d+)\.pickle", file)
        if match:
            generation = int(match.group(1))
        else:
            # If the file doesn't match the expected pattern, skip it.
            continue
        with open(file, "rb") as f:
            net = pickle.load(f)
        networks.append((generation, net))
    return networks


def test_saved_networks_against_each_other(networks, num_games=100):
    """
    Run a round-robin tournament between saved networks.

    For each pair of networks, play num_games (alternating starting players)
    and tally wins. Returns a dictionary mapping network identifiers to total wins.
    """
    # Set up a Pygame display for testing.
    width, height = 750, 700
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Saved Networks Tournament")

    wins = {ident: 0 for ident, net in networks}

    # For every unique pair of networks:
    for i in range(len(networks)):
        for j in range(i + 1, len(networks)):
            ident1, net1 = networks[i]
            ident2, net2 = networks[j]
            wins1 = 0
            wins2 = 0

            # Play num_games matches. Alternate starting player.
            for game_index in range(num_games):
                # Create a fresh game instance for each match.
                dice = DiceGame(win, width, height)
                if game_index % 2 == 0:
                    # Network 1 starts first.
                    result = dice.arena(net1, starting_player=1)
                    if result == 1:
                        wins1 += 1
                    elif result == 2:
                        wins2 += 1
                else:
                    # Network 2 starts first.
                    result = dice.arena(net2, starting_player=1)
                    if result == 1:
                        wins2 += 1
                    elif result == 2:
                        wins1 += 1

            print(f"{ident1} v {ident2} {wins1} , {wins2}")
            wins[ident1] += wins1
            wins[ident2] += wins2

    return wins


def run_saved_tournament():
    pygame.init()
    networks = load_saved_networks()  # Load all saved networks.
    results = test_saved_networks_against_each_other(networks, num_games=1000)

    print("\n--- Tournament Results ---")
    for ident, total_wins in results.items():
        print(f"{ident} {total_wins}")


if __name__ == "__main__":
    pygame.init()  # Initialize all pygame modules
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "configs", "NEAT_prob_strat_config.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    run_neat(config=config)
