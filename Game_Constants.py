"""
Game constants for Liar's Dice NEAT experiment.
Contains all configurable values and thresholds used throughout the project.
"""

# Game Configuration
DICE_FACES = 6
DICE_PER_PLAYER_3DICE = 3
DICE_PER_PLAYER_5DICE = 5
MIN_DICE_VALUE = 1
MAX_DICE_VALUE = 6

# Betting Limits
MIN_BET_QUANTITY = 1
MAX_BET_QUANTITY_3DICE = 6
MAX_BET_QUANTITY_5DICE = 10

# Neural Network Scaling
NETWORK_OUTPUT_SCALING = 5
BET_QUANTITY_OFFSET = 1

# Probability Strategy Thresholds
PROB_STRAT_BLUFF_THRESHOLD_3DICE = 4  # Call bluff if quantity >= 4 (for 3-dice game)
PROB_STRAT_BLUFF_THRESHOLD_5DICE = 6  # Call bluff if quantity >= 6 (for 5-dice game)

# Game Testing Parameters
TEST_GAMES_TOTAL = 10000
GAMES_PER_STARTING_POSITION = 5000
NETWORK_TEST_INTERVAL = 10  # Test every N generations

# Training Parameters
SEEDING_GAMES = 1000
TOURNAMENT_GAMES_PER_MATCHUP = 1000
GAMES_PER_STARTING_PLAYER = 500
TOP_PERFORMERS_CUTOFF = 0.5  # Top 50% advance from seeding

# Fitness Calculation
FITNESS_ROUND_MULTIPLIER = 5
FITNESS_SEEDING_MULTIPLIER = 0.02
FITNESS_EXTRA_WINS_DIVISOR = 100
FITNESS_BASELINE_WINS = 500  # Wins above this count as extra fitness

# Display Configuration
WINDOW_WIDTH = 750
WINDOW_HEIGHT_GAME = 750
WINDOW_HEIGHT_TEST = 700
DICE_DISPLAY_SPACING = 150
DICE_Y_OFFSET = 60
DICE_X_SPACING = 150
PLAYER_LABEL_Y_OFFSET = 40
PLAYER_1_Y_BASE = 60
PLAYER_2_Y_SPACING = 420
BET_DISPLAY_X_OFFSET = 40
BET_DIE_SPACING = 10
DICE_CENTER_OFFSET = 210
DICE_SIZE_OFFSET = 30

# Random Input Range
RANDOM_INPUT_MIN = -1.0
RANDOM_INPUT_MAX = 1.0

# Network Decision Thresholds
BLUFF_CALL_THRESHOLD = 0.5
BET_DECISION_THRESHOLD = 0.5

# File Naming Patterns
NETWORK_SAVE_PREFIX_3DICE = "1v1_best_network"
NETWORK_SAVE_PREFIX_5DICE = "4v4_best_pnetwork"
GENOME_SAVE_NAME = "best_genome_selfplay.pickle"
NETWORK_SAVE_NAME = "best_network_selfplay.pickle"

# Checkpoint Configuration
CHECKPOINT_FREQUENCY = 20
MAX_GENERATIONS = 200