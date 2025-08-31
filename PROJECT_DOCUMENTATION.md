# Liar's Dice NEAT Project - Comprehensive Documentation

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture & Design](#architecture--design)
- [File Structure](#file-structure)
- [Core Components](#core-components)
- [Experimental Variants](#experimental-variants)
- [Configuration & Constants](#configuration--constants)
- [Testing Framework](#testing-framework)
- [Usage Guide](#usage-guide)
- [Research Context](#research-context)
- [Technical Implementation Details](#technical-implementation-details)

## Project Overview

This project implements a NEAT (NeuroEvolution of Augmenting Topologies) system to learn optimal strategies for Liar's Dice, an imperfect information game. The research explores whether artificial neural networks can evolve competitive strategies against rule-based opponents and examines game theory implications.

### Key Research Questions
- Can NEAT networks learn to play Liar's Dice effectively against probability-based strategies?
- How do evolved strategies perform against different generations of networks?
- What network topologies emerge for optimal imperfect information gameplay?

## Architecture & Design

### Design Principles
- **Modular Architecture**: Single-responsibility functions with clear interfaces
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Professional Standards**: Publication-ready code quality with documentation
- **Testability**: Unit test coverage for core functionality
- **Configurability**: Centralized constants for easy experimentation

### Core Design Patterns
- **Dependency Injection**: Eliminates circular imports and improves testability
- **Strategy Pattern**: Different game strategies (neural network vs probability-based)
- **State Management**: Thread-safe champion tracking with ChampionTracker class
- **Template Method**: Consistent tournament evaluation flow across variants

## File Structure

```
Liar's Dice/
├── Core Game Engine
│   ├── Liars_Dice_Game.py          # Main game mechanics and rules
│   ├── Game_Constants.py           # Centralized configuration constants  
│   └── Game_Utils.py               # Shared utility functions
│
├── NEAT Experiments
│   ├── NEAT.py                     # 3-dice tournament evolution experiment
│   └── NEAT_Probability_Strat.py   # 5-dice probability-enhanced experiment
│
├── Support Systems  
│   ├── Champion_Tracker.py         # Thread-safe state management
│   └── configs/                    # NEAT configuration files
│       ├── NEAT_raw_config.txt     # 3-dice experiment config
│       └── NEAT_prob_strat_config.txt # 5-dice experiment config
│
├── Testing Framework
│   ├── tests/
│   │   ├── test_game_mechanics.py  # Core game logic tests (7 tests)
│   │   └── test_game_utils.py      # Utility function tests (4 tests)
│
├── Assets & Output
│   ├── assets/dice/                # Dice image assets
│   └── Topologies/                 # Generated network visualizations
│
└── Documentation
    └── PROJECT_DOCUMENTATION.md    # This file
```

## Core Components

### 1. Game Engine (`Liars_Dice_Game.py`)
**Purpose**: Implements the core Liar's Dice game mechanics

**Key Classes**:
- `Game`: Main game state and rule enforcement

**Key Methods**:
```python
def make_bet(self, quantity: int, value: int, is_player_1: bool) -> None
    # Validates and processes betting decisions

def call_bluff(self, is_player_1: bool) -> bool
    # Handles bluff calling and determines winners

def reroll_hands(self) -> None
    # Generates new random dice for next round
```

**Features**:
- Robust bet validation with automatic round termination for invalid bets
- Professional pygame rendering with fallback text for missing assets
- Comprehensive error handling and state management

### 2. Game Utilities (`Game_Utils.py`)
**Purpose**: Shared functionality used across experiments

**Key Functions**:
```python
def probability_r(r: int, n: int = 3) -> float
    # Calculates probability of rolling at least r dice of same value

def test_best_network(best_net, dice_game_class, n_dice: int = 5) -> int
    # Tests evolved networks against probability strategy

class MetricsReporter(neat.reporting.BaseReporter)
    # Tracks and logs evolution metrics

class BestNetworkTestReporter(neat.reporting.BaseReporter)  
    # Tests and saves best networks each generation
```

**Features**:
- Dependency injection to prevent circular imports
- Professional NEAT reporter classes for comprehensive logging
- Centralized probability calculations used by all strategies

### 3. Configuration Management (`Game_Constants.py`)
**Purpose**: Centralized constants for easy experimentation

**Configuration Categories**:
```python
# Game Rules
DICE_PER_PLAYER_3DICE = 3
DICE_PER_PLAYER_5DICE = 5
MIN_BET_QUANTITY = 1
MAX_BET_QUANTITY_3DICE = 6
MAX_BET_QUANTITY_5DICE = 10

# Tournament Settings
SEEDING_GAMES = 1000
TOURNAMENT_GAMES_PER_MATCHUP = 20
TOP_PERFORMERS_CUTOFF = 0.5

# Strategy Parameters
PROB_STRAT_BLUFF_THRESHOLD_3DICE = 4
PROB_STRAT_BLUFF_THRESHOLD_5DICE = 6

# Evolution Settings
MAX_GENERATIONS = 300
CHECKPOINT_FREQUENCY = 50
```

### 4. Champion Tracking (`Champion_Tracker.py`)
**Purpose**: Thread-safe state management for preserving best performers

**Key Features**:
- Maintains champion information across generations
- Thread-safe operations for concurrent access
- Graceful handling of missing or reset champions

## Experimental Variants

### Experiment 1: 3-Dice Tournament (`NEAT.py`)
**Research Focus**: Pure self-play evolution in simplified environment

**Characteristics**:
- 3 dice per player for faster convergence
- Tournament-style elimination brackets
- Networks compete only against other networks
- Simplified input space (9 inputs) for cleaner evolution

**Tournament Flow**:
1. **Seeding Round**: All genomes evaluated against probability strategy
2. **Elimination Tournament**: Top 50% compete in bracket-style matches  
3. **Fitness Calculation**: Based on tournament progress and extra wins
4. **Champion Tracking**: Best performer preserved across generations

**Key Functions**:
```python
def run_seeding_round(genome_list, config, pygame_window) -> Tuple[List, Dict]
    # Ranks all genomes against baseline strategy

def run_elimination_tournament(genome_list, config, pygame_window) -> Tuple[Optional[Tuple], Dict, Dict]  
    # Runs bracket tournament between qualified genomes

def calculate_fitness_scores(genomes, rounds_reached, extra_wins, seeding_wins) -> None
    # Assigns fitness based on tournament performance
```

### Experiment 2: 5-Dice Probability-Enhanced (`NEAT_Probability_Strat.py`)
**Research Focus**: Enhanced strategy learning with probability features

**Characteristics**:
- 5 dice per player for realistic complexity
- Probability features integrated into network inputs
- Networks trained against sophisticated probability strategy
- Enhanced input space (14 inputs) including calculated probabilities

**Enhanced Network Inputs**:
```python
network_inputs = [
    d1, d2, d3, d4, d5,                    # Individual dice values
    bet_quantity, bet_value,                # Current bet information  
    previous_quantity, previous_value,      # Bet history
    most_common_count,                     # Hand analysis
    best_higher_count,                     # Strategic options
    next_bet_probability,                  # Calculated probabilities
    current_bet_probability,               # Risk assessment
    1                                      # Player identifier
]
```

**Strategic Features**:
- Probability-based opponent strategy using mathematical risk assessment
- Enhanced network decision making with pre-calculated probability features
- More complex betting validation and strategic options

## Configuration & Constants

### NEAT Configuration Files

**3-Dice Configuration** (`configs/NEAT_raw_config.txt`):
- Optimized for simple 3-dice environment
- 9 input nodes (dice + game state)
- 3 output nodes (call_bluff, raise_value, raise_quantity)

**5-Dice Configuration** (`configs/NEAT_prob_strat_config.txt`):
- Enhanced for complex 5-dice environment with probability features
- 14 input nodes (dice + game state + probabilities)
- 3 output nodes with probability-informed decision making

### Experiment Parameters

**Evolution Settings**:
- **Population Size**: 150 genomes per generation
- **Max Generations**: 300 (configurable)
- **Checkpoint Frequency**: Every 50 generations
- **Tournament Size**: Top 50% of seeding round performers

**Game Settings**:
- **Seeding Games**: 1000 games per genome vs probability strategy
- **Tournament Games**: 20 games per matchup (10 each starting position)
- **Fitness Calculation**: Tournament progress + extra wins + seeding performance

## Testing Framework

### Test Coverage
**Total**: 11 unit tests across 2 test files
- `test_game_mechanics.py`: 7 tests covering core game functionality
- `test_game_utils.py`: 4 tests covering utility functions

### Key Test Cases

**Game Mechanics Tests**:
```python
def test_valid_bet_acceptance()
    # Verifies proper bet validation logic

def test_invalid_bet_ends_round()  
    # Ensures invalid bets terminate rounds correctly

def test_call_bluff_success()
    # Tests successful bluff calling scenarios

def test_call_bluff_failure()
    # Tests failed bluff calling scenarios
```

**Utility Function Tests**:
```python
def test_probability_calculation()
    # Validates mathematical probability calculations

def test_network_testing_functionality() 
    # Ensures network testing works correctly
```

### Running Tests
```bash
# Run all tests
python tests/test_game_mechanics.py
python tests/test_game_utils.py

# Expected output: All 11 tests pass
```

## Usage Guide

### Quick Start

1. **Install Dependencies**:
```bash
pip install pygame neat-python numpy
```

2. **Run 3-Dice Experiment**:
```python
python NEAT.py
```

3. **Run 5-Dice Probability Experiment**:
```python
python NEAT_Probability_Strat.py
```

### Configuration Modification

**Adjusting Evolution Parameters**:
Edit `Game_Constants.py` to modify:
- Tournament sizes and game counts
- Evolution generations and population
- Strategy thresholds and probabilities

**Modifying NEAT Settings**:
Edit configuration files in `configs/` directory:
- Network topology parameters
- Evolution algorithm settings
- Fitness function weights

### Network Analysis

**Saved Networks**:
- Networks automatically saved each generation to pickle files
- File naming: `{experiment}_best_network_gen_{generation}.pickle`
- Load and test networks using `test_saved_networks_against_each_other()`

**Visualization**:
- Network topologies saved to `Topologies/` directory
- Use `draw_net_layers()` function for custom visualizations

## Research Context

### Academic Background
This project serves as an undergraduate economics dissertation exploring:
- **Game Theory**: Nash equilibrium analysis in imperfect information games
- **Artificial Intelligence**: NEAT's effectiveness for strategic decision making
- **Evolutionary Computation**: Network topology evolution for game playing

### Key Findings
- Networks successfully learned to beat probability strategies
- Performance varied significantly against different generations
- Complex topologies emerged for handling imperfect information
- Evidence supports Nash equilibrium importance in game theory

### Research Methodology
1. **Baseline Establishment**: Probability-based strategy performance measurement
2. **Evolution Process**: NEAT network evolution against baseline
3. **Cross-Generation Testing**: Tournament between evolved generations
4. **Statistical Analysis**: Performance metrics and topology analysis

## Technical Implementation Details

### Tournament Architecture

**Modular Design**: Each experiment uses identical tournament flow:
1. **Seeding Phase**: Genome evaluation and ranking
2. **Tournament Phase**: Elimination bracket competition  
3. **Fitness Phase**: Score calculation and assignment
4. **Tracking Phase**: Champion preservation and logging

**Thread Safety**: All state management uses ChampionTracker class to prevent race conditions and ensure consistent champion preservation.

**Error Handling**: Comprehensive error handling with graceful degradation:
- Missing pygame assets fall back to text rendering
- Failed tournament rounds reset champion tracking
- Invalid network outputs handled with default strategies

### Network Input Engineering

**3-Dice Inputs** (9 total):
```python
[d1, d2, d3,                          # Player's dice
 bet_quantity, bet_value,              # Current bet
 previous_quantity, previous_value,    # Previous bet
 random_input,                        # Exploration factor
 player_indicator]                    # Player identification
```

**5-Dice Enhanced Inputs** (14 total):
```python  
[d1, d2, d3, d4, d5,                 # Player's dice
 bet_quantity, bet_value,             # Current bet
 previous_quantity, previous_value,   # Previous bet
 most_common_count,                  # Hand strength
 best_higher_count,                  # Strategic options
 next_bet_probability,               # Risk assessment
 current_bet_probability,            # Bluff detection
 player_indicator]                   # Player identification
```

### Performance Optimization

**Memory Management**:
- Efficient pygame surface handling
- Proper network cleanup between games
- Checkpoint system for long-running experiments

**Computation Efficiency**:
- Vectorized probability calculations
- Minimal object creation in game loops
- Optimized tournament bracket algorithms

### Extensibility

**Adding New Strategies**:
1. Implement strategy in DiceGame class as new method
2. Add configuration constants to `Game_Constants.py`
3. Update network input/output handling as needed

**Modifying Tournament Structure**:
1. Extend tournament functions in experiment files
2. Update fitness calculation in `calculate_fitness_scores()`
3. Add new metrics to reporter classes

**Expanding Test Coverage**:
1. Add new test cases to appropriate test files
2. Follow existing test patterns for consistency
3. Ensure all new functionality is covered

---

## Project Maintenance

### Code Quality Standards
- **Type Safety**: All functions include comprehensive type hints
- **Documentation**: Professional docstrings for all public methods
- **Testing**: Minimum 90% test coverage for core functionality
- **Style**: Consistent formatting following professional standards

### Future Enhancements
- Multi-player tournament support (3+ players)
- Advanced visualization of network decision patterns
- Integration with machine learning frameworks
- Extended statistical analysis tools

For detailed implementation history, see `CODE_QUALITY_IMPROVEMENT_PLAN.md` and `NEAT_STANDARDIZATION_IMPLEMENTATION_LOG.md`.