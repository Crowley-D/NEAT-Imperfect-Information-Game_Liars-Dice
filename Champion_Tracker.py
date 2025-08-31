"""
Champion tracking functionality for NEAT experiments.
Encapsulates global champion state management.
"""
import neat
from typing import Optional, Tuple


class ChampionTracker:
    """Manages champion state across tournament generations."""
    
    def __init__(self) -> None:
        """Initialize champion tracking state."""
        self.champion_id: Optional[int] = None
        self.champion_score: int = 0
        self.champion_genome: Optional[neat.DefaultGenome] = None
        self.champion_round: int = 0
    
    def update_if_better(self, genome_id: int, score: int, genome: neat.DefaultGenome, round_num: int) -> bool:
        """Update champion if new genome performs better.
        
        Args:
            genome_id: ID of the candidate genome
            score: Performance score of the candidate
            genome: The candidate genome object  
            round_num: Tournament round reached
            
        Returns:
            True if champion was updated, False otherwise
        """
        if score > self.champion_score:
            self.champion_id = genome_id
            self.champion_score = score
            self.champion_genome = genome
            self.champion_round = round_num
            return True
        return False
    
    def get_champion_info(self) -> Tuple[Optional[int], int, Optional[neat.DefaultGenome], int]:
        """Get current champion information as tuple."""
        return (self.champion_id, self.champion_score, self.champion_genome, self.champion_round)
    
    def reset(self) -> None:
        """Reset champion tracking state."""
        self.champion_id = None
        self.champion_score = 0
        self.champion_genome = None
        self.champion_round = 0