"""
Unit tests for core game mechanics in liars_dice_game.py.
Tests betting validation, bluff calling, and game state management.
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"  # Prevent pygame from opening windows
import pygame
pygame.init()

from Liars_Dice_Game import Game


class TestGameMechanics(unittest.TestCase):
    """Test cases for core game mechanics."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.window = pygame.display.set_mode((750, 750))
        # Create game with known dice for predictable testing
        self.game = Game(self.window, 750, 750, p1_hand=[1, 2, 3], p2_hand=[4, 5, 6])
    
    def test_valid_bet_acceptance(self):
        """Test that valid bets are accepted correctly."""
        # Initial bet should be accepted
        self.game.make_bet(2, 3, is_player_1=True)
        self.assertEqual(self.game.current_bet, (2, 3))
        self.assertFalse(self.game.round_over)
        
        # Higher quantity bet should be accepted
        self.game.make_bet(3, 3, is_player_1=False)
        self.assertEqual(self.game.current_bet, (3, 3))
        self.assertFalse(self.game.round_over)
        
        # Same quantity with higher value should be accepted
        self.game.make_bet(3, 4, is_player_1=True)
        self.assertEqual(self.game.current_bet, (3, 4))
        self.assertFalse(self.game.round_over)
    
    def test_invalid_bet_ends_round(self):
        """Test that invalid bets properly end the round."""
        # Set initial bet
        self.game.make_bet(3, 4, is_player_1=True)
        
        # Invalid bet (lower quantity and value) should end round
        self.game.make_bet(2, 3, is_player_1=False)
        self.assertTrue(self.game.round_over)
        self.assertEqual(self.game.round_winner, 1)  # Player 1 wins (player 2 made invalid bet)
    
    def test_call_bluff_success(self):
        """Test successful bluff calling."""
        # Set bet that cannot be met by actual dice
        # Hands: P1=[1,2,3], P2=[4,5,6], so no player has any 6s
        # But current bet is 2 sixes - this is impossible
        self.game.current_bet = (2, 6)
        
        result = self.game.call_bluff(is_player_1=True)
        self.assertTrue(result)  # Bluff call was successful
        self.assertTrue(self.game.round_over)
        self.assertEqual(self.game.round_winner, 1)  # Player 1 wins (called bluff successfully)
    
    def test_call_bluff_failure(self):
        """Test failed bluff calling."""
        # Set bet that can be met by actual dice
        # Hands: P1=[1,2,3], P2=[4,5,6], so we have 1 of value 1, 1 of value 2, etc.
        # Bet 1 of value 1 - this is achievable
        self.game.current_bet = (1, 1)
        
        result = self.game.call_bluff(is_player_1=True)
        self.assertFalse(result)  # Bluff call failed
        self.assertTrue(self.game.round_over)
        self.assertEqual(self.game.round_winner, 2)  # Player 2 wins (bluff call failed)
    
    def test_call_bluff_no_bet_raises_error(self):
        """Test that calling bluff with no current bet raises ValueError."""
        with self.assertRaises(ValueError):
            self.game.call_bluff(is_player_1=True)
    
    def test_reroll_hands(self):
        """Test that reroll_hands generates new random hands."""
        original_p1_hand = self.game.p1_hand.copy()
        original_p2_hand = self.game.p2_hand.copy()
        
        # Reroll multiple times to ensure randomness
        hands_changed = False
        for _ in range(10):  # Try multiple times since there's a small chance hands stay the same
            self.game.reroll_hands()
            if (self.game.p1_hand != original_p1_hand or 
                self.game.p2_hand != original_p2_hand):
                hands_changed = True
                break
        
        self.assertTrue(hands_changed, "Hands should change after rerolling")
        self.assertEqual(len(self.game.p1_hand), 3)
        self.assertEqual(len(self.game.p2_hand), 3)
        
        # Check all dice values are valid
        for die in self.game.p1_hand + self.game.p2_hand:
            self.assertIn(die, [1, 2, 3, 4, 5, 6])
    
    def test_initial_game_state(self):
        """Test initial game state is correct."""
        self.assertIsNone(self.game.current_bet)
        self.assertEqual(self.game.bet_history, [])
        self.assertFalse(self.game.round_over)
        self.assertIsNone(self.game.round_winner)
        self.assertEqual(len(self.game.p1_hand), 3)
        self.assertEqual(len(self.game.p2_hand), 3)


if __name__ == '__main__':
    unittest.main()