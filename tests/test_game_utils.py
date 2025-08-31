"""
Unit tests for game_utils.py functions.
Tests probability calculations and shared utility functions.
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Game_Utils import probability_r


class TestGameUtils(unittest.TestCase):
    """Test cases for game utility functions."""
    
    def test_probability_r_basic_cases(self):
        """Test probability_r function with basic known cases."""
        # Test requiring 0 successes (should be 1.0)
        self.assertAlmostEqual(probability_r(0, 5, 1/6), 1.0, places=6)
        
        # Test requiring more than possible (should be 0.0)
        self.assertAlmostEqual(probability_r(6, 5, 1/6), 0.0, places=6)
        
        # Test requiring exactly all successes
        self.assertAlmostEqual(probability_r(5, 5, 1/6), (1/6)**5, places=6)
    
    def test_probability_r_negative_required(self):
        """Test probability_r with negative required value."""
        self.assertEqual(probability_r(-1, 5, 1/6), 1.0)
        self.assertEqual(probability_r(-10, 3, 1/6), 1.0)
    
    def test_probability_r_dice_scenarios(self):
        """Test probability_r with realistic dice game scenarios."""
        # Probability of getting at least 1 six in 5 dice
        prob_at_least_1 = probability_r(1, 5, 1/6)
        self.assertGreater(prob_at_least_1, 0.5)  # Should be reasonably likely
        
        # Probability of getting at least 3 sixes in 5 dice  
        prob_at_least_3 = probability_r(3, 5, 1/6)
        self.assertLess(prob_at_least_3, 0.2)  # Should be unlikely
        
        # Probability should decrease as required increases
        prob_1 = probability_r(1, 5, 1/6)
        prob_2 = probability_r(2, 5, 1/6) 
        prob_3 = probability_r(3, 5, 1/6)
        self.assertGreater(prob_1, prob_2)
        self.assertGreater(prob_2, prob_3)
    
    def test_probability_r_edge_cases(self):
        """Test probability_r edge cases."""
        # Single die, require exactly that die
        self.assertAlmostEqual(probability_r(1, 1, 1/6), 1/6, places=6)
        
        # Zero dice (edge case)
        self.assertEqual(probability_r(0, 0, 1/6), 1.0)
        self.assertEqual(probability_r(1, 0, 1/6), 0.0)


if __name__ == '__main__':
    unittest.main()