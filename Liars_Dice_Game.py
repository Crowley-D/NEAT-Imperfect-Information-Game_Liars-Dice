import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import random
from typing import List, Optional, Tuple
from Game_Constants import *

pygame.init()


class Game:
    BET_FONT = pygame.font.SysFont("comicsans", 50)

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    # Load dice sprites with proper error handling
    assets_dir = os.path.join(os.path.dirname(__file__), "assets", "dice")
    dice_images = []
    missing_images = []

    for i in range(1, 7):
        try:
            image_path = os.path.join(assets_dir, f"dice_{i}.png")
            dice_images.append(pygame.image.load(image_path))
        except pygame.error as e:
            print(f"Warning: Could not load dice image {image_path}: {e}")
            missing_images.append(i)
            dice_images.append(None)  # Append placeholder if loading fails

    # Validate that critical assets loaded
    if len(missing_images) > 0:
        print(f"Warning: Missing dice images for values: {missing_images}")
        print("Game will use text fallback for missing images.")

    def __init__(
        self,
        window: pygame.Surface,
        window_width: int,
        window_height: int,
        p1_hand: Optional[List[int]] = None,
        p2_hand: Optional[List[int]] = None,
        dice_per_player: int = DICE_PER_PLAYER_3DICE,
    ) -> None:
        """Initialize a new Liar's Dice game.

        Args:
            window: Pygame surface for display
            window_width: Width of the game window
            window_height: Height of the game window
            p1_hand: Optional predefined dice for player 1
            p2_hand: Optional predefined dice for player 2
            dice_per_player: Number of dice each player has (default: 3)
        """
        self.window_height = window_height
        self.window_width = window_width

        self.current_bet: Optional[Tuple[int, int]] = None
        self.bet_history: List[Tuple[int, int]] = []

        self.p1_dice = dice_per_player
        self.p2_dice = dice_per_player
        self.p1_hand = (
            p1_hand
            if p1_hand is not None
            else [
                random.randint(MIN_DICE_VALUE, MAX_DICE_VALUE)
                for _ in range(self.p1_dice)
            ]
        )
        self.p2_hand = (
            p2_hand
            if p2_hand is not None
            else [
                random.randint(MIN_DICE_VALUE, MAX_DICE_VALUE)
                for _ in range(self.p2_dice)
            ]
        )

        self.window = window

        self.round_over: bool = False
        self.round_winner: Optional[int] = None  # Winner of the round (1 or 2)

    def make_bet(self, quantity: int, value: int, is_player_1: bool) -> None:
        """Make a bet or validate if the bet is legal, ending round if invalid.

        Args:
            quantity: Number of dice claimed to show the face value
            value: Face value being bet on (1-6)
            is_player_1: True if player 1 is making the bet, False for player 2

        Note:
            Invalid bets (not higher than current bet) end the round with the
            betting player losing.
        """
        if self.current_bet is not None:
            self.bet_history.append(self.current_bet)

        current_quantity, current_value = (
            self.current_bet if self.current_bet else (0, 0)
        )
        if quantity > current_quantity:
            self.current_bet = (quantity, value)
        elif value > current_value and quantity >= current_quantity:
            self.current_bet = (quantity, value)
        else:
            if is_player_1:
                self.round_winner = 2  # Opponent wins.
            else:
                self.round_winner = 1  # Opponent wins.

                # Instead of removing dice, we simply reroll hands.
            self.reroll_hands()

            # Reset current bet and mark the round as over.
            self.current_bet = None
            self.round_over = True

    def call_bluff(self, is_player_1: bool) -> bool:
        """Call bluff on current bet and determine round winner.

        Args:
            is_player_1: True if player 1 is calling bluff, False for player 2

        Returns:
            True if bluff call was successful (actual count < bet quantity), False otherwise

        Raises:
            ValueError: If no current bet exists to call bluff on
        """
        if not self.current_bet:
            raise ValueError("No bet to call bluff on.")
        actual_count = sum(
            (
                self.p1_hand.count(self.current_bet[1]),
                self.p2_hand.count(self.current_bet[1]),
            )
        )
        call_success = actual_count < self.current_bet[0]
        # Determine the round winner based on who called bluff and whether the call was correct.
        # If player 1 is the caller:
        if is_player_1:
            if call_success:
                self.round_winner = 1  # Caller (Player 1) wins.
            else:
                self.round_winner = 2  # Opponent wins.
        else:
            if call_success:
                self.round_winner = 2  # Caller (Player 2) wins.
            else:
                self.round_winner = 1  # Opponent wins.

        # Instead of removing dice, we simply reroll hands.
        self.reroll_hands()

        # Reset current bet and mark the round as over.
        self.current_bet = None
        self.round_over = True
        return call_success

    def reroll_hands(self) -> None:
        """Generate new random hands for both players after a round ends."""
        self.p1_hand = [
            random.randint(MIN_DICE_VALUE, MAX_DICE_VALUE) for _ in range(self.p1_dice)
        ]
        self.p2_hand = [
            random.randint(MIN_DICE_VALUE, MAX_DICE_VALUE) for _ in range(self.p2_dice)
        ]

    def draw(self, show_both_hands: bool = True) -> None:
        """Draw the complete game state to the window.

        Args:
            show_both_hands: If True, show both players' dice. If False, only show player 1's dice.
        """
        self.window.fill(self.BLACK)
        self.draw_bet()
        self.render_hands(show_both_hands)

    def render_hands(self, show_both_hands: bool) -> None:
        """Render dice hands for players on the game window.

        Args:
            show_both_hands: If True, render both players' hands. If False, only render player 1's hand.
        """
        y_offset = PLAYER_1_Y_BASE
        # Render Player 1's hand.
        text_surface = self.BET_FONT.render("Player_1", True, self.WHITE)
        text_rect = text_surface.get_rect(
            center=(self.window_width // 2, y_offset - PLAYER_LABEL_Y_OFFSET)
        )
        self.window.blit(text_surface, text_rect)

        x_offset = (
            self.window_width // 2
            - (self.p1_dice * DICE_SIZE_OFFSET)
            - DICE_CENTER_OFFSET
        )
        for die in self.p1_hand:
            if die <= len(self.dice_images) and self.dice_images[die - 1]:
                self.window.blit(self.dice_images[die - 1], (x_offset, y_offset))
            else:
                # Fallback to text rendering for missing or invalid dice images
                self.draw_text(f"[{die}]", x_offset, y_offset, self.RED)
            x_offset += DICE_X_SPACING
        y_offset += PLAYER_2_Y_SPACING

        # Render Player 2's hand if show_both_hands is True.
        if show_both_hands:
            text_surface = self.BET_FONT.render("Player_2", True, self.WHITE)
            text_rect = text_surface.get_rect(
                center=(self.window_width // 2, y_offset - PLAYER_LABEL_Y_OFFSET)
            )
            self.window.blit(text_surface, text_rect)

            x_offset = (
                self.window_width // 2
                - (self.p2_dice * DICE_SIZE_OFFSET)
                - DICE_CENTER_OFFSET
            )
            for die in self.p2_hand:
                if die <= len(self.dice_images) and self.dice_images[die - 1]:
                    self.window.blit(self.dice_images[die - 1], (x_offset, y_offset))
                else:
                    # Fallback to text rendering for missing or invalid dice images
                    self.draw_text(f"[{die}]", x_offset, y_offset, self.RED)
                x_offset += DICE_X_SPACING
            y_offset += 400

    def draw_bet(self) -> None:
        """Draw the current bet information on the game window."""
        if self.current_bet:
            bet_text = f"Current Bid: {self.current_bet[0]} x"
            bet_surface = self.BET_FONT.render(bet_text, True, self.WHITE)
            bet_rect = bet_surface.get_rect(
                center=(
                    self.window_width // 2 - BET_DISPLAY_X_OFFSET,
                    self.window_height // 2,
                )
            )
            self.window.blit(bet_surface, bet_rect)

            face_value = self.current_bet[1]
            if face_value < 1:
                face_value = 1
            elif face_value > 6:
                face_value = 6

            if face_value <= len(self.dice_images) and self.dice_images[face_value - 1]:
                die_image = self.dice_images[face_value - 1]
                self.window.blit(
                    die_image,
                    (
                        bet_rect.right + BET_DIE_SPACING,
                        self.window_height // 2 - die_image.get_height() // 2,
                    ),
                )
            else:
                # Fallback for missing bet display image
                self.draw_text(
                    f"[{face_value}]",
                    bet_rect.right + BET_DIE_SPACING,
                    self.window_height // 2 - 25,
                    self.WHITE,
                )
        else:
            bet_surface = self.BET_FONT.render("Current Bid:", True, self.WHITE)
            bet_rect = bet_surface.get_rect(
                center=(self.window_width // 2, self.window_height // 2)
            )
            self.window.blit(bet_surface, bet_rect)

        pygame.display.flip()

    def draw_text(
        self, text: str, x: int, y: int, color: Tuple[int, int, int] = WHITE
    ) -> None:
        """Draw text at specified coordinates with given color.

        Args:
            text: Text string to render
            x: X coordinate for text placement
            y: Y coordinate for text placement
            color: RGB color tuple for text color
        """
        label = self.BET_FONT.render(text, True, color)
        self.window.blit(label, (x, y))
