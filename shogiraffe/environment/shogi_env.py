import pathlib, os, time

import pygame
import gym
import numpy as np
import shogi


# Function that transforms a Counter object to its List counterpart
def counter_to_list(a):
    l = []
    for i in a:
        l += [i] * a[i]
    return l


X_OFFSET = 90
Y_OFFSET = 120
TILE_SIZE = 80

FILES = {
    'p': 'pawn',
    'l': 'lance',
    'n': 'knight',
    's': 'silver',
    'g': 'gold',
    'b': 'bishop',
    'r': 'rook',
    'k': 'king',
}

PIECE_SYMBOLS = {
    1: 'p',
    2: 'l',
    3: 'n',
    4: 's',
    5: 'g',
    6: 'b',
    7: 'r',
    8: 'k',
}

class ShogiEnv(gym.Env):
    """
    A reinforcement learning environment that simulates a shogi board.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self):
        self.board = shogi.Board()
        self.window = None
        self.clock = None

    def step(self, action):
        """
        Performs an action and modify the state of the board.

        Returns:
            shogi.Board : resulting board after the move
            numpy.ndarray : observation vector of shape 784 (see description in board_to_vec)
            float : reward (1 if victory, -1 if defeat, 0 otherwise)
        """
        self.board.push(action)
        observation = self.board
        reward = (-1 if self.board.turn == 0 else 1) if self.board.is_game_over() else 0
        done = reward != 0
        return observation, reward, done, None

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.board = shogi.Board()
        return self.board

    def render(self, mode="human"):
        """
        Renders the current game in a pygame window.

        Args:
            mode (string) : 'human' or 'rgb_array' depending on whether you want to display it or not.

        Returns:
            None if mode is 'human', RGB array of the board if mode is 'rgb_array'
        """
        start_time = time.time()
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((9 * TILE_SIZE + 2 * X_OFFSET, 9 * TILE_SIZE + 2 * Y_OFFSET))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((9 * TILE_SIZE + 2 * X_OFFSET, 9 * TILE_SIZE + 2 * Y_OFFSET))
        canvas.fill((255, 255, 255))

        # Draw the board
        for i in range(0, 10):
            # Horizontal lines
            pygame.draw.line(
                canvas,
                0,
                (X_OFFSET, TILE_SIZE * i + Y_OFFSET),
                (X_OFFSET + TILE_SIZE * 9, TILE_SIZE * i + Y_OFFSET),
                width=2,
            )
            # Vertical lines
            pygame.draw.line(
                canvas,
                0,
                (TILE_SIZE * i + X_OFFSET, Y_OFFSET),
                (TILE_SIZE * i + X_OFFSET, Y_OFFSET + 9 * TILE_SIZE),
                width=2,
            )

        # For each case, draw the piece that goes on it
        for line in range(9):
            for column in range(9):
                piece = self.board.piece_at(line * 9 + column)

                # If there's a piece on this case
                if piece is not None:
                    x = X_OFFSET + TILE_SIZE * column + int((480 - 403)/480/2 * TILE_SIZE) + 0.1 * TILE_SIZE
                    y = Y_OFFSET + TILE_SIZE * line + 0.1 * TILE_SIZE

                    # Draw it in red if selected, black otherwise
                    filename = pathlib.Path(os.path.abspath(__file__)).parents[0] / "images" / (("promoted_" if piece.is_promoted() else "") + FILES[piece.symbol()[-1].lower()] + ".png")
                    piece_image = pygame.image.load(filename)
                    piece_image = pygame.transform.scale(piece_image, (int(0.8 * TILE_SIZE * 403/480), 0.8 * TILE_SIZE))
                    if piece.color == 1:
                        piece_image = pygame.transform.rotate(piece_image, 180)
                    canvas.blit(piece_image, (x, y))

        # Draw the pieces in human's hands
        hand = counter_to_list(self.board.pieces_in_hand[0])
        for index, piece_number in enumerate(hand):
            x = X_OFFSET + index * TILE_SIZE  + 0.1 * TILE_SIZE + int((480 - 403)/480/2 * TILE_SIZE)
            y = (3 * Y_OFFSET - TILE_SIZE) / 2 + 9 * TILE_SIZE  + 0.1 * TILE_SIZE
            filename = pathlib.Path(os.path.abspath(__file__)).parents[0] / "images" / (FILES[PIECE_SYMBOLS[piece_number]] + ".png")
            piece_image = pygame.image.load(filename)
            piece_image = pygame.transform.scale(piece_image, (int(0.8 * TILE_SIZE * 403 / 480), 0.8 * TILE_SIZE))
            canvas.blit(piece_image, (x, y))

        # Draw the pieces in IA's hands
        hand = counter_to_list(self.board.pieces_in_hand[0])
        for index, piece_number in enumerate(hand):
            x = X_OFFSET + index * TILE_SIZE  + 0.1 * TILE_SIZE + int((480 - 403)/480/2 * TILE_SIZE)
            y = (Y_OFFSET - TILE_SIZE) / 2  + 0.1 * TILE_SIZE
            filename = pathlib.Path(os.path.abspath(__file__)).parents[0] / "images" / (FILES[PIECE_SYMBOLS[piece_number]] + ".png")
            piece_image = pygame.image.load(filename)
            piece_image = pygame.transform.rotate(piece_image, 180)
            piece_image = pygame.transform.scale(piece_image, (int(0.8 * TILE_SIZE * 403 / 480), 0.8 * TILE_SIZE))
            canvas.blit(piece_image, (x, y))

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.set_caption(f'Shogi Env | Display time : {time.time() - start_time}')
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Closes the pygame window.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = ShogiEnv()
    env.render()
    import time
    time.sleep(10)