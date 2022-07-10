import random


class RandomAgent:
    """
    An agent that evaluates the board with a neural network.
    Has two ways to be trained :
      - supervised learning, with Elmo as a mentor
      - reinforcement learning, with the TD-Leaf(lambda) algorithm
    """

    def __init__(self):
        pass

    def play(self, board):
        return random.choice(list(board.legal_moves))
