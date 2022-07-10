import copy
import math


class MinimaxAgent:
    """
    An agent that evaluates the board with a neural network.
    Has two ways to be trained :
      - supervised learning, with Elmo as a mentor
      - reinforcement learning, with the TD-Leaf(lambda) algorithm
    """

    def __init__(self, evaluation_network, side=0, depth=3):
        """
        Args:
           evaluation_network (torch.nn.Module) : the network that will evaluate the board and serve as a way to direct the search
           side (int) : 0 or 1 depending on whether the player is black (1st player to move) or white (2nd player to move)
           depth (int) : maximum depth of the minimax search
        """
        self.evaluation_network = evaluation_network
        self.depth = depth
        self.side = side

    def evaluate(self, board):
        return self.evaluation_network(board)

    def play(self, board):
        move, eval = self.alpha_beta(self.depth, board.turn, board, -math.inf, math.inf)
        return move[0]

    def alpha_beta(self, depth, side, board, alpha, beta, register=None):
        """
        Implements the Minimax algorithm on our game, with alpha-beta pruning
        This supposes that our Agent will try to minimize the maximum gain of
        the opponent, assuming its opponent plays perfectly).

        Args:
            depth (int): the depth at which we'll be looking
            side (int, 0 or 1): the player whose turn it is to play
            board (shogi.Board): an object describing the current state of the board
            alpha (float): the alpha value of A-B algorithm
            beta (float): the beta value of A-B algorithm
            register (dict): used to store the values of already-computed boards with
                their Zobrist hash, for more efficiency //TODO : actually fill that register

        Returns:
            shogi.Move : selected move that the Agent thinks is best
            float : estimated score of the board with its children taken into account.
        """
        # Base case
        if depth == 0:
            return ([], self.evaluate(board))

        # Different cases depending on whose turn it is to play
        elif side == self.side:

            # We'll look at all of the legal moves
            tous_les_coups = [i for i in board.legal_moves]
            # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

            # If the player "0" has lost
            if len(tous_les_coups) == 0:
                return (None, -math.inf)

            max_noeuds_fils = -math.inf

            # For each move, we'll do a recursive call to alpha_beta
            for coup in tous_les_coups:
                plateau = copy.deepcopy(board)
                plateau.push(coup)
                ab_call = self.alpha_beta(depth - 1, 1, plateau, alpha, beta)
                if ab_call[1] > max_noeuds_fils:
                    max_noeuds_fils = ab_call[1]
                    coup_choisi = [coup] + ab_call[0]
                alpha = max(alpha, max_noeuds_fils)

                if alpha >= beta:
                    # No need to go deeper
                    break

            return (coup_choisi, max_noeuds_fils)

        else:
            # We'll look at all of the legal moves
            tous_les_coups = [i for i in board.legal_moves]
            # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

            # If the player "0" has won
            if len(tous_les_coups) == 0:
                return (None, +math.inf)

            min_noeuds_fils = +math.inf

            # For each move, we'll do a recursive call to alpha_beta
            for coup in tous_les_coups:
                plateau = copy.deepcopy(board)
                plateau.push(coup)
                ab_call = self.alpha_beta(depth - 1, 0, plateau, alpha, beta)
                if ab_call[1] < min_noeuds_fils:
                    min_noeuds_fils = ab_call[1]
                    coup_choisi = [coup] + ab_call[0]
                beta = min(beta, min_noeuds_fils)

                if alpha >= beta:
                    # No need to go deeper
                    break

            # This time, the opponent is playing, so we'll return the move that
            # minimizes the possible scores
            return (coup_choisi, min_noeuds_fils)
