import copy
import math
import os
import random
import time
from threading import Thread

import shogi
import tensorflow as tf

from shogiraffe.strategy.agents.neural_network import board2vec


def simple_heuristic(board, liste_de_coups):
    """
    A basic heuristic that only classes promotions and captures in front of
    all other moves.
    Parameters :
      - board : a shogi.Board object
      - liste_de_coups : a list of shogi.Move objects
    """

    parachutage = []
    prises = []
    promotion = []
    prises_promotion = []
    autres = []

    for coup in liste_de_coups:
        # if str(coup)[1] == '*':
        # parachutage
        #    parachutage.append(coup)
        if board.piece_at(coup.to_square) != None:
            # prise
            if coup.promotion:
                prises_promotion.append(coup)
            else:
                prises.append(coup)
        elif coup.promotion:
            # promotion
            promotion.append(coup)
        else:
            autres.append(coup)

    return prises_promotion + prises + promotion + autres


def evaluation_based_heuristic(board, liste_de_coups, agent, premier_joueur, aggressive_pruning=False):
    """
    A new heuristic that evaluates the boards and sorts them in ascending/
    descending order of their evaluation funcitons.
    Parameters :
      - board : a shogi.Board object
      - liste_de_coups : a list of shogi.Move objects
      - agent : the agent to use for evaluation
      - premier_joueur : the first player to move (1:IA or 0:human)
    """

    scores = []
    boards = []

    # For each move, play the move and add the move to the list
    for coup in liste_de_coups:
        b = copy.deepcopy(board)
        b.push(coup)
        boards.append(b)

    # Evaluate all the boards at once for faster results
    eval = agent.evaluate(boards)

    for i in range(len(liste_de_coups)):
        scores.append((eval[i][0], random.random(), liste_de_coups[i]))

    # Sort the list in ascending order
    scores.sort()

    # If the player is the human, reverse
    if premier_joueur == 0:
        scores.reverse()

    somme_eval = sum(c[0] for c in scores)

    # If we have enabled aggressive pruning :
    if aggressive_pruning:
        # Cut all nodes whose evaluation is under average
        scores = [c for c in scores if c[0] > somme_eval / len(scores)]

    return [(c[2], c[0], somme_eval) for c in scores]


def minimax(profondeur, premier_joueur, board):
    """
    Implements the Minimax algorithm on our Shogi board.
    This supposes that our AI player will try to minimize the maximum gain of
    the opponent (it assumes its opponent plays perfectly).

    Parameters:
    profondeur (int): the depth at which we'll be looking
    premier_joueur (int, 0 or 1): the player whose turn it is to play
    board (shogi.Board): an object describing the current state of the board

    Returns:
    score (float): for now it only returns the score of the Board with its children
      taken into account, but //TODO : make it return the move as well, just like
      alpha-beta does at the moment.
    """

    # Base case
    if profondeur == 0:
        return evaluate(board)

    # Different cases depending on whose turn it is to play
    elif premier_joueur == 0:

        # We'll look at all of the legal moves
        tous_les_coups = [i for i in board.legal_moves]

        # If the player "0" has lost
        if len(tous_les_coups) == 0:
            return -math.inf

        max_noeuds_fils = -math.inf

        # For each move, we'll do a recursive call to minimax
        for coup in tous_les_coups:
            plateau = copy.deepcopy(board)
            plateau.push(coup)
            max_noeuds_fils = max(max_noeuds_fils, minimax(profondeur - 1, 1, plateau))

        # And return the move that maximizes the score
        return max_noeuds_fils

    else:

        # We'll look at all of the legal moves
        tous_les_coups = [i for i in board.legal_moves]

        # If the player "0" has won
        if len(tous_les_coups) == 0:
            return +math.inf

        min_noeuds_fils = +math.inf

        # For each move, we'll do a recursive call to minimax
        for coup in tous_les_coups:
            plateau = board
            plateau.push(coup)
            min_noeuds_fils = min(min_noeuds_fils, minimax(profondeur - 1, 0, plateau))

        # This time, the opponent is playing, so we'll return the move that
        # minimizes the possible scores
        return min_noeuds_fils


def alpha_beta(profondeur, premier_joueur, board, alpha, beta, agent, registre=None):
    """
    Implements the Minimax algorithm on our Shogi board.
    This supposes that our AI player will try to minimize the maximum gain of
    the opponent (it assumes its opponent plays perfectly).

    Parameters:
    profondeur (int): the depth at which we'll be looking
    premier_joueur (int, 0 or 1): the player whose turn it is to play
    board (shogi.Board): an object describing the current state of the board
    alpha (float): the alpha value of A-B algorithm
    beta (float): the beta value of A-B algorithm
    registre (dict): used to store the values of already-computed boards with
      their Zobrist hash, for more efficiency //TODO : actually fill that register

    Returns:
    (coup_choisi, score) where coup_choisi is type shogi.Move and represents the
    move that the AI thinks is best, and score is type float and represents the
    estimated score of the board with its children taken into account.
    """
    # Base case
    if profondeur == 0:
        return ([], evaluate(board, agent))

    # Different cases depending on whose turn it is to play
    elif premier_joueur == 0:

        # We'll look at all of the legal moves
        tous_les_coups = [i for i in board.legal_moves]
        tous_les_coups = evaluation_based_heuristic(board, tous_les_coups)
        # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

        # If the player "0" has lost
        if len(tous_les_coups) == 0:
            return (None, -math.inf)

        max_noeuds_fils = -math.inf

        # For each move, we'll do a recursive call to alpha_beta
        for coup in tous_les_coups:
            plateau = copy.deepcopy(board)
            plateau.push(coup)
            ab_call = alpha_beta(profondeur - 1, 1, plateau, alpha, beta)
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
        tous_les_coups = evaluation_based_heuristic(board, tous_les_coups)
        # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

        # If the player "0" has won
        if len(tous_les_coups) == 0:
            return (None, +math.inf)

        min_noeuds_fils = +math.inf

        # For each move, we'll do a recursive call to alpha_beta
        for coup in tous_les_coups:
            plateau = copy.deepcopy(board)
            plateau.push(coup)
            ab_call = alpha_beta(profondeur - 1, 0, plateau, alpha, beta)
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


def alpha_beta_probabiliste(
    profondeur,
    premier_joueur,
    board,
    alpha,
    beta,
    proba,
    proba_seuil,
    agent,
    verbose=True,
    registre=None,
):
    """
    Implements the Minimax algorithm on our Shogi board.
    This supposes that our AI player will try to minimize the maximum gain of
    the opponent (it assumes its opponent plays perfectly).

    Parameters:
    profondeur (int): the depth at which we'll be looking
    premier_joueur (int, 0 or 1): the player whose turn it is to play
    board (shogi.Board): an object describing the current state of the board
    alpha (float): the alpha value of A-B algorithm
    beta (float): the beta value of A-B algorithm
    proba (float): the probability of the given node
    proba_seuil (float): the cutoff probability
    registre (dict): used to store the values of already-computed boards with
      their Zobrist hash, for more efficiency

    Returns:
    (coup_choisi, score) where coup_choisi is type shogi.Move and represents the
    move that the AI thinks is best, and score is type float and represents the
    estimated score of the board with its children taken into account.
    """

    # zh = board.zobrist_hash()

    # If the board has already been processed, use the result
    # if zh in registre :
    #    return registre[zh]

    # If the board has not yet been processed
    if profondeur <= 1 or proba < proba_seuil:
        if premier_joueur == 0:
            nb = 0
            tous_les_coups = [i for i in board.legal_moves]
            boards = []

            for coup in tous_les_coups:
                plateau = copy.deepcopy(board)
                plateau.push(coup)
                boards.append(plateau)

            e = agent.evaluate(boards)

            return ([], max(e)[0], len(tous_les_coups))

        elif premier_joueur == 1:
            nb = 0
            tous_les_coups = [i for i in board.legal_moves]
            boards = []

            for coup in tous_les_coups:
                plateau = copy.deepcopy(board)
                plateau.push(coup)
                boards.append(plateau)

            e = agent.evaluate(boards)

            return ([], min(e)[0], len(tous_les_coups))

    # Different cases depending on whose turn it is to play
    elif premier_joueur == 0:

        # We'll look at all of the legal moves
        tous_les_coups = [i for i in board.legal_moves]
        tous_les_coups = evaluation_based_heuristic(board, tous_les_coups, agent, premier_joueur)
        # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

        # If the player "0" has lost
        if len(tous_les_coups) == 0:
            return (None, -math.inf)

        max_noeuds_fils = -math.inf

        # For each move, we'll do a recursive call to alpha_beta_probabiliste
        nb = 0

        for (coup, eval, somme_eval) in tous_les_coups:
            # if verbose :
            #    print("Processing branch n°", nb+1, "out of", len(tous_les_coups))
            plateau = copy.deepcopy(board)
            plateau.push(coup)
            new_proba = proba * (eval / somme_eval)
            ab_call = alpha_beta_probabiliste(
                profondeur - 1,
                1,
                plateau,
                alpha,
                beta,
                new_proba,
                proba_seuil,
                agent,
                False,
            )
            nb += ab_call[2]
            if ab_call[1] > max_noeuds_fils:
                max_noeuds_fils = ab_call[1]
                coup_choisi = [coup] + ab_call[0]
            alpha = max(alpha, max_noeuds_fils)

            if alpha >= beta:
                # No need to go deeper
                break

        # registre[zh] = (coup_choisi, max_noeuds_fils, nb)
        return (coup_choisi, max_noeuds_fils, nb)

    else:

        # We'll look at all of the legal moves
        tous_les_coups = [i for i in board.legal_moves]
        tous_les_coups = evaluation_based_heuristic(board, tous_les_coups, agent, premier_joueur)
        # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

        # If the player "0" has won
        if len(tous_les_coups) == 0:
            return ([], +math.inf)

        if proba < proba_seuil:
            return ([], agent.evaluate(board))

        min_noeuds_fils = +math.inf
        nb = 0

        # For each move, we'll do a recursive call to alpha_beta
        for (coup, eval, somme_eval) in tous_les_coups:
            # if verbose :
            #    print("Processing branch n°", nb+1, "out of", len(tous_les_coups))
            plateau = copy.deepcopy(board)
            plateau.push(coup)
            new_proba = proba * (eval / somme_eval)
            ab_call = alpha_beta_probabiliste(
                profondeur - 1,
                0,
                plateau,
                alpha,
                beta,
                new_proba,
                proba_seuil,
                agent,
                False,
            )
            nb += ab_call[2]
            if ab_call[1] < min_noeuds_fils:
                min_noeuds_fils = ab_call[1]
                coup_choisi = [coup] + ab_call[0]
            beta = min(beta, min_noeuds_fils)

            if alpha >= beta:
                # No need to go deeper
                break

        # This time, the opponent is playing, so we'll return the move that
        # minimizes the possible scores
        # registre[zh] = (coup_choisi, min_noeuds_fils, nb)
        return (coup_choisi, min_noeuds_fils, nb)


class Parallel(Thread):
    """
    A thread that will run alpha-beta
    """

    def __init__(
        self,
        profondeur,
        premier_joueur,
        board,
        alpha,
        beta,
        proba,
        proba_seuil,
        model,
        registre=None,
    ):
        Thread.__init__(self)
        self.profondeur = profondeur
        self.premier_joueur = premier_joueur
        self.board = board
        self.alpha = alpha
        self.beta = beta
        self.proba = proba
        self.proba_seuil = proba_seuil
        self.model = model
        self.registre = registre
        self._return = (0, 0, 0)
        # print("Runing thread with board", board.sfen())

    def run(self):
        """
        Implements the Minimax algorithm on our Shogi board.
        This supposes that our AI player will try to minimize the maximum gain of
        the opponent (it assumes its opponent plays perfectly).

        Parameters:
        profondeur (int): the depth at which we'll be looking
        premier_joueur (int, 0 or 1): the player whose turn it is to play
        board (shogi.Board): an object describing the current state of the board
        alpha (float): the alpha value of A-B algorithm
        beta (float): the beta value of A-B algorithm
        proba (float): the probability of the given node
        proba_seuil (float): the cutoff probability
        registre (dict): used to store the values of already-computed boards with
        their Zobrist hash, for more efficiency

        Returns:
        (coup_choisi, score) where coup_choisi is type shogi.Move and represents the
        move that the AI thinks is best, and score is type float and represents the
        estimated score of the board with its children taken into account.
        """

        # zh = board.zobrist_hash()

        # If the board has already been processed, use the result
        # if zh in registre :
        #    return registre[zh]

        # If the board has not yet been processed
        if self.profondeur <= 1 or self.proba < self.proba_seuil:
            if self.premier_joueur == 0:
                nb = 0
                tous_les_coups = [i for i in self.board.legal_moves]
                boards = []

                for coup in tous_les_coups:
                    plateau = copy.deepcopy(self.board)
                    plateau.push(coup)
                    boards.append(plateau)

                e = evaluate(boards, self.model)

                self._return = ([], max(e), len(tous_les_coups))
                return

            elif self.premier_joueur == 1:
                nb = 0
                tous_les_coups = [i for i in self.board.legal_moves]
                boards = []

                for coup in tous_les_coups:
                    plateau = copy.deepcopy(self.board)
                    plateau.push(coup)
                    boards.append(plateau)

                e = evaluate(boards, self.model)

                self._return = ([], min(e), len(tous_les_coups))
                return

        # Different cases depending on whose turn it is to play
        elif self.premier_joueur == 0:

            # We'll look at all of the legal moves
            tous_les_coups = [i for i in self.board.legal_moves]
            tous_les_coups = evaluation_based_heuristic(self.board, tous_les_coups, model, premier_joueur)
            # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

            # If the player "0" has lost
            if len(tous_les_coups) == 0:
                self._return = (None, -math.inf)
                return

            max_noeuds_fils = -math.inf
            nb = 0

            # We'll keep a list of our threads here
            threads = []

            # For each move, we'll do a recursive call to alpha_beta
            for (coup, eval, somme_eval) in tous_les_coups:

                if coup == tous_les_coups[0]:

                    # First node of the tree, we do it before the others
                    plateau = copy.deepcopy(self.board)
                    plateau.push(coup)
                    new_proba = self.proba / len(tous_les_coups)

                    # Create and start a new thread
                    ab_call = Parallel(
                        self.profondeur - 1,
                        1,
                        plateau,
                        self.alpha,
                        self.beta,
                        new_proba,
                        self.proba_seuil,
                        self.model,
                    )
                    ab_call.start()

                    # Wait for the result
                    result = ab_call.join()

                    # Process the result
                    nb += result[2]
                    if result[1] > max_noeuds_fils:
                        max_noeuds_fils = result[1]
                        coup_choisi = [coup] + result[0]
                    self.alpha = max(self.alpha, max_noeuds_fils)

                    if self.alpha >= self.beta:
                        # No need to go deeper
                        break

                else:

                    # For every other node, we'll launch the threads in parallel
                    plateau = copy.deepcopy(self.board)
                    plateau.push(coup)
                    new_proba = self.proba / len(tous_les_coups)

                    # Create and start the new thread
                    ab_call = Parallel(
                        self.profondeur - 1,
                        1,
                        plateau,
                        self.alpha,
                        self.beta,
                        new_proba,
                        self.proba_seuil,
                        self.model,
                    )
                    ab_call.start()
                    threads.append(ab_call)

            for thread in threads:
                results = thread.join()
                nb += results[2]
                if results[1] > max_noeuds_fils:
                    max_noeuds_fils = results[1]
                    coup_choisi = [coup] + results[0]
                self.alpha = max(self.alpha, max_noeuds_fils)

                if self.alpha >= self.beta:
                    # No need to go deeper
                    break

            # registre[zh] = (coup_choisi, max_noeuds_fils, nb)
            self._return = (coup_choisi, max_noeuds_fils, nb)
            return

        else:

            # We'll look at all of the legal moves
            tous_les_coups = [i for i in self.board.legal_moves]
            tous_les_coups = evaluation_based_heuristic(self.board, tous_les_coups, model, premier_joueur)
            # tous_les_coups = tous_les_coups[0:min(len(tous_les_coups), 30)]

            # If the player "0" has won
            if len(tous_les_coups) == 0:
                self._return = ([], +math.inf)
                return

            min_noeuds_fils = +math.inf
            nb = 0

            # We'll keep a list of our threads here
            threads = []

            # For each move, we'll do a recursive call to alpha_beta
            for (coup, eval, somme_eval) in tous_les_coups:

                if coup == tous_les_coups[0]:

                    # First node of the tree, we do it before the others
                    plateau = copy.deepcopy(self.board)
                    plateau.push(coup)
                    new_proba = proba * (eval / somme_eval)

                    # Create and start a new thread
                    ab_call = Parallel(
                        self.profondeur - 1,
                        0,
                        plateau,
                        self.alpha,
                        self.beta,
                        new_proba,
                        self.proba_seuil,
                        self.model,
                    )
                    ab_call.start()

                    # Wait for the result
                    result = ab_call.join()

                    # Process the result
                    nb += result[2]
                    if result[1] < min_noeuds_fils:
                        min_noeuds_fils = result[1]
                        coup_choisi = [coup] + result[0]
                    self.beta = min(self.beta, min_noeuds_fils)

                    if self.alpha >= self.beta:
                        # No need to go deeper
                        break

                else:

                    # For every other node, we'll launch the threads in parallel
                    plateau = copy.deepcopy(self.board)
                    plateau.push(coup)
                    new_proba = self.proba / len(tous_les_coups)

                    # Create and start the new thread
                    ab_call = Parallel(
                        self.profondeur - 1,
                        0,
                        plateau,
                        self.alpha,
                        self.beta,
                        new_proba,
                        self.proba_seuil,
                        self.model,
                    )
                    ab_call.start()
                    threads.append(ab_call)

            for thread in threads:
                results = thread.join()
                nb += results[2]
                if results[1] < min_noeuds_fils:
                    min_noeuds_fils = results[1]
                    coup_choisi = [coup] + results[0]
                self.beta = min(self.beta, min_noeuds_fils)

                if self.alpha >= self.beta:
                    # No need to go deeper
                    break

            # This time, the opponent is playing, so we'll return the move that
            # minimizes the possible scores
            # registre[zh] = (coup_choisi, min_noeuds_fils, nb)
            self._return = (coup_choisi, min_noeuds_fils, nb)
            return

    def join(self, *args):
        Thread.join(self, *args)
        # print("Returning from whatever thread with value", self._return)
        return self._return


def parallel_alpha_beta_probabiliste(
    profondeur,
    premier_joueur,
    board,
    alpha,
    beta,
    proba,
    proba_seuil,
    model,
    registre=None,
):
    """
    Starts a thread for probabilistic alpha-beta in parallel computing.
    """

    p = Parallel(
        profondeur,
        premier_joueur,
        board,
        alpha,
        beta,
        proba,
        proba_seuil,
        model,
        registre=None,
    )
    p.start()
    return p.join()
