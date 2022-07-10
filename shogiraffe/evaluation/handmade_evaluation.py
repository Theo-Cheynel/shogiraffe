# Imports
import shogi

# Constants
PIECE_VALUE = {
    'p': 1,
    'l': 4,
    'n': 4,
    's': 6,
    'g': 7,
    'k': 0,
    'b': 11,
    'r': 11,
    'P': -1,
    'L': -4,
    'N': -4,
    'S': -6,
    'G': -7,
    'K': 0,
    'B': -11,
    'R': -11,
}

PIECE_SYMBOLS = {
    1 : 'p',
    2 : 'l',
    3 : 'n',
    4 : 's',
    5 : 'g',
    6 : 'b',
    7 : 'r',
    8 : 'k',
}

PROMOTED_VALUE = {
    'l' : 2,
    'n' : 2,
    's' : 1,
    'b' : 5,
    'r' : 7,
    'L' : 2,
    'N' : 2,
    'S' : 1,
    'B' : 5,
    'R' : 7,
}


def around(piece):
    """
    Evaluate the different valid squares around a given piece

    :param board: (shogi.Board): the board to evaluate
    :param piece: (int): the number of the square whose surroundings we're evaluating
    :return surroundings: (list): the squares around the piece
    """

    voisins = [piece - 10, piece - 9, piece - 8, piece - 1, piece + 1, piece + 8, piece + 9, piece + 10]
    if piece == 0:
        return([1, 9, 10])
    if piece == 8:
        return([7, 16, 17])
    if piece == 72:
        return([63, 64, 73])
    if piece == 80:
        return([70, 71, 79])
    if piece < 8:
        return(voisins[3:])
    if piece > 72:
        return(voisins[:5])
    if piece % 9 == 0:
        return([piece - 9, piece - 8, piece + 1, piece + 9, piece + 10])
    if piece % 9 == 8:
        return([piece - 10, piece - 9, piece - 1, piece + 8, piece + 9])
    return(voisins)

# table_neighbours will contain the list of the valid square surrounding one
table_neighbours = []
for i in range(80):
    table_neighbours.append(around(i))


class HandmadeEvaluator:
    def __init__(self):
        pass

    def __call__(self, board):
        """
        Implements a custom board evaluation. It takes account of :
          - the safety of the king
          - the liberties around the king
          - the material advantage (on board, in hand, promoted or not)
          - the positional advantage (efficiency of rook and bishop)

        :param board: (shogi.Board): the board to evaluate
        :return score: (float): the score of the board
        """

        difference_on_board = 0
        in_hand_white = 0
        in_hand_black = 0
        additional_promoted_white = 0
        additional_promoted_black = 0
        rook_efficiency_white = 0
        bishop_efficiency_white = 0
        rook_efficiency_black = 0
        bishop_efficiency_black = 0
        is_check_white = 1 if board.is_check() and board.turn == 1 else 0
        is_checkmate_white = 1 if board.is_checkmate() and board.turn == 1 else 0
        is_check_black = 1 if board.is_check() and board.turn == 0 else 0
        is_checkmate_black = 1 if board.is_checkmate() and board.turn == 0 else 0

        # We'll be scanning each square on the board
        for i in range(81):

            piece = board.piece_at(i)

            # If there's a piece :
            if piece != None :

                # Change the scoring difference :
                difference_on_board += PIECE_VALUE[piece.symbol()[-1]]

                # White's promoted pieces
                if piece.is_promoted() and piece.color==0 :

                    if piece.symbol()[-1] == "P" :
                        if i//9 == 0 :
                            additional_promoted_white += 2
                        if i//9 == 1 :
                            additional_promoted_white += 3
                        else :
                            additional_promoted_white += 5

                    else :
                        additional_promoted_white += PROMOTED_VALUE[piece.symbol()[-1]]


                # Black's promoted pieces
                if piece.is_promoted() and piece.color==1 :

                    if piece.symbol()[-1] == "p" :
                        if i//9 == 0 :
                            additional_promoted_black += 2
                        if i//9 == 1 :
                            additional_promoted_black += 3
                        else :
                            additional_promoted_black += 5

                    else :
                        additional_promoted_black += PROMOTED_VALUE[piece.symbol()[-1]]

                # If it's a white Rook
                if piece.symbol()[-1] == 'R' :

                    # Check how far you can go left :
                    j = 0
                    for j in range(1,i%9+1) :
                        if board.piece_at(i-j) != None:
                            j-=1-1*board.piece_at(i-j).color
                            break
                    rook_efficiency_white+=j

                    # Check how far you can go right :
                    j = 0
                    for j in range(1, 9-i%9) :
                        if board.piece_at(i+j) != None:
                            j-=1-1*board.piece_at(i+j).color
                            break
                    rook_efficiency_white+=j

                    # Check how far you can go up :
                    j = 0
                    for j in range(1,i//9+1) :
                        if board.piece_at(i-9*j) != None:
                            j-=1-1*board.piece_at(i-9*j).color
                            break
                    rook_efficiency_white+=j

                    # Check how far you can go down :
                    j = 0
                    for j in range(1,9-i//9) :
                        if board.piece_at(i+9*j) != None:
                            j-=1-1*board.piece_at(i+9*j).color
                            break
                    rook_efficiency_white+=j

                # If it's a white Bishop
                if piece.symbol()[-1] == 'B' :
                    # Check how far you can go left and up :
                    j = 0
                    for j in range(1,min(i%9+1, i//9+1)) :
                        if board.piece_at(i-10*j) != None:
                            j-=1-1*board.piece_at(i-10*j).color
                            break
                    bishop_efficiency_white+=j

                    # Check how far you can go right and down :
                    j = 0
                    for j in range(1, min(9-i%9-1, 9-i//9-1)) :
                        if board.piece_at(i+10*j) != None:
                            j-=1-1*board.piece_at(i+10*j).color
                            break
                    bishop_efficiency_white+=j

                    # Check how far you can go right and up :
                    j = 0
                    for j in range(1,min(9-i%9-1,i//9+1)) :
                        if board.piece_at(i-8*j) != None:
                            j-=1-1*board.piece_at(i-8*j).color
                            break
                    bishop_efficiency_white+=j

                    # Check how far you can go left and down :
                    j = 0
                    for j in range(1,min(i%9+1, 9-i//9-1)) :
                        if board.piece_at(i+8*j) != None:
                            j-=1-1*board.piece_at(i+8*j).color
                            break
                    bishop_efficiency_white+=j


                # If it's a white Rook
                if piece.symbol()[-1] == 'r' :

                    # Check how far you can go left :
                    j = 0
                    for j in range(1,i%9+1) :
                        if board.piece_at(i-j) != None:
                            j-=1*board.piece_at(i-j).color
                            break
                    rook_efficiency_black+=j

                    # Check how far you can go right :
                    j = 0
                    for j in range(1, 9-i%9) :
                        if board.piece_at(i+j) != None:
                            j-=1*board.piece_at(i+j).color
                            break
                    rook_efficiency_black+=j

                    # Check how far you can go up :
                    j = 0
                    for j in range(1,i//9+1) :
                        if board.piece_at(i-9*j) != None:
                            j-=1*board.piece_at(i-9*j).color
                            break
                    rook_efficiency_black+=j

                    # Check how far you can go down :
                    j = 0
                    for j in range(1,9-i//9) :
                        if board.piece_at(i+9*j) != None:
                            j-=1*board.piece_at(i+9*j).color
                            break
                    rook_efficiency_black+=j

                # If it's a white Bishop
                if piece.symbol()[-1] == 'b' :

                    # Check how far you can go left and up :
                    j = 0
                    for j in range(1,min(i%9+1, i//9+1)) :
                        if board.piece_at(i-10*j) != None:
                            j-=1*board.piece_at(i-10*j).color
                            break
                    bishop_efficiency_black+=j

                    # Check how far you can go right and down :
                    j = 0
                    for j in range(1, min(9-i%9-1, 9-i//9-1)) :
                        if board.piece_at(i+10*j) != None:
                            j-=1*board.piece_at(i+10*j).color
                            break
                    bishop_efficiency_black+=j

                    # Check how far you can go right and up :
                    j = 0
                    for j in range(1,min(9-i%9-1,i//9+1)) :
                        if board.piece_at(i-8*j) != None:
                            j-=1*board.piece_at(i-8*j).color
                            break
                    bishop_efficiency_black+=j

                    # Check how far you can go left and down :
                    j = 0
                    for j in range(1,min(i%9+1, 9-i//9-1)) :
                        if board.piece_at(i+8*j) != None:
                            j-=1*board.piece_at(i+8*j).color
                            break
                    bishop_efficiency_black+=j

                # Check if the square around the king are attacked
                if piece.symbol()[-1] == 'K' :
                    white_king_sec = 0
                    for j in table_neighbours[i]:
                        white_king_sec -= len(board.attackers(1,j))
                        white_king_sec += 0.75 * len(board.attackers(0,j))

                if piece.symbol()[-1] == 'k' :
                    black_king_sec = 0
                    for j in table_neighbours[i]:
                        black_king_sec -= len(board.attackers(0,j))
                        black_king_sec += 0.75 * len(board.attackers(1,j))



        pieces = board.pieces_in_hand

        # For each piece in white's hands
        for p in pieces[0] :
            in_hand_white += PIECE_VALUE[PIECE_SYMBOLS[p]] * pieces[0][p]

        # For each piece in black's hands
        for p in pieces[1] :
            in_hand_black += PIECE_VALUE[PIECE_SYMBOLS[p]] * pieces[1][p]

        x1 = difference_on_board
        x2 = in_hand_white
        x3 = in_hand_black
        x4 = additional_promoted_white
        x5 = additional_promoted_black
        x10 = rook_efficiency_white + bishop_efficiency_white
        x11 = rook_efficiency_black + bishop_efficiency_black
        x12 = white_king_sec
        x13 = black_king_sec
        #print(x12,x13)
        #print(x1, x2, x3, x4, x5, x10, x11, rook_efficiency_black, rook_efficiency_white, bishop_efficiency_black, bishop_efficiency_white)
        return -(x1 * 0.2 + (x3 - x2) * 0.3 + (x5 - x4) * 0.1 + (x11 - x10) * 0.1 + (x13 - x12) * 0.3) + is_check_white * .2 - is_check_black * .2 + is_checkmate_white * 1000 - is_checkmate_black * 1000



if __name__ == '__main__':
    eval = HandmadeEvaluator()

    sfen = 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1'
    print(shogi.Board(sfen))
    print(eval(shogi.Board(sfen)))

    sfen = 'lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2'
    print(shogi.Board(sfen))
    print(eval(shogi.Board(sfen)))

    sfen = 'lnsgkgsnl/1r5b1/pppppp1pp/5p3/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3'
    print(shogi.Board(sfen))
    print(eval(shogi.Board(sfen)))

    sfen = 'lnsgkgsnl/1r5b1/pppppp1pp/5p3/9/2PP5/PP2PPPPP/1B5R1/LNSGKGSNL w - 4'
    print(shogi.Board(sfen))
    print(eval(shogi.Board(sfen)))

    sfen = 'lnsgkgsnl/1r7/pppppp1pp/5p3/9/2Pb5/PP1PPPPPP/1B5R1/LNSGKGSNL b P 5'
    print(shogi.Board(sfen))
    print(eval(shogi.Board(sfen)))

    sfen = 'lnsgkgsnl/1r7/pppppp1pp/5p3/9/2PB5/PP1PPPPPP/7R1/LNSGKGSNL w Pb 6'
    print(shogi.Board(sfen))
    print(eval(shogi.Board(sfen)))