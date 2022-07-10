import shogi
import numpy as np
import time
import math
from numba import jit

"""
Structure of our numpy array :

[0] : side to move
[1] : turn number
##
[2:9] : nb of pawns/lance/knight/silver/gold/bishop/rook for white
[9:16] : nb of pawns/lance/knight/silver/gold/bishop/rook for black
##
[16:106] : 18*(pawn on board + pawn_color + pawn_position_x + pawn_position_y + promotion)
[106:126] : 4*(lance on board + lance_color + lance_position_x + lance_position_y + promotion)
[126:146] : 4*(knight on board + knight_color + knight_position_x + knight_position_y + promotion)
[146:166] : 4*(silver on board + silver_color + silver_position_x + silver_position_y + promotion)
[166:182] : 4*(gold on board + gold_color + gold_position_x + gold_position_y)
[182:200] : 2*(bishop on board + bishop_color + bishop_position_x + bishop_position_y + sliding_distance_leftup + sliding_distance_rightdown + sliding_distance_rightup + sliding_distance_leftdown + promotion)
[200:218] : 2*(rook on board + rook_color + rook_position_x + rook_position_y + sliding_distance_left + sliding_distance_right + sliding_distance_up + sliding_distance_down + promotion)
[218:220] : (white_king_position_x, white_king_position_y)
[220:222] : (black_king_position_x, black_king_position_y)
##
[222:240] : 18*(pawn in hand white)
[240:244] : 4*(lance in hand white)
[244:248] : 4*(knight in hand white)
[248:252] : 4*(silver in hand white)
[252:254] : 4*(gold in hand white)
[256:258] : 2*(bishop in hand white)
[258:260] : 2*(rook in hand white)
##
[260:278] : 18*(pawn in hand black)
[278:282] : 4*(lance in hand black)
[282:286] : 4*(knight in hand black)
[286:290] : 4*(silver in hand black)
[290:294] : 4*(gold in hand black)
[294:296] : 2*(bishop in hand black)
[296:298] : 2*(rook in hand black)
##
[298:541] : lowest-valued attacker of each square : (position_x, position_y, number_of_attackers)*81
[541:784] : lowest-valued attacker of each square : (position_x, position_y, number_of_defenders)*81
"""

PIECE_SYMBOLS = '.plnsgbrk'

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

table_neighbours = []
for i in range(80):
    table_neighbours.append(around(i))
# table_neighbours contains the list of the valid square surrounding one


def board2vec(board):
    """
    Computes the vector representation of the board.
    You can find a description of the format of this vector in the file
    neural_network.py.

    Args:
        board (shogi.Board): the board to vectorize

    Returns:
        np.ndarray : the vector representation of the board

    Todo:
        numba-ify this method !
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

    pawn_index = 0
    lance_index = 0
    knight_index = 0
    silver_index = 0
    gold_index = 0
    bishop_index = 0
    rook_index = 0

    vector = [0 for i in range(784)]

    vector[0] = float(board.turn)
    vector[1] = float(max(0, math.atan((board.move_number-10)/15)*2/3.142))

    # We'll be scanning each square on the board
    for i in range(81):

        case = 9*(i%9)+i//9

        piece = board.piece_at(case)

        # If there's a piece :
        if piece != None :

            if piece.color == 0 :

                if piece.symbol()[-1] == 'P':
                    vector[2] += 1/18

                    vector[16+pawn_index*5] = 1
                    vector[16+pawn_index*5+1] = 0
                    vector[16+pawn_index*5+2] = (case%9)/8
                    vector[16+pawn_index*5+3] = (case//9)/8
                    vector[16+pawn_index*5+4] = 1*piece.is_promoted()
                    pawn_index += 1


                if piece.symbol()[-1] == 'L':
                    vector[3] += 1/4

                    vector[106+lance_index*5] = 1
                    vector[106+lance_index*5+1] = 0
                    vector[106+lance_index*5+2] = (case%9)/8
                    vector[106+lance_index*5+3] = (case//9)/8
                    vector[106+lance_index*5+4] = 1*piece.is_promoted()
                    lance_index += 1


                if piece.symbol()[-1] == 'N':
                    vector[4] += 1/4

                    vector[126+knight_index*5] = 1
                    vector[126+knight_index*5+1] = 0
                    vector[126+knight_index*5+2] = (case%9)/8
                    vector[126+knight_index*5+3] = (case//9)/8
                    vector[126+knight_index*5+4] = 1*piece.is_promoted()
                    knight_index += 1

                if piece.symbol()[-1] == 'S':
                    vector[5] += 1/4

                    vector[146+silver_index*5] = 1
                    vector[146+silver_index*5+1] = 0
                    vector[146+silver_index*5+2] = (case%9)/8
                    vector[146+silver_index*5+3] = (case//9)/8
                    vector[146+silver_index*5+4] = 1*piece.is_promoted()
                    silver_index += 1


                if piece.symbol()[-1] == 'G':
                    vector[6] += 1/4

                    vector[166+gold_index*4] = 1
                    vector[166+gold_index*4+1] = 0
                    vector[166+gold_index*4+2] = (case%9)/8
                    vector[166+gold_index*4+3] = (case//9)/8
                    gold_index += 1


                if piece.symbol()[-1] == 'B':
                    vector[7] += 1/2

                    upleft = 0
                    upright = 0
                    downright = 0
                    downleft = 0

                    # Check how far you can go left and up :
                    j = 0
                    for j in range(1,min(i%9+1, i//9+1)) :
                        if board.piece_at(i-10*j) != None:
                            j-=1*board.piece_at(i-10*j).color
                            break
                    upleft+=j

                    # Check how far you can go right and down :
                    j = 0
                    for j in range(1, min(9-i%9-1, 9-i//9-1)) :
                        if board.piece_at(i+10*j) != None:
                            j-=1*board.piece_at(i+10*j).color
                            break
                    downright+=j

                    # Check how far you can go right and up :
                    j = 0
                    for j in range(1,min(9-i%9-1,i//9+1)) :
                        if board.piece_at(i-8*j) != None:
                            j-=1*board.piece_at(i-8*j).color
                            break
                    upright+=j

                    # Check how far you can go left and down :
                    j = 0
                    for j in range(1,min(i%9+1, 9-i//9-1)) :
                        if board.piece_at(i+8*j) != None:
                            j-=1*board.piece_at(i+8*j).color
                            break
                    downleft+=j

                    vector[182+bishop_index*9] = 1
                    vector[182+bishop_index*9+1] = 0
                    vector[182+bishop_index*9+2] = (case%9)/8
                    vector[182+bishop_index*9+3] = (case//9)/8
                    vector[182+bishop_index*9+4] = upleft
                    vector[182+bishop_index*9+5] = downright
                    vector[182+bishop_index*9+6] = upright
                    vector[182+bishop_index*9+7] = downleft
                    vector[182+bishop_index*9+8] = 1*piece.is_promoted()
                    bishop_index += 1


                if piece.symbol()[-1] == 'R':
                    vector[8] += 1/2

                    left = 0
                    right = 0
                    up = 0
                    down = 0

                    # Check how far you can go left :
                    j = 0
                    for j in range(1,i%9+1) :
                        if board.piece_at(i-j) != None:
                            j-=1*board.piece_at(i-j).color
                            break
                    left+=j

                    # Check how far you can go right :
                    j = 0
                    for j in range(1, 9-i%9-1) :
                        if board.piece_at(i+j) != None:
                            j-=1*board.piece_at(i+j).color
                            break
                    right+=j

                    # Check how far you can go up :
                    j = 0
                    for j in range(1,i//9+1) :
                        if board.piece_at(i-9*j) != None:
                            j-=1*board.piece_at(i-9*j).color
                            break
                    up+=j

                    # Check how far you can go down :
                    j = 0
                    for j in range(1,9-i//9-1) :
                        if board.piece_at(i+9*j) != None:
                            j-=1*board.piece_at(i+9*j).color
                            break
                    down+=j

                    vector[200+rook_index*9] = 1
                    vector[200+rook_index*9+1] = 0
                    vector[200+rook_index*9+2] = (case%9)/8
                    vector[200+rook_index*9+3] = (case//9)/8
                    vector[200+rook_index*9+4] = left
                    vector[200+rook_index*9+5] = right
                    vector[200+rook_index*9+6] = up
                    vector[200+rook_index*9+7] = down
                    vector[200+rook_index*9+8] = 1*piece.is_promoted()
                    rook_index += 1

                if piece.symbol()[-1]=='K':
                    vector[218] = (case%9)/8
                    vector[219] = (case//9)/8


            elif piece.color == 1 :

                if piece.symbol()[-1] == 'p':
                    vector[9] += 1/16

                    vector[16+pawn_index*5] = 1
                    vector[16+pawn_index*5+1] = 1
                    vector[16+pawn_index*5+2] = (case%9)/8
                    vector[16+pawn_index*5+3] = (case//9)/8
                    vector[16+pawn_index*5+4] = 1*piece.is_promoted()
                    pawn_index += 1

                if piece.symbol()[-1] == 'l':
                    vector[10] += 1/4

                    vector[106+lance_index*5] = 1
                    vector[106+lance_index*5+1] = 1
                    vector[106+lance_index*5+2] = (case%9)/8
                    vector[106+lance_index*5+3] = (case//9)/8
                    vector[106+lance_index*5+4] = 1*piece.is_promoted()
                    lance_index += 1


                if piece.symbol()[-1] == 'n':
                    vector[11] += 1/4

                    vector[126+knight_index*5] = 1
                    vector[126+knight_index*5+1] = 1
                    vector[126+knight_index*5+2] = (case%9)/8
                    vector[126+knight_index*5+3] = (case//9)/8
                    vector[126+knight_index*5+4] = 1*piece.is_promoted()
                    knight_index += 1


                if piece.symbol()[-1] == 's':
                    vector[12] += 1/4

                    vector[146+silver_index*5] = 1
                    vector[146+silver_index*5+1] = 1
                    vector[146+silver_index*5+2] = (case%9)/8
                    vector[146+silver_index*5+3] = (case//9)/8
                    vector[146+silver_index*5+4] = 1*piece.is_promoted()
                    silver_index += 1


                if piece.symbol()[-1] == 'g':
                    vector[13] += 1/4

                    vector[166+gold_index*4] = 1
                    vector[166+gold_index*4+1] = 1
                    vector[166+gold_index*4+2] = (case%9)/8
                    vector[166+gold_index*4+3] = (case//9)/8
                    gold_index += 1


                if piece.symbol()[-1] == 'b':
                    vector[14] += 1/2

                    upleft = 0
                    upright = 0
                    downright = 0
                    downleft = 0

                    # Check how far you can go left and up :
                    j = 0
                    for j in range(1,min(i%9+1, i//9+1)) :
                        if board.piece_at(i-10*j) != None:
                            j-=1-1*board.piece_at(i-10*j).color
                            break
                    upleft+=j

                    # Check how far you can go right and down :
                    j = 0
                    for j in range(1, min(9-i%9-1, 9-i//9-1)) :
                        if board.piece_at(i+10*j) != None:
                            j-=1-1*board.piece_at(i+10*j).color
                            break
                    downright+=j

                    # Check how far you can go right and up :
                    j = 0
                    for j in range(1,min(9-i%9-1,i//9+1)) :
                        if board.piece_at(i-8*j) != None:
                            j-=1-1*board.piece_at(i-8*j).color
                            break
                    upright+=j

                    # Check how far you can go left and down :
                    j = 0
                    for j in range(1,min(i%9+1, 9-i//9-1)) :
                        if board.piece_at(i+8*j) != None:
                            j-=1-1*board.piece_at(i+8*j).color
                            break
                    downleft+=j

                    vector[182+bishop_index*9] = 1
                    vector[182+bishop_index*9+1] = 1
                    vector[182+bishop_index*9+2] = (case%9)/8
                    vector[182+bishop_index*9+3] = (case//9)/8
                    vector[182+bishop_index*9+4] = upleft
                    vector[182+bishop_index*9+5] = downright
                    vector[182+bishop_index*9+6] = upright
                    vector[182+bishop_index*9+7] = downleft
                    vector[182+bishop_index*9+8] = 1*piece.is_promoted()
                    bishop_index += 1



                if piece.symbol()[-1] == 'r':
                    vector[15] += 1/2

                    left = 0
                    right = 0
                    up = 0
                    down = 0

                    # Check how far you can go left :
                    j = 0
                    for j in range(1,i%9+1) :
                        if board.piece_at(i-j) != None:
                            j-=1-1*board.piece_at(i-j).color
                            break
                    left+=j

                    # Check how far you can go right :
                    j = 0
                    for j in range(1, 9-i%9-1) :
                        if board.piece_at(i+j) != None:
                            j-=1-1*board.piece_at(i+j).color
                            break
                    right+=j

                    # Check how far you can go up :
                    j = 0
                    for j in range(1,i//9+1) :
                        if board.piece_at(i-9*j) != None:
                            j-=1-1*board.piece_at(i-9*j).color
                            break
                    up+=j

                    # Check how far you can go down :
                    j = 0
                    for j in range(1,9-i//9-1) :
                        if board.piece_at(i+9*j) != None:
                            j-=1-1*board.piece_at(i+9*j).color
                            break
                    down+=j

                    vector[200+rook_index*9] = 1
                    vector[200+rook_index*9+1] = 1
                    vector[200+rook_index*9+2] = (case%9)/8
                    vector[200+rook_index*9+3] = (case//9)/8
                    vector[200+rook_index*9+4] = left
                    vector[200+rook_index*9+5] = right
                    vector[200+rook_index*9+6] = up
                    vector[200+rook_index*9+7] = down
                    vector[200+rook_index*9+8] = 1*piece.is_promoted()
                    rook_index += 1


                if piece.symbol()[-1]=='k':
                    vector[220] = (case%9)/8
                    vector[221] = (case//9)/8



    pieces = board.pieces_in_hand

    # For each piece in white's hands

    counter_pawn = 0
    counter_lance = 0
    counter_knight = 0
    counter_silver = 0
    counter_gold = 0
    counter_bishop = 0
    counter_rook = 0

    for p in pieces[0] :
        if PIECE_SYMBOLS[p] == 'p' :
            vector[222 + counter_pawn] = 1
            counter_pawn += 1
        if PIECE_SYMBOLS[p] == 'l' :
            vector[240 + counter_lance] = 1
            counter_lance += 1
        if PIECE_SYMBOLS[p] == 'k' :
            vector[244 + counter_knight] = 1
            counter_knight += 1
        if PIECE_SYMBOLS[p] == 's' :
            vector[248 + counter_silver] = 1
            counter_silver += 1
        if PIECE_SYMBOLS[p] == 'g' :
            vector[252 + counter_gold] = 1
            counter_gold += 1
        if PIECE_SYMBOLS[p] == 'b' :
            vector[256 + counter_bishop] = 1
            counter_bishop += 1
        if PIECE_SYMBOLS[p] == 'r' :
            vector[258 + counter_rook] = 1
            counter_rook += 1


    counter_pawn = 0
    counter_lance = 0
    counter_knight = 0
    counter_silver = 0
    counter_gold = 0
    counter_bishop = 0
    counter_rook = 0


    # For each piece in black's hands
    for p in pieces[1] :
        if PIECE_SYMBOLS[p] == 'P' :
            vector[260 + counter_pawn] = 1
            counter_pawn += 1
        if PIECE_SYMBOLS[p] == 'p' :
            vector[278 + counter_lance] = 1
            counter_lance += 1
        if PIECE_SYMBOLS[p] == 'k' :
            vector[282 + counter_knight] = 1
            counter_knight += 1
        if PIECE_SYMBOLS[p] == 's' :
            vector[286 + counter_silver] = 1
            counter_silver += 1
        if PIECE_SYMBOLS[p] == 'g' :
            vector[290 + counter_gold] = 1
            counter_gold += 1
        if PIECE_SYMBOLS[p] == 'b' :
            vector[294 + counter_bishop] = 1
            counter_bishop += 1
        if PIECE_SYMBOLS[p] == 'r' :
            vector[296 + counter_rook] = 1
            counter_rook += 1


    order = {None : 9, 'p' : 1, 'l' : 2, 'n': 3, 's':4, 'g':5, 'r' :6, 'b' :7, 'k':8}

    for i in range(81):
        attacker = None
        defender = None

        attackers = board.attackers(0, i)
        defenders = board.attackers(1, i)

        for a in attackers:
            if order[str(board.piece_at(a)).lower()[-1]] < order[attacker]:
                attacker = str(board.piece_at(a)).lower()[-1]
                vector[298 + 3 * i] = (a % 9) / 8
                vector[299 + 3 * i] = (a // 9) / 8

        vector[300 + 3 * i] = len(attackers)

        for d in defenders:
            if order[str(board.piece_at(d)).lower()[-1]] < order[defender]:
                defender = str(board.piece_at(d)).lower()[-1]
                vector[541 + 3 * i] = (d % 9) / 8
                vector[542 + 3 * i] = (d // 9) / 8

        vector[543 + 3 * i] = len(defenders)

    return vector


# Tests
if __name__ == "__main__" :
    a = shogi.Board('lnsgk2nl/1r4gs1/p1pppp1pp/1p4p2/7P1/2P6/PP1PPPP1P/1SG4R1/LN2KGSNL b PLKSGBRb 10')
    t = time.time()
    b = board2vec(a)
    print("Successfully vectorized the board as input in", time.time()-t, "seconds !")