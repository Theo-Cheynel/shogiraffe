import shogi

base_test = """
#KIF version=2.0 encoding=UTF-8
# KIF形式棋譜ファイル
# Generated by Shogidokoro
手合割：平手
先手：Human
後手：Human
手数----指手---------消費時間--"""

chiffres = {1 : '１', 2 : '２', 3 : '３', 4 : '４', 5 : '５', 6 : '６', 7 : '７', 8 : '８', 9 : '９'}

kanji = {1 : '一', 2 : '二', 3 : '三', 4 : '四', 5 : '五', 6 : '六', 7 : '七', 8 : '八', 9 : '九'}

pieces = {'p' : '歩', 'l' : '香', 'n' : '桂', 's' : '銀', 'g':'金', 'b': '角', 'r' : '飛', 'k':'玉'}

promoted_pieces = {'p' : 'と', 'l' : '杏', 'n' : '圭', 's' : '全', 'b': '馬', 'r' : '龍'}

drop = "打"

promotion = "成"


def game2kif(game, filename):
    """
    Translates a game into an ugly .kif file readable by Shogidokoro.
    """

    # Start with an empty board
    b = shogi.Board()
    s = base_test

    # For each move
    for i in range(int(len(game)/4)) :

        # Begin a new line
        newline = "\n" + str(i+1).rjust(4) + " "
        coup = game[4*i:4*i+4]

        # If the move is a promotion
        if coup[3] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" :
            coup = coup[:3] + coup[3].lower() + "+"

        # If the move is a drop
        if coup[1] == "*" :
            newline += chiffres[int(coup[2])]
            newline += kanji["abcdefghi".index(coup[3])+1]
            newline += pieces[coup[0].lower()]
            newline += drop

        # Otherwise, it's a regular move
        else :
            newline += chiffres[int(coup[2])]
            newline += kanji["abcdefghi".index(coup[3])+1]
            piece = str(b.piece_at("abcdefghi".index(coup[1])*9+(9-int(coup[0]))))
            if piece[0] == '+' :
                newline += promoted_pieces[piece[-1].lower()]
            else :
                newline += pieces[piece[-1].lower()]

            if len(coup) == 5 :
                newline += promotion

            newline += "("
            newline += str(coup[0]) + str("abcdefghi".index(coup[1])+1)
            newline += ")"

        # Timer data is useless for what we want to do
        newline += "    (00:00 / 00:00:00)"

        s += newline

        b.push_usi(coup)


    # Open and write the data
    file = open(filename, mode='w+', encoding='utf-8')
    file.writelines(s)

    return i



def games2kif(nb_games):
    """
    Translates all the games in our database into .kif format.
    """

    # Open the database
    f = open('database.txt', encoding='utf-8')
    i = 0
    nb_boards = 0

    # Convert each game to .kif format
    for l in f.readlines():
        line = l.strip()
        nb_boards += game2kif(line, 'kif/'+str(i)+'.kif')
        i += 1
        if i == nb_games :
            break

        if i%50==0 :
            print("Number of processed games so far :", i, " (", nb_boards, " boards)")

    print("Number of processed games in total :", i, " (", nb_boards, " boards)")



if __name__=="__main__":
    games2kif(100000)