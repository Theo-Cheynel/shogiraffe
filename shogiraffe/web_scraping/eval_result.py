import shogi
import time
import glob

def read_kfk(filename) :
    """
    Reads the kfk file output fom Shogidokoro to parse the
    value of the evaluation function.
    """

    # Ouvrir le fichier
    f = open(filename, encoding='utf-8')
    contents = f.read()

    # Lire les scores
    result = contents[contents.find('<analysisScoreList>') + len('<analysisScoreList>'):contents.find('</analysisScoreList>')]
    result = result.split(',')
    for i in range(len(result)):
        if i%2 == 1:
            result[i] = int(result[i]) * -1
            result[i] = str(result[i])

    # Faire la liste de tous les coups
    coups = []
    indice = 0
    while contents.find('<move>', indice+1) != -1:
        indice = contents.find('<move>', indice+1)
        coups.append(contents[indice + len('<move>'):contents.find('</move>', indice)])
    coups = coups[1:]

    # Transformer la liste de coups en liste de planches
    boards = []
    board = shogi.Board()
    for coup in coups :
        coup = coup.split()[-1]
        if "*" in coup :
            board.push_usi(coup)
        elif coup[-1] == "+" :
            board.push_usi(coup[-6:-4]+coup[-3:])
        else :
            board.push_usi(coup[-5:-3]+coup[-2:])
        boards.append(board.sfen())

    # Ecrire dans le fichier de résultats
    f = open('result.txt', mode='at', encoding='utf-8')
    for i in range(len(result)):
        x = result[i]
        y = boards[i]
        f.write(x+" | "+y+"\n")
    f.close()


if __name__ == "__main__" :
    g = glob.glob("E:/Shogi/shogi/Code/web_scraping/kif/*/*.kfk")
    print("Nombre de parties trouvées :", len(g))
    for i in range(len(g)) :
        if i % (len(g)/20) < 1 :
            print(str(i)+" parties traitées sur "+str(len(g)))
        filename = g[i]
        read_kfk(filename)